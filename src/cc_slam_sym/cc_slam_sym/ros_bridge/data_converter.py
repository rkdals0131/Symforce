#!/usr/bin/env python3
"""
Data conversion utilities between ROS2 messages and SLAM data structures
Following GLIM's clean separation of ROS dependencies
"""

import numpy as np
from typing import List, Optional, Tuple
from geometry_msgs.msg import Pose, Point, Quaternion, Transform, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
import tf2_geometry_msgs
from rclpy.time import Time

# Import custom messages
from custom_interface.msg import TrackedCone, TrackedConeArray

from ..utils.data_structures import (
    ConeCluster, OdometryData, Keyframe, Landmark
)


class RosSlamConverter:
    """Convert between ROS2 messages and SLAM data structures"""
    
    @staticmethod
    def cone_array_to_clusters(msg: TrackedConeArray) -> List[ConeCluster]:
        """Convert TrackedConeArray message to ConeCluster list
        
        Args:
            msg: TrackedConeArray message
            
        Returns:
            List of ConeCluster objects
        """
        clusters = []
        
        for cone in msg.cones:
            # Extract position
            position = np.array([
                cone.position.x,
                cone.position.y,
                cone.position.z
            ])
            
            # Extract color from string format
            # Dummy publisher uses: "Yellow cone", "Blue cone", "Red cone", "Unknown"
            color_str = cone.color.lower()
            if "yellow" in color_str:
                color = "yellow"
            elif "blue" in color_str:
                color = "blue"
            elif "red" in color_str:
                color = "red"
            else:
                color = "unknown"
            
            # Extract track ID
            track_id = cone.track_id
            
            # Default confidence (not provided in TrackedCone)
            confidence = 0.8
            
            # Create cluster
            cluster = ConeCluster(
                timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                position=position,
                color=color,
                confidence=confidence,
                track_id=track_id
            )
            clusters.append(cluster)
            
        return clusters
    
    @staticmethod
    def odometry_to_data(msg: Odometry) -> OdometryData:
        """Convert Odometry message to OdometryData
        
        Args:
            msg: Odometry message
            
        Returns:
            OdometryData object
        """
        # Extract position (2D for SLAM)
        position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ])
        
        # Extract orientation (convert quaternion to yaw)
        q = msg.pose.pose.orientation
        yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        
        # Extract linear velocity (2D)
        linear_velocity = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y
        ])
        
        # Extract angular velocity (just z component)
        angular_velocity = msg.twist.twist.angular.z
        
        # Create odometry data
        odom_data = OdometryData(
            timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            position=position,
            orientation=yaw,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity
        )
        
        return odom_data
    
    @staticmethod
    def pose2_to_transform(pose, timestamp: Time, 
                          child_frame: str, parent_frame: str) -> TransformStamped:
        """Convert GTSAM Pose2 to ROS TransformStamped
        
        Args:
            pose: GTSAM Pose2 object
            timestamp: ROS timestamp
            child_frame: Child frame ID
            parent_frame: Parent frame ID
            
        Returns:
            TransformStamped message
        """
        t = TransformStamped()
        
        # Header
        t.header.stamp = timestamp.to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        
        # Translation
        t.transform.translation.x = pose.x()
        t.transform.translation.y = pose.y()
        t.transform.translation.z = 0.0
        
        # Rotation (yaw to quaternion)
        yaw = pose.theta()
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = np.sin(yaw / 2.0)
        t.transform.rotation.w = np.cos(yaw / 2.0)
        
        return t
    
    @staticmethod
    def landmarks_to_pointcloud(landmarks: List[Landmark], 
                               timestamp: Time,
                               frame_id: str = "map") -> PointCloud2:
        """Convert landmarks to PointCloud2 message
        
        Args:
            landmarks: List of Landmark objects
            timestamp: ROS timestamp
            frame_id: Frame ID
            
        Returns:
            PointCloud2 message
        """
        # Create points array
        points = []
        for landmark in landmarks:
            # x, y, z, intensity (color encoded)
            color_intensity = {
                "yellow": 1.0,
                "blue": 2.0,
                "red": 3.0,
                "unknown": 0.0
            }
            intensity = color_intensity.get(landmark.color, 0.0)
            
            points.append([
                landmark.position[0],
                landmark.position[1],
                0.0,  # z
                intensity
            ])
        
        if not points:
            # Return empty pointcloud
            msg = PointCloud2()
            msg.header.stamp = timestamp.to_msg()
            msg.header.frame_id = frame_id
            return msg
        
        # Convert to numpy array
        points_np = np.array(points, dtype=np.float32)
        
        # Create PointCloud2 message
        msg = PointCloud2()
        msg.header.stamp = timestamp.to_msg()
        msg.header.frame_id = frame_id
        
        # Set dimensions
        msg.height = 1
        msg.width = len(points)
        msg.is_dense = True
        msg.is_bigendian = False
        
        # Define fields
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        
        msg.point_step = 16  # 4 fields * 4 bytes
        msg.row_step = msg.point_step * msg.width
        msg.data = points_np.tobytes()
        
        return msg
    
    @staticmethod
    def create_path_message(poses: List[Tuple[float, object]], 
                           timestamp: Time,
                           frame_id: str = "map"):
        """Create Path message from poses
        
        Args:
            poses: List of (timestamp, Pose2) tuples
            timestamp: Current ROS timestamp
            frame_id: Frame ID
            
        Returns:
            Path message
        """
        from nav_msgs.msg import Path
        from geometry_msgs.msg import PoseStamped
        
        path = Path()
        path.header.stamp = timestamp.to_msg()
        path.header.frame_id = frame_id
        
        for ts, pose in poses:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = frame_id
            pose_stamped.header.stamp = Time(seconds=int(ts), nanoseconds=int((ts % 1) * 1e9)).to_msg()
            
            # Position
            pose_stamped.pose.position.x = pose.x()
            pose_stamped.pose.position.y = pose.y()
            pose_stamped.pose.position.z = 0.0
            
            # Orientation (yaw to quaternion)
            yaw = pose.theta()
            pose_stamped.pose.orientation.x = 0.0
            pose_stamped.pose.orientation.y = 0.0
            pose_stamped.pose.orientation.z = np.sin(yaw / 2.0)
            pose_stamped.pose.orientation.w = np.cos(yaw / 2.0)
            
            path.poses.append(pose_stamped)
            
        return path


class TfPublisher:
    """Helper class for publishing TF transforms"""
    
    def __init__(self, tf_broadcaster: TransformBroadcaster):
        """Initialize TF publisher
        
        Args:
            tf_broadcaster: ROS2 TF broadcaster
        """
        self.tf_broadcaster = tf_broadcaster
        
    def publish_map_to_odom(self, pose, timestamp: Time):
        """Publish map->odom transform
        
        Args:
            pose: GTSAM Pose2 representing map->odom transform
            timestamp: ROS timestamp
        """
        transform = RosSlamConverter.pose2_to_transform(
            pose, timestamp, "odom", "map"
        )
        self.tf_broadcaster.sendTransform(transform)
        
    def publish_keyframe_poses(self, keyframes: List[Keyframe], 
                              optimized_poses: dict,
                              timestamp: Time):
        """Publish transforms for all keyframes
        
        Args:
            keyframes: List of keyframes
            optimized_poses: Dict of keyframe_id -> optimized pose
            timestamp: Current timestamp
        """
        transforms = []
        
        for kf in keyframes:
            if kf.id in optimized_poses:
                pose = optimized_poses[kf.id]
                transform = RosSlamConverter.pose2_to_transform(
                    pose, timestamp,
                    f"keyframe_{kf.id}", "map"
                )
                transforms.append(transform)
                
        if transforms:
            self.tf_broadcaster.sendTransform(transforms)


# Import PointField if available
try:
    from sensor_msgs.msg import PointField
except ImportError:
    # Define a simple PointField class if not available
    class PointField:
        FLOAT32 = 7
        
        def __init__(self, name, offset, datatype, count):
            self.name = name
            self.offset = offset
            self.datatype = datatype
            self.count = count