#!/usr/bin/env python3
"""
Dummy publisher node for testing CC-SLAM-SYM
Publishes simulated cone observations, IMU, and GPS data
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Header
from sensor_msgs.msg import Imu, NavSatFix
from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped, TwistWithCovarianceStamped, Point
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import Path, Odometry
from custom_interface.msg import TrackedConeArray, TrackedCone
import tf2_ros
from tf_transformations import quaternion_from_euler

import numpy as np
from typing import Dict, List, Tuple
import yaml

from .cone_definitions import GROUND_TRUTH_CONES_SCENARIO_1, GROUND_TRUTH_CONES_SCENARIO_2

class DummyPublisher(Node):
    def __init__(self):
        super().__init__('dummy_publisher')
        
        # Declare parameters with hierarchical structure matching YAML
        self.declare_parameter('scenario.id', 1)
        self.declare_parameter('publish_rates.cones', 20.0)
        self.declare_parameter('publish_rates.imu', 100.0)
        self.declare_parameter('publish_rates.gps', 8.0)
        self.declare_parameter('vehicle.speed', 5.0)
        self.declare_parameter('sensors.cone_detection.max_range', 15.0)
        self.declare_parameter('sensors.cone_detection.fov_deg', 120.0)
        self.declare_parameter('sensors.cone_detection.roi_type', 'sector')
        self.declare_parameter('sensors.noise.position_stddev', 0.05)
        self.declare_parameter('sensors.noise.odom_position', 0.1)  # Increased from 0.02 to 0.1 (10cm)
        self.declare_parameter('sensors.noise.odom_angle', 0.05)  # Increased from 0.01 to 0.05 (2.9 degrees)
        
        # Get parameters
        self.scenario = self.get_parameter('scenario.id').value
        self.publish_rate = self.get_parameter('publish_rates.cones').value
        self.imu_rate = self.get_parameter('publish_rates.imu').value
        self.gps_rate = self.get_parameter('publish_rates.gps').value
        self.vehicle_speed = self.get_parameter('vehicle.speed').value
        self.detection_range = self.get_parameter('sensors.cone_detection.max_range').value
        self.fov_rad = np.radians(self.get_parameter('sensors.cone_detection.fov_deg').value)
        self.roi_type = self.get_parameter('sensors.cone_detection.roi_type').value
        
        # Get noise parameters
        self.cone_position_noise = self.get_parameter('sensors.noise.position_stddev').value
        self.odom_position_noise = self.get_parameter('sensors.noise.odom_position').value
        self.odom_angle_noise = self.get_parameter('sensors.noise.odom_angle').value
        
        # Load ground truth cones
        if self.scenario == 1:
            self.ground_truth_cones = GROUND_TRUTH_CONES_SCENARIO_1
            self.get_logger().info("Using Scenario 1: Straight track with AEB zone")
        else:
            self.ground_truth_cones = GROUND_TRUTH_CONES_SCENARIO_2
            self.get_logger().info("Using Scenario 2: Formula Student track")
        
        # Generate centerline from ground truth cones
        self.centerline_points = self.generate_centerline()
        self.centerline_index = 0
        
        # QoS for ground truth (TRANSIENT_LOCAL)
        gt_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Publishers (with PRD-compliant topic names)
        self.odom_pub = self.create_publisher(Odometry, '/odom_sim', 10)
        self.cone_pub = self.create_publisher(TrackedConeArray, '/fused_sorted_cones_ukf_sim', 10)
        self.imu_pub = self.create_publisher(Imu, '/ouster/imu_sim', 10)
        self.gps_pub = self.create_publisher(NavSatFix, '/ublox_gps_node/fix_sim', 10)
        self.gps_vel_pub = self.create_publisher(TwistWithCovarianceStamped, '/ublox_gps_node/fix_velocity_sim', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/robot/pose', 10)
        self.path_pub = self.create_publisher(Path, '/robot/path', 10)
        self.gt_cones_pub = self.create_publisher(MarkerArray, '/ground_truth_map_cones', gt_qos)
        self.roi_pub = self.create_publisher(Marker, '/roi_visualization', 10)
        self.centerline_pub = self.create_publisher(Marker, '/centerline_visualization', 10)
        # Detected cones visualization with latched QoS for initial message
        vis_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.detected_cones_vis_pub = self.create_publisher(MarkerArray, '/detected_cones_visualization', vis_qos)
        
        # TF Broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Ground truth robot state (perfect tracking)
        self.gt_x = 0.0
        self.gt_y = 0.0
        self.gt_theta = 0.0
        self.gt_vx = self.vehicle_speed
        self.gt_vy = 0.0
        self.gt_vtheta = 0.0
        
        # Noisy robot state (published as odom)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.robot_vx = self.vehicle_speed
        self.robot_vy = 0.0
        self.robot_vtheta = 0.0
        
        # Initialize positions
        if self.scenario == 1:
            # Start in the middle of the track
            self.gt_y = 0.0
            self.robot_y = 0.0
        else:
            # Scenario 2 start position from PRD
            self.gt_x = 30.0
            self.gt_y = 12.5
            self.robot_x = 30.0
            self.robot_y = 12.5
        
        # Path tracking
        self.path = Path()
        self.path.header.frame_id = "map"
        self.gt_path = Path()
        self.gt_path.header.frame_id = "map"
        
        # Cone tracking state - simple simulation matching CALICO
        self.cone_track_mapping = {}  # cone_id -> track_id (only for currently visible)
        self.next_track_id = 0  # Start from 0 like CALICO
        
        # Last detected cones for visualization
        self.last_detected_cones = []  # List of (track_id, local_pos, cone_type)
        
        # IMU bias (for realistic simulation)
        self.imu_accel_bias = np.random.normal(0, 0.01, 3)
        self.imu_gyro_bias = np.random.normal(0, 0.001, 3)
        
        # GPS origin (for UTM conversion)
        self.gps_origin_lat = 37.541383
        self.gps_origin_lon = 127.077763
        
        # Timers
        self.cone_timer = self.create_timer(1.0 / self.publish_rate, self.publish_cones)
        self.imu_timer = self.create_timer(1.0 / self.imu_rate, self.publish_imu)
        self.gps_timer = self.create_timer(1.0 / self.gps_rate, self.publish_gps)
        self.motion_timer = self.create_timer(0.01, self.update_motion)  # 100Hz motion update
        self.gt_timer = self.create_timer(1.0, self.publish_ground_truth_cones)  # 1Hz for ground truth
        self.roi_timer = self.create_timer(0.1, self.publish_roi)  # 10Hz for ROI
        self.centerline_timer = self.create_timer(1.0, self.publish_centerline)  # 1Hz for centerline
        
        # Publish initial visualizations immediately
        self.publish_ground_truth_cones()
        self.publish_centerline()
        self.publish_roi()  # Publish ROI immediately
        
        # Publish empty detected cones initially to establish the topic
        empty_markers = MarkerArray()
        delete_all = Marker()
        delete_all.header.stamp = self.get_clock().now().to_msg()
        delete_all.header.frame_id = "base_link"
        delete_all.ns = "detected_cones"
        delete_all.action = Marker.DELETEALL
        empty_markers.markers.append(delete_all)
        self.detected_cones_vis_pub.publish(empty_markers)
        
        self.get_logger().info("Dummy publisher initialized")
    
    def generate_centerline(self) -> List[Tuple[float, float]]:
        """Generate centerline points from ground truth cones"""
        centerline = []
        
        if self.scenario == 1:
            # Simple straight line for scenario 1
            for x in np.arange(0, 151, 1.0):
                centerline.append((x, 0.0))
        else:
            # For scenario 2, use proven waypoints from successful implementation
            base_waypoints = [
                (35.0, 12.5), (88.0, 12.5),
                (88.50, 12.00), (90.10, 6.72), (94.36, 3.22), (99.85, 2.68), (104.72, 5.28), 
                (107.32, 10.15), (106.78, 15.64), (103.28, 19.90), (98.00, 21.50),
                (95.0, 22.0), (80.0, 27.5),
                (75.0, 27.5), (70.0, 27.5), (65.0, 27.5), (60.0, 27.5), (55.0, 27.5),
                (50.0, 27.5), (45.0, 27.5), (40.0, 27.5), (35.0, 27.5), (30.0, 27.5),
                (27.6, 26.6), (12.0, 21.5),
                (12.00, 21.50), (6.72, 19.90), (3.22, 15.64), 
                (2.68, 10.15), (5.28, 5.28), (10.15, 2.68), (15.64, 3.22), (19.90, 6.72), (21.50, 12.00),
                (23.0, 12.5), (30.0, 12.5),
            ]
            
            # Interpolate between waypoints for smoother path
            for i in range(len(base_waypoints)):
                current = base_waypoints[i]
                next_wp = base_waypoints[(i + 1) % len(base_waypoints)]
                
                # Add current waypoint
                centerline.append(current)
                
                # Interpolate between current and next
                dx = next_wp[0] - current[0]
                dy = next_wp[1] - current[1]
                distance = np.sqrt(dx**2 + dy**2)
                
                # Add interpolated points every 0.5 meters
                num_interp = int(distance / 0.5)
                for j in range(1, num_interp):
                    t = j / float(num_interp)
                    x = current[0] + t * dx
                    y = current[1] + t * dy
                    centerline.append((x, y))
        
        return centerline
    
    def update_motion(self):
        """Update robot motion - ground truth follows centerline perfectly"""
        dt = 0.01
        
        # Update ground truth robot (perfect centerline following)
        if self.centerline_index < len(self.centerline_points) - 1:
            # Get next centerline point
            next_point = self.centerline_points[self.centerline_index + 1]
            
            # Calculate distance to next point
            dx = next_point[0] - self.gt_x
            dy = next_point[1] - self.gt_y
            dist = np.sqrt(dx*dx + dy*dy)
            
            # Move to next point if close enough
            if dist < 0.5:
                self.centerline_index += 1
                if self.centerline_index < len(self.centerline_points) - 1:
                    next_point = self.centerline_points[self.centerline_index + 1]
                    dx = next_point[0] - self.gt_x
                    dy = next_point[1] - self.gt_y
                    dist = np.sqrt(dx*dx + dy*dy)
            
            # Calculate perfect heading to next point
            if dist > 0.01:
                self.gt_theta = np.arctan2(dy, dx)
                
                # Perfect velocity towards next point
                self.gt_vx = self.vehicle_speed
                self.gt_vy = 0.0
                self.gt_vtheta = 0.0
                
                # Update ground truth position
                self.gt_x += self.gt_vx * np.cos(self.gt_theta) * dt
                self.gt_y += self.gt_vx * np.sin(self.gt_theta) * dt
        else:
            # Stop at end
            self.gt_vx = 0.0
            self.gt_vy = 0.0
            self.gt_vtheta = 0.0
        
        # Update noisy robot state (odom_sim = ground_truth + noise)
        # Position noise
        self.robot_x = self.gt_x + np.random.normal(0, self.odom_position_noise)
        self.robot_y = self.gt_y + np.random.normal(0, self.odom_position_noise)
        self.robot_theta = self.gt_theta + np.random.normal(0, self.odom_angle_noise)
        
        # Velocity noise
        self.robot_vx = self.gt_vx + np.random.normal(0, 0.05)
        self.robot_vy = self.gt_vy + np.random.normal(0, 0.02)
        self.robot_vtheta = self.gt_vtheta + np.random.normal(0, 0.01)
        
        # Normalize angles
        while self.robot_theta > np.pi:
            self.robot_theta -= 2 * np.pi
        while self.robot_theta < -np.pi:
            self.robot_theta += 2 * np.pi
        while self.gt_theta > np.pi:
            self.gt_theta -= 2 * np.pi
        while self.gt_theta < -np.pi:
            self.gt_theta += 2 * np.pi
        
        # Publish pose, odometry, and TF
        self.publish_pose()
        self.publish_odometry()
        self.publish_tf()
    
    def publish_tf(self):
        """Publish TF transforms including ground truth"""
        now = self.get_clock().now().to_msg()
        
        # Map -> Ground_truth_odom (identity - ground truth has no drift)
        map_to_gt_odom = TransformStamped()
        map_to_gt_odom.header.stamp = now
        map_to_gt_odom.header.frame_id = "map"
        map_to_gt_odom.child_frame_id = "ground_truth_odom"
        map_to_gt_odom.transform.rotation.w = 1.0
        
        # Ground_truth_odom -> Ground_truth_base_link
        gt_odom_to_gt_base = TransformStamped()
        gt_odom_to_gt_base.header.stamp = now
        gt_odom_to_gt_base.header.frame_id = "ground_truth_odom"
        gt_odom_to_gt_base.child_frame_id = "ground_truth_base_link"
        
        gt_odom_to_gt_base.transform.translation.x = self.gt_x
        gt_odom_to_gt_base.transform.translation.y = self.gt_y
        gt_odom_to_gt_base.transform.translation.z = 0.0
        
        q_gt = quaternion_from_euler(0, 0, self.gt_theta)
        gt_odom_to_gt_base.transform.rotation.x = q_gt[0]
        gt_odom_to_gt_base.transform.rotation.y = q_gt[1]
        gt_odom_to_gt_base.transform.rotation.z = q_gt[2]
        gt_odom_to_gt_base.transform.rotation.w = q_gt[3]
        
        # Map -> Odom transform (identity for now - SLAM will update this)
        map_to_odom = TransformStamped()
        map_to_odom.header.stamp = now
        map_to_odom.header.frame_id = "map"
        map_to_odom.child_frame_id = "odom"
        map_to_odom.transform.rotation.w = 1.0
        
        # Odom -> Base_link transform (noisy)
        odom_to_base = TransformStamped()
        odom_to_base.header.stamp = now
        odom_to_base.header.frame_id = "odom"
        odom_to_base.child_frame_id = "base_link"
        
        odom_to_base.transform.translation.x = self.robot_x
        odom_to_base.transform.translation.y = self.robot_y
        odom_to_base.transform.translation.z = 0.0
        
        q = quaternion_from_euler(0, 0, self.robot_theta)
        odom_to_base.transform.rotation.x = q[0]
        odom_to_base.transform.rotation.y = q[1]
        odom_to_base.transform.rotation.z = q[2]
        odom_to_base.transform.rotation.w = q[3]
        
        # Base_link -> IMU_link
        base_to_imu = TransformStamped()
        base_to_imu.header.stamp = now
        base_to_imu.header.frame_id = "base_link"
        base_to_imu.child_frame_id = "imu_link"
        base_to_imu.transform.rotation.w = 1.0
        
        # Base_link -> GPS_link
        base_to_gps = TransformStamped()
        base_to_gps.header.stamp = now
        base_to_gps.header.frame_id = "base_link"
        base_to_gps.child_frame_id = "gps_link"
        base_to_gps.transform.rotation.w = 1.0
        
        # Broadcast all transforms
        self.tf_broadcaster.sendTransform([
            map_to_gt_odom, gt_odom_to_gt_base,
            map_to_odom, odom_to_base, 
            base_to_imu, base_to_gps
        ])
    
    def publish_pose(self):
        """Publish current robot pose (noisy)"""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        
        msg.pose.position.x = self.robot_x
        msg.pose.position.y = self.robot_y
        msg.pose.position.z = 0.0
        
        # Convert theta to quaternion
        q = quaternion_from_euler(0, 0, self.robot_theta)
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        
        self.pose_pub.publish(msg)
        
        # Add to path
        self.path.poses.append(msg)
        if len(self.path.poses) > 1000:  # Limit path length
            self.path.poses = self.path.poses[-1000:]
        self.path.header.stamp = msg.header.stamp
        self.path_pub.publish(self.path)
        
        # Also track ground truth path
        gt_msg = PoseStamped()
        gt_msg.header = msg.header
        gt_msg.pose.position.x = self.gt_x
        gt_msg.pose.position.y = self.gt_y
        gt_msg.pose.position.z = 0.0
        q_gt = quaternion_from_euler(0, 0, self.gt_theta)
        gt_msg.pose.orientation.x = q_gt[0]
        gt_msg.pose.orientation.y = q_gt[1]
        gt_msg.pose.orientation.z = q_gt[2]
        gt_msg.pose.orientation.w = q_gt[3]
        self.gt_path.poses.append(gt_msg)
        if len(self.gt_path.poses) > 1000:
            self.gt_path.poses = self.gt_path.poses[-1000:]
    
    def publish_odometry(self):
        """Publish odometry message (noisy)"""
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        
        # Position
        msg.pose.pose.position.x = self.robot_x
        msg.pose.pose.position.y = self.robot_y
        msg.pose.pose.position.z = 0.0
        
        # Orientation
        q = quaternion_from_euler(0, 0, self.robot_theta)
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]
        
        # Velocity in robot frame
        msg.twist.twist.linear.x = self.robot_vx
        msg.twist.twist.linear.y = self.robot_vy
        msg.twist.twist.angular.z = self.robot_vtheta
        
        # Covariances (diagonal)
        # Position covariance
        for i in range(36):
            if i == 0 or i == 7:  # x, y
                msg.pose.covariance[i] = 0.01
            elif i == 35:  # theta
                msg.pose.covariance[i] = 0.05
            else:
                msg.pose.covariance[i] = 0.0
        
        # Velocity covariance
        for i in range(36):
            if i == 0:  # vx
                msg.twist.covariance[i] = 0.02
            elif i == 7:  # vy
                msg.twist.covariance[i] = 0.01
            elif i == 35:  # vtheta
                msg.twist.covariance[i] = 0.01
            else:
                msg.twist.covariance[i] = 0.0
        
        self.odom_pub.publish(msg)
    
    def publish_cones(self):
        """Publish visible cone observations as TrackedConeArray - Simple simulation"""
        # Get cones visible from ground truth position
        visible_cones = self.get_visible_cones()
        
        # Create TrackedConeArray message
        msg = TrackedConeArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"  # Changed from os_sensor to base_link for our simulation
        
        # Store for visualization
        self.last_detected_cones = []
        
        # Track which cones are currently visible
        current_visible_ids = {cone_id for cone_id, _, _ in visible_cones}
        
        # Remove cones that are no longer visible
        to_remove = [cid for cid in self.cone_track_mapping.keys() if cid not in current_visible_ids]
        for cone_id in to_remove:
            del self.cone_track_mapping[cone_id]
        
        for cone_id, local_pos_gt, cone_type in visible_cones:
            # If cone is newly visible, assign new track ID
            if cone_id not in self.cone_track_mapping:
                self.cone_track_mapping[cone_id] = self.next_track_id
                self.next_track_id += 1
            
            track_id = self.cone_track_mapping[cone_id]
            
            # Create TrackedCone
            tracked_cone = TrackedCone()
            tracked_cone.track_id = track_id
            
            # Add noise to ground truth observation
            noise = np.random.normal(0, self.cone_position_noise, 3)
            tracked_cone.position.x = local_pos_gt[0] + noise[0]
            tracked_cone.position.y = local_pos_gt[1] + noise[1]
            tracked_cone.position.z = -0.15 + noise[2]  # Cone height with noise
            
            # Set color with proper format (matching CALICO output)
            if cone_type == 'yellow':
                tracked_cone.color = "Yellow cone"
            elif cone_type == 'blue':
                tracked_cone.color = "Blue cone"
            elif cone_type == 'red':
                tracked_cone.color = "Red cone"
            elif cone_type == 'orange':
                tracked_cone.color = "Orange cone"
            else:
                tracked_cone.color = "Unknown"
            
            msg.cones.append(tracked_cone)
            
            # Store for visualization
            noisy_pos = np.array([local_pos_gt[0] + noise[0], local_pos_gt[1] + noise[1]])
            self.last_detected_cones.append((track_id, noisy_pos, cone_type))
        
        self.cone_pub.publish(msg)
        
        # Also publish visualization of detected cones
        self.publish_detected_cones_visualization()
    
    def get_visible_cones(self) -> List[Tuple[int, np.ndarray, str]]:
        """Get cones visible from ground truth robot pose"""
        visible = []
        
        for cone_id, cone_data in self.ground_truth_cones.items():
            # Transform to ground truth robot frame
            dx = cone_data['pos'][0] - self.gt_x
            dy = cone_data['pos'][1] - self.gt_y
            
            # Rotate to robot frame
            local_x = dx * np.cos(-self.gt_theta) - dy * np.sin(-self.gt_theta)
            local_y = dx * np.sin(-self.gt_theta) + dy * np.cos(-self.gt_theta)
            
            # Check if in ROI
            if self.is_in_roi(local_x, local_y):
                visible.append((cone_id, np.array([local_x, local_y]), cone_data['type']))
        
        return visible
    
    def is_in_roi(self, x: float, y: float) -> bool:
        """Check if point is in Region of Interest"""
        if self.roi_type == 'sector':
            # Sector (fan) shape
            dist = np.sqrt(x*x + y*y)
            if dist > self.detection_range:
                return False
            
            angle = np.arctan2(y, x)
            if abs(angle) > self.fov_rad / 2:
                return False
            
            return True
        else:
            # Rectangle shape
            if x < 0 or x > self.detection_range:
                return False
            
            max_y = self.detection_range * np.tan(self.fov_rad / 2)
            if abs(y) > max_y:
                return False
            
            return True
    
    def publish_roi(self):
        """Publish ROI visualization"""
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "ground_truth_base_link"  # Attach to ground truth frame
        marker.ns = "roi"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Red color for ROI
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        marker.scale.x = 0.1  # Line width
        marker.pose.orientation.w = 1.0  # Ensure proper orientation
        
        # Generate ROI boundary points
        if self.roi_type == 'sector':
            # Sector shape
            points = []
            
            # Origin
            p = Point()
            p.x = 0.0
            p.y = 0.0
            p.z = 0.0
            points.append(p)
            
            # Arc points
            num_arc_points = 20
            for i in range(num_arc_points + 1):
                angle = -self.fov_rad/2 + i * self.fov_rad / num_arc_points
                p = Point()
                p.x = self.detection_range * np.cos(angle)
                p.y = self.detection_range * np.sin(angle)
                p.z = 0.0
                points.append(p)
            
            # Back to origin
            p = Point()
            p.x = 0.0
            p.y = 0.0
            p.z = 0.0
            points.append(p)
            
        else:
            # Rectangle shape
            max_y = self.detection_range * np.tan(self.fov_rad / 2)
            
            # Rectangle corners
            corners = [
                (0, -max_y),
                (self.detection_range, -max_y),
                (self.detection_range, max_y),
                (0, max_y),
                (0, -max_y)
            ]
            
            points = []
            for x, y in corners:
                p = Point()
                p.x = x
                p.y = y
                p.z = 0.0
                points.append(p)
        
        marker.points = points
        self.roi_pub.publish(marker)
    
    def publish_centerline(self):
        """Publish centerline visualization"""
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "map"
        marker.ns = "centerline"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Green color for centerline
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        marker.scale.x = 0.2  # Line width
        
        # Add centerline points
        for x, y in self.centerline_points:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0
            marker.points.append(p)
        
        self.centerline_pub.publish(marker)
    
    # Remove the complex tracking function - no longer needed
    
    def publish_imu(self):
        """Publish IMU data"""
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "imu_link"
        
        # Linear acceleration (with gravity and bias)
        accel = np.array([0.0, 0.0, 9.81])  # Gravity
        
        # Add acceleration from motion (using ground truth for physics)
        # (simplified - in reality would need proper kinematics)
        accel[0] += self.gt_vtheta * self.gt_vy  # Centripetal
        
        # Add noise and bias
        accel += self.imu_accel_bias
        accel += np.random.normal(0, 0.01, 3)
        
        msg.linear_acceleration.x = accel[0]
        msg.linear_acceleration.y = accel[1]
        msg.linear_acceleration.z = accel[2]
        
        # Angular velocity (from ground truth + noise)
        gyro = np.array([0.0, 0.0, self.gt_vtheta])
        gyro += self.imu_gyro_bias
        gyro += np.random.normal(0, 0.001, 3)
        
        msg.angular_velocity.x = gyro[0]
        msg.angular_velocity.y = gyro[1]
        msg.angular_velocity.z = gyro[2]
        
        # Covariances (must assign element by element)
        # Linear acceleration covariance
        for i in range(9):
            if i in [0, 4, 8]:  # Diagonal elements
                msg.linear_acceleration_covariance[i] = 0.01
            else:
                msg.linear_acceleration_covariance[i] = 0.0
        
        # Angular velocity covariance
        for i in range(9):
            if i in [0, 4, 8]:  # Diagonal elements
                msg.angular_velocity_covariance[i] = 0.001
            else:
                msg.angular_velocity_covariance[i] = 0.0
        
        # Orientation not provided
        msg.orientation_covariance[0] = -1.0
        
        self.imu_pub.publish(msg)
    
    def publish_gps(self):
        """Publish GPS data"""
        # Position fix (based on noisy robot position)
        fix_msg = NavSatFix()
        fix_msg.header.stamp = self.get_clock().now().to_msg()
        fix_msg.header.frame_id = "gps_link"
        
        # Convert robot position to lat/lon
        # Simplified - in reality use proper geodetic conversion
        meters_per_deg_lat = 111319.5
        meters_per_deg_lon = meters_per_deg_lat * np.cos(np.radians(self.gps_origin_lat))
        
        fix_msg.latitude = self.gps_origin_lat + self.robot_y / meters_per_deg_lat
        fix_msg.longitude = self.gps_origin_lon + self.robot_x / meters_per_deg_lon
        fix_msg.altitude = 100.0
        
        # Add RTK-level noise (2cm)
        fix_msg.latitude += np.random.normal(0, 0.02 / meters_per_deg_lat)
        fix_msg.longitude += np.random.normal(0, 0.02 / meters_per_deg_lon)
        
        # Status
        fix_msg.status.status = 2  # STATUS_FIX
        fix_msg.status.service = 1  # SERVICE_GPS
        
        # Position covariance (2cm std dev)
        fix_msg.position_covariance_type = 2  # COVARIANCE_TYPE_DIAGONAL_KNOWN
        for i in range(9):
            if i == 0 or i == 4:  # x, y variance
                fix_msg.position_covariance[i] = 0.0004
            elif i == 8:  # z variance
                fix_msg.position_covariance[i] = 0.01
            else:
                fix_msg.position_covariance[i] = 0.0
        
        self.gps_pub.publish(fix_msg)
        
        # Velocity
        vel_msg = TwistWithCovarianceStamped()
        vel_msg.header = fix_msg.header
        
        # Convert robot velocity to ENU frame
        vel_msg.twist.twist.linear.x = self.robot_vx * np.cos(self.robot_theta) - self.robot_vy * np.sin(self.robot_theta)
        vel_msg.twist.twist.linear.y = self.robot_vx * np.sin(self.robot_theta) + self.robot_vy * np.cos(self.robot_theta)
        vel_msg.twist.twist.linear.z = 0.0
        
        # Velocity covariance (8.5cm std dev as per PRD)
        for i in range(36):
            if i == 0 or i == 7:  # vx, vy
                vel_msg.twist.covariance[i] = 0.007225
            elif i == 14:  # vz
                vel_msg.twist.covariance[i] = 0.007225
            elif i in [21, 28, 35]:  # angular not provided
                vel_msg.twist.covariance[i] = -1.0
            else:
                vel_msg.twist.covariance[i] = 0.0
        
        self.gps_vel_pub.publish(vel_msg)
    
    def publish_ground_truth_cones(self):
        """Publish all ground truth cones for visualization"""
        marker_array = MarkerArray()
        
        for cone_id, cone_data in self.ground_truth_cones.items():
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = "map"
            marker.ns = "ground_truth_cones"
            marker.id = cone_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            marker.pose.position.x = cone_data['pos'][0]
            marker.pose.position.y = cone_data['pos'][1]
            marker.pose.position.z = 0.15
            
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            
            # Semi-transparent gray for all ground truth cones
            marker.color.r = 0.5
            marker.color.g = 0.5
            marker.color.b = 0.5
            marker.color.a = 0.3
            
            # Lifetime = 0 means permanent
            marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
            
            marker_array.markers.append(marker)
        
        # Also add a DELETE_ALL marker first to clear old markers
        delete_marker = Marker()
        delete_marker.header = marker_array.markers[0].header if marker_array.markers else Header()
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.header.frame_id = "map"
        delete_marker.ns = "ground_truth_cones"
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.insert(0, delete_marker)
        
        self.gt_cones_pub.publish(marker_array)
        self.get_logger().info(f"Published {len(marker_array.markers)} ground truth cones")
    
    def publish_detected_cones_visualization(self):
        """Publish visualization of detected cones in the robot's frame"""
        marker_array = MarkerArray()
        
        # First, clear all previous markers
        delete_all = Marker()
        delete_all.header.stamp = self.get_clock().now().to_msg()
        delete_all.header.frame_id = "base_link"
        delete_all.ns = "detected_cones"
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)
        
        # Also clear text markers
        delete_text = Marker()
        delete_text.header.stamp = self.get_clock().now().to_msg()
        delete_text.header.frame_id = "base_link"
        delete_text.ns = "detected_cones_text"
        delete_text.action = Marker.DELETEALL
        marker_array.markers.append(delete_text)
        
        # Then add current visible cones
        for idx, (track_id, local_pos, cone_type) in enumerate(self.last_detected_cones):
            # Cone cylinder marker
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = "base_link"
            marker.ns = "detected_cones"
            marker.id = idx * 2  # Even IDs for cylinders
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            # Position already has noise from tracking
            marker.pose.position.x = local_pos[0]
            marker.pose.position.y = local_pos[1]
            marker.pose.position.z = 0.15
            
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            
            # Color based on type
            if cone_type == 'yellow':
                marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0
            elif cone_type == 'blue':
                marker.color.r, marker.color.g, marker.color.b = 0.0, 0.0, 1.0
            elif cone_type == 'red':
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
            elif cone_type == 'orange':
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.5, 0.0
            marker.color.a = 0.8
            
            # No lifetime - markers stay until explicitly deleted
            marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
            
            marker_array.markers.append(marker)
            
            # Track ID text marker
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = "detected_cones_text"
            text_marker.id = idx * 2 + 1  # Odd IDs for text
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            # Position text above cone
            text_marker.pose.position.x = marker.pose.position.x
            text_marker.pose.position.y = marker.pose.position.y
            text_marker.pose.position.z = 0.5  # Above the cone
            text_marker.pose.orientation.w = 1.0
            
            # Display track ID
            text_marker.text = str(track_id)
            
            text_marker.scale.z = 0.15  # Text height
            
            # White text
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            text_marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
            
            marker_array.markers.append(text_marker)
        
        self.detected_cones_vis_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = DummyPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()