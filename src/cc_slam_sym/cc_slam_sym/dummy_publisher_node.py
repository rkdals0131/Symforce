#!/usr/bin/env python3
"""
Dummy publisher node for testing CC-SLAM-SYM
Publishes simulated cone observations, IMU, and GPS data
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Header
from sensor_msgs.msg import Imu, NavSatFix
from geometry_msgs.msg import PoseStamped, TwistStamped
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import Path

import numpy as np
from typing import Dict, List, Tuple
import yaml

from .cone_definitions import GROUND_TRUTH_CONES_SCENARIO_1, GROUND_TRUTH_CONES_SCENARIO_2

class DummyPublisher(Node):
    def __init__(self):
        super().__init__('dummy_publisher')
        
        # Declare parameters
        self.declare_parameter('scenario', 1)
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('imu_rate', 100.0)
        self.declare_parameter('gps_rate', 8.0)
        self.declare_parameter('cone_detection_range', 15.0)
        self.declare_parameter('cone_fov_deg', 120.0)
        self.declare_parameter('vehicle_speed', 5.0)
        
        # Get parameters
        self.scenario = self.get_parameter('scenario').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.imu_rate = self.get_parameter('imu_rate').value
        self.gps_rate = self.get_parameter('gps_rate').value
        self.detection_range = self.get_parameter('cone_detection_range').value
        self.fov_rad = np.radians(self.get_parameter('cone_fov_deg').value)
        self.vehicle_speed = self.get_parameter('vehicle_speed').value
        
        # Load ground truth cones
        if self.scenario == 1:
            self.ground_truth_cones = GROUND_TRUTH_CONES_SCENARIO_1
            self.get_logger().info("Using Scenario 1: Straight track with AEB zone")
        else:
            self.ground_truth_cones = GROUND_TRUTH_CONES_SCENARIO_2
            self.get_logger().info("Using Scenario 2: Formula Student track")
        
        # Publishers
        self.cone_pub = self.create_publisher(MarkerArray, '/cones/clusters', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        self.gps_pub = self.create_publisher(NavSatFix, '/gps/fix', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/robot/pose', 10)
        self.twist_pub = self.create_publisher(TwistStamped, '/robot/twist', 10)
        self.path_pub = self.create_publisher(Path, '/robot/path', 10)
        self.gt_cones_pub = self.create_publisher(MarkerArray, '/cones/ground_truth', 10)
        
        # State
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.robot_vx = self.vehicle_speed
        self.robot_vy = 0.0
        self.robot_vtheta = 0.0
        
        # Path tracking
        self.path = Path()
        self.path.header.frame_id = "map"
        
        # Cone tracking state
        self.cone_track_ids = {}  # cone_id -> track_id
        self.next_track_id = 0
        self.lost_track_counter = {}  # track_id -> frames_lost
        
        # IMU bias (for realistic simulation)
        self.imu_accel_bias = np.random.normal(0, 0.01, 3)
        self.imu_gyro_bias = np.random.normal(0, 0.001, 3)
        
        # GPS origin (for UTM conversion)
        self.gps_origin_lat = 37.0
        self.gps_origin_lon = 127.0
        
        # Timers
        self.cone_timer = self.create_timer(1.0 / self.publish_rate, self.publish_cones)
        self.imu_timer = self.create_timer(1.0 / self.imu_rate, self.publish_imu)
        self.gps_timer = self.create_timer(1.0 / self.gps_rate, self.publish_gps)
        self.motion_timer = self.create_timer(0.01, self.update_motion)  # 100Hz motion update
        
        # Publish ground truth cones once
        self.publish_ground_truth_cones()
        
        self.get_logger().info("Dummy publisher initialized")
    
    def update_motion(self):
        """Update robot motion using simple path following"""
        dt = 0.01
        
        # Find centerline ahead
        look_ahead = 5.0
        target_x = self.robot_x + look_ahead * np.cos(self.robot_theta)
        target_y = self.robot_y + look_ahead * np.sin(self.robot_theta)
        
        # Find nearest cones to target point
        blue_cones = []
        yellow_cones = []
        
        for cone_id, cone_data in self.ground_truth_cones.items():
            dist = np.linalg.norm(cone_data['pos'] - np.array([target_x, target_y]))
            if dist < 10.0:  # Only consider nearby cones
                if cone_data['type'] == 'blue':
                    blue_cones.append(cone_data['pos'])
                elif cone_data['type'] == 'yellow':
                    yellow_cones.append(cone_data['pos'])
        
        # Calculate centerline
        if blue_cones and yellow_cones:
            # Find closest blue and yellow cones
            blue_cones = np.array(blue_cones)
            yellow_cones = np.array(yellow_cones)
            
            # Simple approach: average of nearest blue and yellow
            blue_dists = np.linalg.norm(blue_cones - np.array([target_x, target_y]), axis=1)
            yellow_dists = np.linalg.norm(yellow_cones - np.array([target_x, target_y]), axis=1)
            
            nearest_blue = blue_cones[np.argmin(blue_dists)]
            nearest_yellow = yellow_cones[np.argmin(yellow_dists)]
            
            centerline_point = (nearest_blue + nearest_yellow) / 2.0
            
            # Calculate heading error
            desired_heading = np.arctan2(
                centerline_point[1] - self.robot_y,
                centerline_point[0] - self.robot_x
            )
            heading_error = desired_heading - self.robot_theta
            
            # Normalize to [-pi, pi]
            while heading_error > np.pi:
                heading_error -= 2 * np.pi
            while heading_error < -np.pi:
                heading_error += 2 * np.pi
            
            # Simple P controller for steering
            kp = 0.5
            self.robot_vtheta = kp * heading_error
            
            # Limit angular velocity
            max_vtheta = 0.5
            self.robot_vtheta = np.clip(self.robot_vtheta, -max_vtheta, max_vtheta)
        
        # Update position
        self.robot_x += self.robot_vx * np.cos(self.robot_theta) * dt
        self.robot_y += self.robot_vx * np.sin(self.robot_theta) * dt
        self.robot_theta += self.robot_vtheta * dt
        
        # Normalize theta
        while self.robot_theta > np.pi:
            self.robot_theta -= 2 * np.pi
        while self.robot_theta < -np.pi:
            self.robot_theta += 2 * np.pi
        
        # Publish pose and twist
        self.publish_pose()
        self.publish_twist()
    
    def publish_pose(self):
        """Publish current robot pose"""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        
        msg.pose.position.x = self.robot_x
        msg.pose.position.y = self.robot_y
        msg.pose.position.z = 0.0
        
        # Convert theta to quaternion
        msg.pose.orientation.z = np.sin(self.robot_theta / 2.0)
        msg.pose.orientation.w = np.cos(self.robot_theta / 2.0)
        
        self.pose_pub.publish(msg)
        
        # Add to path
        self.path.poses.append(msg)
        self.path.header.stamp = msg.header.stamp
        self.path_pub.publish(self.path)
    
    def publish_twist(self):
        """Publish current robot velocity"""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        
        msg.twist.linear.x = self.robot_vx
        msg.twist.linear.y = self.robot_vy
        msg.twist.angular.z = self.robot_vtheta
        
        self.twist_pub.publish(msg)
    
    def publish_cones(self):
        """Publish visible cone observations"""
        visible_cones = self.get_visible_cones()
        
        # Update tracking
        self.update_tracking(visible_cones)
        
        # Create marker array
        marker_array = MarkerArray()
        
        for i, (cone_id, local_pos, color) in enumerate(visible_cones):
            # Get or assign track ID
            if cone_id not in self.cone_track_ids:
                self.cone_track_ids[cone_id] = self.next_track_id
                self.next_track_id += 1
            
            track_id = self.cone_track_ids[cone_id]
            
            # Create marker
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = "base_link"
            marker.ns = "cone_observations"
            marker.id = track_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            # Position in robot frame with noise
            noise = np.random.normal(0, 0.05, 2)
            marker.pose.position.x = local_pos[0] + noise[0]
            marker.pose.position.y = local_pos[1] + noise[1]
            marker.pose.position.z = 0.15
            
            # Size
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            
            # Color
            if color == 'yellow':
                marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0
            elif color == 'blue':
                marker.color.r, marker.color.g, marker.color.b = 0.0, 0.0, 1.0
            elif color == 'red':
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
            elif color == 'orange':
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.5, 0.0
            marker.color.a = 0.8
            
            marker_array.markers.append(marker)
        
        self.cone_pub.publish(marker_array)
    
    def get_visible_cones(self) -> List[Tuple[int, np.ndarray, str]]:
        """Get cones visible from current robot pose"""
        visible = []
        
        for cone_id, cone_data in self.ground_truth_cones.items():
            # Transform to robot frame
            dx = cone_data['pos'][0] - self.robot_x
            dy = cone_data['pos'][1] - self.robot_y
            
            # Rotate to robot frame
            local_x = dx * np.cos(-self.robot_theta) - dy * np.sin(-self.robot_theta)
            local_y = dx * np.sin(-self.robot_theta) + dy * np.cos(-self.robot_theta)
            
            # Check range
            dist = np.sqrt(local_x**2 + local_y**2)
            if dist > self.detection_range:
                continue
            
            # Check FOV
            angle = np.arctan2(local_y, local_x)
            if abs(angle) > self.fov_rad / 2:
                continue
            
            # Cone is visible
            visible.append((cone_id, np.array([local_x, local_y]), cone_data['type']))
        
        return visible
    
    def update_tracking(self, visible_cones):
        """Update cone tracking state"""
        visible_ids = {cone_id for cone_id, _, _ in visible_cones}
        
        # Update lost track counters
        for track_id in list(self.lost_track_counter.keys()):
            cone_id = None
            for cid, tid in self.cone_track_ids.items():
                if tid == track_id:
                    cone_id = cid
                    break
            
            if cone_id and cone_id in visible_ids:
                # Cone is visible again
                self.lost_track_counter.pop(track_id, None)
            else:
                # Increment lost counter
                self.lost_track_counter[track_id] = self.lost_track_counter.get(track_id, 0) + 1
                
                # Remove track if lost for too long
                if self.lost_track_counter[track_id] > 50:  # 2.5 seconds at 20Hz
                    self.lost_track_counter.pop(track_id)
                    if cone_id:
                        self.cone_track_ids.pop(cone_id, None)
    
    def publish_imu(self):
        """Publish IMU data"""
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "imu_link"
        
        # Linear acceleration (with gravity and bias)
        accel = np.array([0.0, 0.0, 9.81])  # Gravity
        
        # Add acceleration from motion
        # (simplified - in reality would need proper kinematics)
        accel[0] += self.robot_vtheta * self.robot_vy  # Centripetal
        
        # Add noise and bias
        accel += self.imu_accel_bias
        accel += np.random.normal(0, 0.01, 3)
        
        msg.linear_acceleration.x = accel[0]
        msg.linear_acceleration.y = accel[1]
        msg.linear_acceleration.z = accel[2]
        
        # Angular velocity
        gyro = np.array([0.0, 0.0, self.robot_vtheta])
        gyro += self.imu_gyro_bias
        gyro += np.random.normal(0, 0.001, 3)
        
        msg.angular_velocity.x = gyro[0]
        msg.angular_velocity.y = gyro[1]
        msg.angular_velocity.z = gyro[2]
        
        # Covariances
        msg.linear_acceleration_covariance = (np.eye(3) * 0.01).flatten().tolist()
        msg.angular_velocity_covariance = (np.eye(3) * 0.001).flatten().tolist()
        
        self.imu_pub.publish(msg)
    
    def publish_gps(self):
        """Publish GPS data"""
        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "gps_link"
        
        # Convert robot position to lat/lon
        # Simplified - in reality use proper geodetic conversion
        meters_per_deg_lat = 111319.5
        meters_per_deg_lon = meters_per_deg_lat * np.cos(np.radians(self.gps_origin_lat))
        
        msg.latitude = self.gps_origin_lat + self.robot_y / meters_per_deg_lat
        msg.longitude = self.gps_origin_lon + self.robot_x / meters_per_deg_lon
        msg.altitude = 100.0
        
        # Add RTK-level noise (2cm)
        msg.latitude += np.random.normal(0, 0.02 / meters_per_deg_lat)
        msg.longitude += np.random.normal(0, 0.02 / meters_per_deg_lon)
        
        # Status
        msg.status.status = 2  # STATUS_FIX
        msg.status.service = 1  # SERVICE_GPS
        
        # Position covariance (2cm std dev)
        msg.position_covariance_type = 2  # COVARIANCE_TYPE_DIAGONAL_KNOWN
        msg.position_covariance = [0.0004, 0.0, 0.0, 0.0, 0.0004, 0.0, 0.0, 0.0, 0.01]
        
        self.gps_pub.publish(msg)
    
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
            
            # Color
            if cone_data['type'] == 'yellow':
                marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0
            elif cone_data['type'] == 'blue':
                marker.color.r, marker.color.g, marker.color.b = 0.0, 0.0, 1.0
            elif cone_data['type'] == 'red':
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
            elif cone_data['type'] == 'orange':
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.5, 0.0
            marker.color.a = 0.5
            
            marker_array.markers.append(marker)
        
        self.gt_cones_pub.publish(marker_array)
        self.get_logger().info(f"Published {len(marker_array.markers)} ground truth cones")

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