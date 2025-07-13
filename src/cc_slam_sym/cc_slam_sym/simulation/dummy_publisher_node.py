#!/usr/bin/env python3
"""
Dummy publisher node for testing CC-SLAM-SYM
Publishes simulated cone observations, IMU, and GPS data

This is a refactored version using modular components:
- sensor_simulator: Handles IMU, GPS, and odometry simulation
- motion_controller: Manages vehicle motion along trajectories
- visualization utils: Provides reusable visualization functions
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

# Import modular components
from .cone_definitions import GROUND_TRUTH_CONES_SCENARIO_1, GROUND_TRUTH_CONES_SCENARIO_2
from .sensor_simulator import SensorSimulator, SensorNoiseConfig
from .motion_controller import MotionController, MotionScenario, VehicleState
from ..utils.visualization import VisualizationHelper, publish_cone_array, publish_detected_cones


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
        self.declare_parameter('sensors.noise.odom_position', 0.1)
        self.declare_parameter('sensors.noise.odom_angle', 0.05)
        
        # Drift and bias parameters
        self.declare_parameter('sensors.noise.imu_accel_bias_drift', 0.02)
        self.declare_parameter('sensors.noise.imu_gyro_bias_drift', 0.001)
        self.declare_parameter('sensors.noise.imu_accel_white_noise', 0.01)
        self.declare_parameter('sensors.noise.imu_gyro_white_noise', 0.0017)
        self.declare_parameter('sensors.noise.odom_drift_rate_linear', 0.02)
        self.declare_parameter('sensors.noise.odom_drift_rate_angular', 0.01)
        
        # Detection error parameters
        self.declare_parameter('sensors.detection_errors.enable', True)
        self.declare_parameter('sensors.detection_errors.false_negative_rate', 0.07)
        self.declare_parameter('sensors.detection_errors.false_positive_rate', 0.002)
        self.declare_parameter('sensors.detection_errors.wrong_color_rate', 0.002)
        self.declare_parameter('sensors.detection_errors.unknown_color_rate', 0.08)
        
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
        
        # Create sensor noise configuration
        noise_config = SensorNoiseConfig(
            imu_accel_bias_drift=self.get_parameter('sensors.noise.imu_accel_bias_drift').value,
            imu_gyro_bias_drift=self.get_parameter('sensors.noise.imu_gyro_bias_drift').value,
            imu_accel_white_noise=self.get_parameter('sensors.noise.imu_accel_white_noise').value,
            imu_gyro_white_noise=self.get_parameter('sensors.noise.imu_gyro_white_noise').value,
            odom_position_noise=self.get_parameter('sensors.noise.odom_position').value,
            odom_angle_noise=self.get_parameter('sensors.noise.odom_angle').value,
            odom_drift_rate_linear=self.get_parameter('sensors.noise.odom_drift_rate_linear').value,
            odom_drift_rate_angular=self.get_parameter('sensors.noise.odom_drift_rate_angular').value
        )
        
        # Get detection error parameters
        self.detection_errors_enabled = self.get_parameter('sensors.detection_errors.enable').value
        self.false_negative_rate = self.get_parameter('sensors.detection_errors.false_negative_rate').value
        self.false_positive_rate = self.get_parameter('sensors.detection_errors.false_positive_rate').value
        self.wrong_color_rate = self.get_parameter('sensors.detection_errors.wrong_color_rate').value
        self.unknown_color_rate = self.get_parameter('sensors.detection_errors.unknown_color_rate').value
        
        # Load ground truth cones
        if self.scenario == 1:
            self.ground_truth_cones = GROUND_TRUTH_CONES_SCENARIO_1
            self.get_logger().info("Using Scenario 1: Straight track with AEB zone")
            motion_scenario = MotionScenario.STRAIGHT_TRACK
        else:
            self.ground_truth_cones = GROUND_TRUTH_CONES_SCENARIO_2
            self.get_logger().info("Using Scenario 2: Formula Student track")
            motion_scenario = MotionScenario.FORMULA_STUDENT
        
        # Initialize modular components
        self.sensor_sim = SensorSimulator(noise_config)
        self.motion_controller = MotionController(motion_scenario, self.vehicle_speed)
        
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
        
        # Track start time for elapsed time
        self.start_time = self.get_clock().now()
        
        # Initialize vehicle state
        self.vehicle_state = self.motion_controller.state
        
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
        
        self.get_logger().info("Dummy publisher initialized (refactored version)")
    
    def update_motion(self):
        """Update robot motion using motion controller"""
        dt = 0.01
        elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        
        # Update vehicle state using motion controller
        self.vehicle_state = self.motion_controller.update_motion(dt, elapsed_time)
        
        # Publish transforms based on ground truth state
        self.publish_transforms()
        
        # Update and publish ground truth path
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = "map"
        pose_stamped.pose.position.x = self.vehicle_state.position[0]
        pose_stamped.pose.position.y = self.vehicle_state.position[1]
        pose_stamped.pose.position.z = self.vehicle_state.position[2]
        
        q = quaternion_from_euler(
            self.vehicle_state.orientation[0],
            self.vehicle_state.orientation[1], 
            self.vehicle_state.orientation[2]
        )
        pose_stamped.pose.orientation.x = q[0]
        pose_stamped.pose.orientation.y = q[1]
        pose_stamped.pose.orientation.z = q[2]
        pose_stamped.pose.orientation.w = q[3]
        
        self.gt_path.poses.append(pose_stamped)
        if len(self.gt_path.poses) > 1000:  # Limit path length
            self.gt_path.poses.pop(0)
        
        # Publish ground truth pose
        self.pose_pub.publish(pose_stamped)
    
    def publish_transforms(self):
        """Publish TF transforms"""
        now = self.get_clock().now()
        
        # Map -> Ground truth base_link (perfect)
        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "ground_truth_base_link"
        t.transform.translation.x = self.vehicle_state.position[0]
        t.transform.translation.y = self.vehicle_state.position[1]
        t.transform.translation.z = self.vehicle_state.position[2]
        
        q = quaternion_from_euler(
            self.vehicle_state.orientation[0],
            self.vehicle_state.orientation[1],
            self.vehicle_state.orientation[2]
        )
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        
        self.tf_broadcaster.sendTransform(t)
        
        # Ground truth base_link -> sensors (fixed transforms)
        # IMU link
        t_imu = TransformStamped()
        t_imu.header.stamp = now.to_msg()
        t_imu.header.frame_id = "ground_truth_base_link"
        t_imu.child_frame_id = "imu_link"
        t_imu.transform.translation.x = 0.0
        t_imu.transform.translation.y = 0.0
        t_imu.transform.translation.z = 0.3
        t_imu.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t_imu)
        
        # GPS link
        t_gps = TransformStamped()
        t_gps.header.stamp = now.to_msg()
        t_gps.header.frame_id = "ground_truth_base_link"
        t_gps.child_frame_id = "gps_link"
        t_gps.transform.translation.x = -0.5
        t_gps.transform.translation.y = 0.0
        t_gps.transform.translation.z = 0.5
        t_gps.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t_gps)
        
        # Note: We don't publish map->base_link transform as that's typically
        # done by a localization system (e.g., robot_localization package)
    
    def publish_ground_truth_cones(self):
        """Publish ground truth cone positions for visualization"""
        cones_list = []
        for cone_id, cone_data in self.ground_truth_cones.items():
            cones_list.append({
                'pos': cone_data['pos'],
                'type': cone_data['type'],
                'id': cone_id
            })
        
        publish_cone_array(
            self.gt_cones_pub,
            cones_list,
            namespace="ground_truth_cones",
            frame_id="map",
            with_text=True,
            timestamp=self.get_clock().now()
        )
    
    def publish_cones(self):
        """Publish cone observations with realistic errors"""
        msg = TrackedConeArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        
        # Get visible cones from ground truth position
        visible_cones = self.get_visible_cones()
        
        # Track which cones are newly visible
        currently_visible_ids = set()
        self.last_detected_cones = []
        
        for cone_id, local_pos, cone_type in visible_cones:
            currently_visible_ids.add(cone_id)
            
            # Apply false negative (miss detection)
            if self.detection_errors_enabled and np.random.random() < self.false_negative_rate:
                continue  # Skip this cone
            
            # Get or assign track ID
            if cone_id not in self.cone_track_mapping:
                self.cone_track_mapping[cone_id] = self.next_track_id
                self.next_track_id += 1
            
            track_id = self.cone_track_mapping[cone_id]
            
            # Create tracked cone message
            tracked_cone = TrackedCone()
            tracked_cone.track_id = track_id
            
            # Add position noise
            noisy_pos = local_pos + np.random.normal(0, self.cone_position_noise, 2)
            tracked_cone.position.x = noisy_pos[0]
            tracked_cone.position.y = noisy_pos[1]
            tracked_cone.position.z = -0.15
            
            # Handle color errors
            detected_color = cone_type
            
            # Wrong color classification
            if self.detection_errors_enabled and np.random.random() < self.wrong_color_rate:
                # Randomly assign wrong color
                color_options = ['yellow', 'blue', 'red']
                color_options.remove(cone_type)  # Remove correct color
                detected_color = np.random.choice(color_options)
            
            # Unknown color (sensor fusion failure)
            elif self.detection_errors_enabled and np.random.random() < self.unknown_color_rate:
                detected_color = 'unknown'
            
            # Set color string
            if detected_color == 'yellow':
                tracked_cone.color = "Yellow cone"
            elif detected_color == 'blue':
                tracked_cone.color = "Blue cone"
            elif detected_color == 'red':
                tracked_cone.color = "Red cone"
            else:
                tracked_cone.color = "Unknown"
            
            msg.cones.append(tracked_cone)
            self.last_detected_cones.append((track_id, noisy_pos, detected_color))
        
        # Remove track IDs for cones no longer visible
        lost_cone_ids = set(self.cone_track_mapping.keys()) - currently_visible_ids
        for cone_id in lost_cone_ids:
            del self.cone_track_mapping[cone_id]
        
        # False positive detections
        if self.detection_errors_enabled:
            num_false_positives = np.random.poisson(self.false_positive_rate * len(self.ground_truth_cones))
            
            for _ in range(num_false_positives):
                # Generate random false detection within ROI
                if self.roi_type == 'sector':
                    # Random point in sector
                    r = np.random.uniform(2.0, self.detection_range)
                    theta = np.random.uniform(-self.fov_rad/2, self.fov_rad/2)
                    fake_local_x = r * np.cos(theta)
                    fake_local_y = r * np.sin(theta)
                else:
                    # Random point in rectangle
                    fake_local_x = np.random.uniform(2.0, self.detection_range)
                    max_y = self.detection_range * np.tan(self.fov_rad / 2)
                    fake_local_y = np.random.uniform(-max_y, max_y)
                
                # Create fake cone
                fake_cone = TrackedCone()
                fake_cone.track_id = self.next_track_id
                self.next_track_id += 1
                
                fake_cone.position.x = fake_local_x
                fake_cone.position.y = fake_local_y
                fake_cone.position.z = -0.15
                
                # Random color for fake cone
                fake_color = np.random.choice(['yellow', 'blue', 'red', 'unknown'])
                if fake_color == 'yellow':
                    fake_cone.color = "Yellow cone"
                elif fake_color == 'blue':
                    fake_cone.color = "Blue cone"
                elif fake_color == 'red':
                    fake_cone.color = "Red cone"
                else:
                    fake_cone.color = "Unknown"
                
                msg.cones.append(fake_cone)
                self.last_detected_cones.append((fake_cone.track_id, np.array([fake_local_x, fake_local_y]), fake_color))
        
        self.cone_pub.publish(msg)
        
        # Also publish visualization of detected cones
        self.publish_detected_cones_visualization()
    
    def get_visible_cones(self) -> List[Tuple[int, np.ndarray, str]]:
        """Get cones visible from ground truth robot pose"""
        visible = []
        
        for cone_id, cone_data in self.ground_truth_cones.items():
            # Transform to ground truth robot frame
            dx = cone_data['pos'][0] - self.vehicle_state.position[0]
            dy = cone_data['pos'][1] - self.vehicle_state.position[1]
            
            # Rotate to robot frame
            theta = self.vehicle_state.orientation[2]
            local_x = dx * np.cos(-theta) - dy * np.sin(-theta)
            local_y = dx * np.sin(-theta) + dy * np.cos(-theta)
            
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
        roi_marker = VisualizationHelper.create_roi_marker(
            self.roi_type,
            self.detection_range,
            self.fov_rad,
            frame_id="ground_truth_base_link",
            timestamp=self.get_clock().now()
        )
        self.roi_pub.publish(roi_marker)
    
    def publish_centerline(self):
        """Publish centerline visualization"""
        centerline_points = self.motion_controller.get_centerline_points()
        centerline_marker = VisualizationHelper.create_path_marker(
            centerline_points,
            color="green",
            namespace="centerline",
            frame_id="map",
            timestamp=self.get_clock().now()
        )
        self.centerline_pub.publish(centerline_marker)
    
    def publish_detected_cones_visualization(self):
        """Publish visualization of detected cones"""
        publish_detected_cones(
            self.detected_cones_vis_pub,
            self.last_detected_cones,
            namespace="detected_cones",
            frame_id="base_link",
            timestamp=self.get_clock().now()
        )
    
    def publish_imu(self):
        """Publish IMU data using sensor simulator"""
        dt = 1.0 / self.imu_rate
        
        # Generate IMU message using sensor simulator
        imu_msg = self.sensor_sim.imu_sim.generate_imu_data(
            self.vehicle_state.linear_acceleration,
            self.vehicle_state.angular_velocity,
            quaternion_from_euler(
                self.vehicle_state.orientation[0],
                self.vehicle_state.orientation[1],
                self.vehicle_state.orientation[2]
            ),
            dt,
            self.get_clock().now()
        )
        
        self.imu_pub.publish(imu_msg)
    
    def publish_gps(self):
        """Publish GPS data using sensor simulator"""
        # Generate GPS message
        gps_msg = self.sensor_sim.gps_sim.generate_gps_data(
            self.vehicle_state.position[0],
            self.vehicle_state.position[1],
            self.vehicle_state.position[2],
            self.get_clock().now()
        )
        
        self.gps_pub.publish(gps_msg)
        
        # Also publish GPS velocity (simulated)
        gps_vel_msg = TwistWithCovarianceStamped()
        gps_vel_msg.header.stamp = self.get_clock().now().to_msg()
        gps_vel_msg.header.frame_id = "gps_link"
        
        # GPS velocity in ENU frame
        vel_magnitude = np.sqrt(
            self.vehicle_state.linear_velocity[0]**2 + 
            self.vehicle_state.linear_velocity[1]**2
        )
        gps_vel_msg.twist.twist.linear.x = self.vehicle_state.linear_velocity[0] + np.random.normal(0, 0.1)
        gps_vel_msg.twist.twist.linear.y = self.vehicle_state.linear_velocity[1] + np.random.normal(0, 0.1)
        gps_vel_msg.twist.twist.linear.z = 0.0
        
        # Covariance
        gps_vel_msg.twist.covariance[0] = 0.01  # vx
        gps_vel_msg.twist.covariance[7] = 0.01  # vy
        gps_vel_msg.twist.covariance[14] = 0.1  # vz
        
        self.gps_vel_pub.publish(gps_vel_msg)
        
        # Publish odometry with drift
        dt = 1.0 / self.gps_rate
        odom_msg = self.sensor_sim.odom_sim.generate_odometry_data(
            self.vehicle_state.position[0],
            self.vehicle_state.position[1],
            self.vehicle_state.orientation[2],
            self.vehicle_state.linear_velocity[0],
            self.vehicle_state.linear_velocity[1],
            self.vehicle_state.angular_velocity[2],
            dt,
            self.get_clock().now()
        )
        
        self.odom_pub.publish(odom_msg)
        
        # Update path with noisy odometry
        pose_stamped = PoseStamped()
        pose_stamped.header = odom_msg.header
        pose_stamped.pose = odom_msg.pose.pose
        
        self.path.poses.append(pose_stamped)
        if len(self.path.poses) > 1000:  # Limit path length
            self.path.poses.pop(0)
        
        self.path_pub.publish(self.path)


def main(args=None):
    rclpy.init(args=args)
    node = DummyPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()