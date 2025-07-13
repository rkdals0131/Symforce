#!/usr/bin/env python3
"""
Sensor Simulation Module for CC-SLAM-SYM

Provides realistic sensor data generation with noise models for:
- IMU (Inertial Measurement Unit)
- GPS (Global Positioning System)
- Odometry

Includes drift, bias, and white noise simulation for realistic sensor behavior.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Vector3, Point, Pose, Twist, TwistWithCovariance, PoseWithCovariance
from std_msgs.msg import Header
from tf_transformations import quaternion_from_euler
import rclpy.time


@dataclass
class SensorNoiseConfig:
    """Configuration for sensor noise parameters"""
    # IMU parameters
    imu_accel_bias_drift: float = 0.1      # m/s² - 가속도계 바이어스 변화율
    imu_gyro_bias_drift: float = 0.01      # rad/s - 자이로스코프 바이어스 변화율
    imu_accel_white_noise: float = 0.002   # m/s² - 가속도계 백색 잡음
    imu_gyro_white_noise: float = 0.0002   # rad/s - 자이로스코프 백색 잡음
    
    # Odometry parameters
    odom_position_noise: float = 0.1       # m - 오도메트리 위치 노이즈
    odom_angle_noise: float = 0.05         # rad - 오도메트리 각도 노이즈
    odom_drift_rate_linear: float = 0.5    # % per meter - 선형 이동 누적 드리프트
    odom_drift_rate_angular: float = 0.2   # % per radian - 회전 누적 드리프트


class ImuSimulator:
    """Simulates IMU sensor data with realistic noise and drift"""
    
    def __init__(self, config: SensorNoiseConfig):
        self.config = config
        
        # Bias states (slowly changing over time)
        self.accel_bias_x = 0.0
        self.accel_bias_y = 0.0
        self.accel_bias_z = 0.0
        self.gyro_bias_x = 0.0
        self.gyro_bias_y = 0.0
        self.gyro_bias_z = 0.0
        
        # Initial gravity vector (assuming flat ground)
        self.gravity = 9.81
    
    def update_bias(self, dt: float) -> None:
        """Update bias states with slow drift"""
        # Acceleration bias drift
        self.accel_bias_x += np.random.normal(0, self.config.imu_accel_bias_drift * dt)
        self.accel_bias_y += np.random.normal(0, self.config.imu_accel_bias_drift * dt)
        self.accel_bias_z += np.random.normal(0, self.config.imu_accel_bias_drift * dt)
        
        # Gyroscope bias drift
        self.gyro_bias_x += np.random.normal(0, self.config.imu_gyro_bias_drift * dt)
        self.gyro_bias_y += np.random.normal(0, self.config.imu_gyro_bias_drift * dt)
        self.gyro_bias_z += np.random.normal(0, self.config.imu_gyro_bias_drift * dt)
        
        # Limit bias to reasonable values
        max_accel_bias = 0.5  # m/s²
        max_gyro_bias = 0.05   # rad/s
        
        self.accel_bias_x = np.clip(self.accel_bias_x, -max_accel_bias, max_accel_bias)
        self.accel_bias_y = np.clip(self.accel_bias_y, -max_accel_bias, max_accel_bias)
        self.accel_bias_z = np.clip(self.accel_bias_z, -max_accel_bias, max_accel_bias)
        
        self.gyro_bias_x = np.clip(self.gyro_bias_x, -max_gyro_bias, max_gyro_bias)
        self.gyro_bias_y = np.clip(self.gyro_bias_y, -max_gyro_bias, max_gyro_bias)
        self.gyro_bias_z = np.clip(self.gyro_bias_z, -max_gyro_bias, max_gyro_bias)
    
    def generate_imu_data(self, true_accel: np.ndarray, true_angular_vel: np.ndarray, 
                         true_orientation: Tuple[float, float, float, float],
                         dt: float, timestamp: rclpy.time.Time) -> Imu:
        """
        Generate IMU message with realistic noise
        
        Args:
            true_accel: True linear acceleration [ax, ay, az] in robot frame
            true_angular_vel: True angular velocity [wx, wy, wz] in robot frame
            true_orientation: True orientation as quaternion (x, y, z, w)
            dt: Time step
            timestamp: ROS timestamp
        
        Returns:
            Imu message with noise and bias
        """
        # Update bias
        self.update_bias(dt)
        
        # Create IMU message
        imu_msg = Imu()
        imu_msg.header.stamp = timestamp.to_msg()
        imu_msg.header.frame_id = "imu_link"
        
        # Apply gravity to acceleration (IMU measures gravity)
        # In robot frame, gravity appears as upward acceleration when stationary
        gravity_in_robot = np.array([0, 0, self.gravity])
        
        # Add bias and white noise to acceleration
        accel_with_gravity = true_accel + gravity_in_robot
        imu_msg.linear_acceleration.x = (accel_with_gravity[0] + self.accel_bias_x + 
                                        np.random.normal(0, self.config.imu_accel_white_noise))
        imu_msg.linear_acceleration.y = (accel_with_gravity[1] + self.accel_bias_y + 
                                        np.random.normal(0, self.config.imu_accel_white_noise))
        imu_msg.linear_acceleration.z = (accel_with_gravity[2] + self.accel_bias_z + 
                                        np.random.normal(0, self.config.imu_accel_white_noise))
        
        # Add bias and white noise to angular velocity
        imu_msg.angular_velocity.x = (true_angular_vel[0] + self.gyro_bias_x + 
                                     np.random.normal(0, self.config.imu_gyro_white_noise))
        imu_msg.angular_velocity.y = (true_angular_vel[1] + self.gyro_bias_y + 
                                     np.random.normal(0, self.config.imu_gyro_white_noise))
        imu_msg.angular_velocity.z = (true_angular_vel[2] + self.gyro_bias_z + 
                                     np.random.normal(0, self.config.imu_gyro_white_noise))
        
        # Set orientation (with small noise)
        orientation_noise = 0.001  # Very small orientation noise
        imu_msg.orientation.x = true_orientation[0] + np.random.normal(0, orientation_noise)
        imu_msg.orientation.y = true_orientation[1] + np.random.normal(0, orientation_noise)
        imu_msg.orientation.z = true_orientation[2] + np.random.normal(0, orientation_noise)
        imu_msg.orientation.w = true_orientation[3] + np.random.normal(0, orientation_noise)
        
        # Normalize quaternion
        quat_norm = np.sqrt(imu_msg.orientation.x**2 + imu_msg.orientation.y**2 + 
                           imu_msg.orientation.z**2 + imu_msg.orientation.w**2)
        imu_msg.orientation.x /= quat_norm
        imu_msg.orientation.y /= quat_norm
        imu_msg.orientation.z /= quat_norm
        imu_msg.orientation.w /= quat_norm
        
        # Set covariances (diagonal matrices) - must be exactly 9 floats
        imu_msg.orientation_covariance = [
            0.01, 0.0, 0.0,
            0.0, 0.01, 0.0,
            0.0, 0.0, 0.01
        ]
        
        imu_msg.angular_velocity_covariance = [
            float(self.config.imu_gyro_white_noise**2), 0.0, 0.0,
            0.0, float(self.config.imu_gyro_white_noise**2), 0.0,
            0.0, 0.0, float(self.config.imu_gyro_white_noise**2)
        ]
        
        imu_msg.linear_acceleration_covariance = [
            float(self.config.imu_accel_white_noise**2), 0.0, 0.0,
            0.0, float(self.config.imu_accel_white_noise**2), 0.0,
            0.0, 0.0, float(self.config.imu_accel_white_noise**2)
        ]
        
        return imu_msg


class GpsSimulator:
    """Simulates GPS sensor data with realistic noise"""
    
    def __init__(self, origin_lat: float = 37.5665, origin_lon: float = 126.9780):
        """
        Initialize GPS simulator
        
        Args:
            origin_lat: Origin latitude (default: Seoul)
            origin_lon: Origin longitude (default: Seoul)
        """
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        
        # GPS noise parameters
        self.position_noise = 2.0  # meters (typical GPS accuracy)
        self.altitude_noise = 5.0  # meters (GPS altitude is less accurate)
    
    def xy_to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        """Convert local XY coordinates to latitude/longitude"""
        # Approximate conversion (good for small distances)
        lat_per_meter = 1.0 / 111111.0
        lon_per_meter = 1.0 / (111111.0 * np.cos(np.radians(self.origin_lat)))
        
        lat = self.origin_lat + y * lat_per_meter
        lon = self.origin_lon + x * lon_per_meter
        
        return lat, lon
    
    def generate_gps_data(self, true_x: float, true_y: float, true_z: float,
                         timestamp: rclpy.time.Time) -> NavSatFix:
        """
        Generate GPS NavSatFix message with noise
        
        Args:
            true_x, true_y: True position in local coordinates (meters)
            true_z: True altitude (meters)
            timestamp: ROS timestamp
        
        Returns:
            NavSatFix message with GPS noise
        """
        # Add noise to position
        noisy_x = true_x + np.random.normal(0, self.position_noise)
        noisy_y = true_y + np.random.normal(0, self.position_noise)
        noisy_z = true_z + np.random.normal(0, self.altitude_noise)
        
        # Convert to lat/lon
        lat, lon = self.xy_to_latlon(noisy_x, noisy_y)
        
        # Create GPS message
        gps_msg = NavSatFix()
        gps_msg.header.stamp = timestamp.to_msg()
        gps_msg.header.frame_id = "gps_link"
        
        gps_msg.latitude = lat
        gps_msg.longitude = lon
        gps_msg.altitude = noisy_z
        
        # Status (0 = no fix, 1 = GPS fix, 2 = DGPS fix)
        gps_msg.status.status = 1
        gps_msg.status.service = 1  # GPS service
        
        # Position covariance (m²) - must be exactly 9 floats
        gps_msg.position_covariance = [
            float(self.position_noise**2), 0.0, 0.0,
            0.0, float(self.position_noise**2), 0.0,
            0.0, 0.0, float(self.altitude_noise**2)
        ]
        gps_msg.position_covariance_type = 2  # Diagonal known
        
        return gps_msg


class OdometrySimulator:
    """Simulates wheel odometry with cumulative drift"""
    
    def __init__(self, config: SensorNoiseConfig):
        self.config = config
        
        # Cumulative drift states
        self.drift_x = 0.0
        self.drift_y = 0.0
        self.drift_theta = 0.0
        
        # Track total distance for drift calculation
        self.total_distance = 0.0
        self.total_rotation = 0.0
    
    def update_drift(self, distance_moved: float, angle_rotated: float) -> None:
        """Update cumulative drift based on movement"""
        # Linear drift proportional to distance
        if distance_moved > 0:
            drift_magnitude = distance_moved * self.config.odom_drift_rate_linear * 0.01
            drift_angle = np.random.uniform(0, 2 * np.pi)
            # Fixed: Remove random multiplier that can cancel out drift
            self.drift_x += drift_magnitude * np.cos(drift_angle)
            self.drift_y += drift_magnitude * np.sin(drift_angle)
        
        # Angular drift proportional to rotation
        if abs(angle_rotated) > 0:
            # Fixed: Consistent drift accumulation direction
            self.drift_theta += angle_rotated * self.config.odom_drift_rate_angular * 0.01
        
        # Update totals
        self.total_distance += distance_moved
        self.total_rotation += abs(angle_rotated)
    
    def generate_odometry_data(self, true_x: float, true_y: float, true_theta: float,
                              true_vx: float, true_vy: float, true_vtheta: float,
                              dt: float, timestamp: rclpy.time.Time,
                              child_frame_id: str = "base_link") -> Odometry:
        """
        Generate Odometry message with cumulative drift and noise
        
        Args:
            true_x, true_y, true_theta: True pose
            true_vx, true_vy, true_vtheta: True velocities
            dt: Time step
            timestamp: ROS timestamp
            child_frame_id: Child frame ID
        
        Returns:
            Odometry message with drift and noise
        """
        # Calculate movement since last update
        distance_moved = np.sqrt(true_vx**2 + true_vy**2) * dt
        angle_rotated = true_vtheta * dt
        
        # Update cumulative drift
        self.update_drift(distance_moved, angle_rotated)
        
        # Debug: Print drift values periodically
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 50 == 0:
            print(f"DEBUG DRIFT: x={self.drift_x:.6f}, y={self.drift_y:.6f}, theta={self.drift_theta:.6f}, "
                  f"dist_moved={distance_moved:.6f}, angle_rot={angle_rotated:.6f}, total_dist={self.total_distance:.2f}, "
                  f"drift_rates=[{self.config.odom_drift_rate_linear}, {self.config.odom_drift_rate_angular}]")
        
        # Create odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = timestamp.to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = child_frame_id
        
        # Apply drift and noise to position
        odom_msg.pose.pose.position.x = (true_x + self.drift_x + 
                                        np.random.normal(0, self.config.odom_position_noise))
        odom_msg.pose.pose.position.y = (true_y + self.drift_y + 
                                        np.random.normal(0, self.config.odom_position_noise))
        odom_msg.pose.pose.position.z = 0.0
        
        # Apply drift and noise to orientation
        noisy_theta = true_theta + self.drift_theta + np.random.normal(0, self.config.odom_angle_noise)
        q = quaternion_from_euler(0, 0, noisy_theta)
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]
        
        # Add noise to velocities
        odom_msg.twist.twist.linear.x = true_vx + np.random.normal(0, 0.05)
        odom_msg.twist.twist.linear.y = true_vy + np.random.normal(0, 0.05)
        odom_msg.twist.twist.angular.z = true_vtheta + np.random.normal(0, 0.01)
        
        # Covariances
        # Pose covariance
        pose_cov = np.zeros(36)
        pose_cov[0] = self.config.odom_position_noise**2  # x
        pose_cov[7] = self.config.odom_position_noise**2  # y
        pose_cov[35] = self.config.odom_angle_noise**2    # theta
        odom_msg.pose.covariance = pose_cov.tolist()
        
        # Twist covariance
        twist_cov = np.zeros(36)
        twist_cov[0] = 0.01   # vx
        twist_cov[7] = 0.01   # vy
        twist_cov[35] = 0.01  # vtheta
        odom_msg.twist.covariance = twist_cov.tolist()
        
        return odom_msg


class SensorSimulator:
    """Main sensor simulator combining IMU, GPS, and Odometry"""
    
    def __init__(self, config: SensorNoiseConfig):
        self.config = config
        self.imu_sim = ImuSimulator(config)
        self.gps_sim = GpsSimulator()
        self.odom_sim = OdometrySimulator(config)
    
    def generate_all_sensors(self, vehicle_state: Dict, dt: float, 
                           timestamp: rclpy.time.Time) -> Tuple[Imu, NavSatFix, Odometry]:
        """
        Generate all sensor data for current vehicle state
        
        Args:
            vehicle_state: Dictionary containing:
                - position: [x, y, z]
                - orientation: [roll, pitch, yaw]
                - linear_velocity: [vx, vy, vz]
                - angular_velocity: [wx, wy, wz]
                - linear_acceleration: [ax, ay, az]
            dt: Time step
            timestamp: ROS timestamp
        
        Returns:
            Tuple of (IMU, GPS, Odometry) messages
        """
        # Extract state
        pos = vehicle_state['position']
        orient = vehicle_state['orientation']
        lin_vel = vehicle_state['linear_velocity']
        ang_vel = vehicle_state['angular_velocity']
        lin_acc = vehicle_state.get('linear_acceleration', [0, 0, 0])
        
        # Generate quaternion from Euler angles
        q = quaternion_from_euler(orient[0], orient[1], orient[2])
        
        # Generate sensor data
        imu_msg = self.imu_sim.generate_imu_data(
            np.array(lin_acc), np.array(ang_vel), q, dt, timestamp
        )
        
        gps_msg = self.gps_sim.generate_gps_data(
            pos[0], pos[1], pos[2], timestamp
        )
        
        odom_msg = self.odom_sim.generate_odometry_data(
            pos[0], pos[1], orient[2],
            lin_vel[0], lin_vel[1], ang_vel[2],
            dt, timestamp
        )
        
        return imu_msg, gps_msg, odom_msg