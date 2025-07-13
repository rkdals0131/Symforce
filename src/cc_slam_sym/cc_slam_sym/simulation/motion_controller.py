#!/usr/bin/env python3
"""
Motion Controller Module for CC-SLAM-SYM

Handles vehicle motion simulation along predefined paths.
Supports different scenarios like straight tracks and Formula Student circuits.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class MotionScenario(Enum):
    """Available motion scenarios"""
    STRAIGHT_TRACK = 1  # Straight line with AEB test
    FORMULA_STUDENT = 2  # Elliptical Formula Student track


@dataclass
class VehicleState:
    """Complete vehicle state"""
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [roll, pitch, yaw]
    linear_velocity: np.ndarray  # [vx, vy, vz]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    linear_acceleration: np.ndarray  # [ax, ay, az]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for sensor simulator"""
        return {
            'position': self.position.tolist(),
            'orientation': self.orientation.tolist(),
            'linear_velocity': self.linear_velocity.tolist(),
            'angular_velocity': self.angular_velocity.tolist(),
            'linear_acceleration': self.linear_acceleration.tolist()
        }


class TrajectoryGenerator:
    """Generates smooth trajectories for different scenarios"""
    
    @staticmethod
    def generate_straight_track_centerline(length: float = 150.0, 
                                         num_points: int = 300) -> List[Tuple[float, float]]:
        """Generate centerline for straight track scenario"""
        points = []
        for i in range(num_points):
            x = i * length / (num_points - 1)
            y = 0.0
            points.append((x, y))
        return points
    
    @staticmethod
    def generate_formula_student_centerline(num_points: int = 400) -> List[Tuple[float, float]]:
        """Generate centerline for Formula Student elliptical track"""
        # Use proven waypoints from successful implementation
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
        
        points = []
        # Interpolate between waypoints for smoother path
        for i in range(len(base_waypoints)):
            current = base_waypoints[i]
            next_wp = base_waypoints[(i + 1) % len(base_waypoints)]
            
            # Add current waypoint
            points.append(current)
            
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
                points.append((x, y))
        
        return points
    
    @staticmethod
    def smooth_centerline(points: List[Tuple[float, float]], 
                         smoothing_factor: float = 0.1) -> List[Tuple[float, float]]:
        """Apply smoothing to centerline points"""
        if len(points) < 3:
            return points
        
        smoothed = []
        for i in range(len(points)):
            if i == 0 or i == len(points) - 1:
                smoothed.append(points[i])
            else:
                # Simple moving average
                prev_point = points[i-1]
                curr_point = points[i]
                next_point = points[(i+1) % len(points)]
                
                avg_x = (prev_point[0] + curr_point[0] + next_point[0]) / 3
                avg_y = (prev_point[1] + curr_point[1] + next_point[1]) / 3
                
                # Blend with original
                smooth_x = curr_point[0] * (1 - smoothing_factor) + avg_x * smoothing_factor
                smooth_y = curr_point[1] * (1 - smoothing_factor) + avg_y * smoothing_factor
                
                smoothed.append((smooth_x, smooth_y))
        
        return smoothed


class MotionController:
    """Controls vehicle motion along trajectories"""
    
    def __init__(self, scenario: MotionScenario, base_speed: float = 5.0):
        """
        Initialize motion controller
        
        Args:
            scenario: Motion scenario to use
            base_speed: Base vehicle speed in m/s
        """
        self.scenario = scenario
        self.base_speed = base_speed
        
        # Generate centerline based on scenario
        if scenario == MotionScenario.STRAIGHT_TRACK:
            self.centerline = TrajectoryGenerator.generate_straight_track_centerline()
        else:
            self.centerline = TrajectoryGenerator.generate_formula_student_centerline()
        
        # Smooth the centerline
        self.centerline = TrajectoryGenerator.smooth_centerline(self.centerline)
        
        # Initialize vehicle state based on scenario
        if scenario == MotionScenario.STRAIGHT_TRACK:
            # Scenario 1: Start at beginning of track
            initial_pos = np.array([0.0, 0.0, 0.0])
            self.centerline_index = 0
        else:
            # Scenario 2: Start at specific position
            initial_pos = np.array([30.0, 12.5, 0.0])
            # Find closest centerline point
            min_dist = float('inf')
            closest_idx = 0
            for i, point in enumerate(self.centerline):
                dist = np.sqrt((point[0] - initial_pos[0])**2 + (point[1] - initial_pos[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            self.centerline_index = closest_idx
        
        self.distance_along_segment = 0.0
        
        self.state = VehicleState(
            position=initial_pos,
            orientation=np.array([0.0, 0.0, 0.0]),
            linear_velocity=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            linear_acceleration=np.array([0.0, 0.0, 0.0])
        )
        
        # Previous values for derivative calculations
        self.prev_velocity = np.array([0.0, 0.0, 0.0])
        self.prev_yaw = 0.0
    
    def get_centerline_points(self) -> List[Tuple[float, float]]:
        """Get centerline points for visualization"""
        return self.centerline
    
    def get_current_speed(self, elapsed_time: float) -> float:
        """
        Get current speed based on scenario and time
        
        Args:
            elapsed_time: Time since start in seconds
        
        Returns:
            Current speed in m/s
        """
        if self.scenario == MotionScenario.STRAIGHT_TRACK:
            # Accelerate for first 5 seconds, then constant
            if elapsed_time < 5.0:
                return min(self.base_speed, elapsed_time * self.base_speed / 5.0)
            else:
                return self.base_speed
        else:
            # Formula Student: Vary speed based on track curvature
            # Slower in curves, faster on straights
            return self.base_speed
    
    def update_motion(self, dt: float, elapsed_time: float) -> VehicleState:
        """
        Update vehicle motion for one time step
        
        Args:
            dt: Time step in seconds
            elapsed_time: Total elapsed time since start
        
        Returns:
            Updated vehicle state
        """
        # Get current speed
        speed = self.get_current_speed(elapsed_time)
        
        # Calculate distance to move
        distance_to_move = speed * dt
        
        # Move along centerline
        while distance_to_move > 0 and self.centerline_index < len(self.centerline) - 1:
            # Current and next points
            current_point = self.centerline[self.centerline_index]
            next_point = self.centerline[self.centerline_index + 1]
            
            # Vector to next point
            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]
            segment_length = np.sqrt(dx**2 + dy**2)
            
            if segment_length < 1e-6:
                # Skip zero-length segments
                self.centerline_index += 1
                continue
            
            # Remaining distance in current segment
            remaining_in_segment = segment_length - self.distance_along_segment
            
            if distance_to_move <= remaining_in_segment:
                # Move within current segment
                self.distance_along_segment += distance_to_move
                fraction = self.distance_along_segment / segment_length
                
                # Interpolate position
                self.state.position[0] = current_point[0] + fraction * dx
                self.state.position[1] = current_point[1] + fraction * dy
                
                # Calculate heading
                new_yaw = np.arctan2(dy, dx)
                self.state.orientation[2] = new_yaw
                
                # Calculate velocities
                self.state.linear_velocity[0] = speed * np.cos(new_yaw)
                self.state.linear_velocity[1] = speed * np.sin(new_yaw)
                
                # Calculate angular velocity (yaw rate)
                yaw_diff = self._normalize_angle(new_yaw - self.prev_yaw)
                self.state.angular_velocity[2] = yaw_diff / dt if dt > 0 else 0.0
                self.prev_yaw = new_yaw
                
                distance_to_move = 0
            else:
                # Move to next segment
                distance_to_move -= remaining_in_segment
                self.centerline_index += 1
                self.distance_along_segment = 0.0
                
                # Wrap around for closed tracks
                if self.scenario == MotionScenario.FORMULA_STUDENT:
                    if self.centerline_index >= len(self.centerline) - 1:
                        self.centerline_index = 0
        
        # Calculate acceleration
        new_velocity = self.state.linear_velocity.copy()
        if dt > 0:
            self.state.linear_acceleration = (new_velocity - self.prev_velocity) / dt
        self.prev_velocity = new_velocity.copy()
        
        return self.state
    
    def get_look_ahead_point(self, look_ahead_distance: float) -> Optional[Tuple[float, float]]:
        """
        Get a point ahead on the centerline for trajectory following
        
        Args:
            look_ahead_distance: Distance to look ahead in meters
        
        Returns:
            Look-ahead point or None if end of track
        """
        # Start from current position
        temp_index = self.centerline_index
        temp_distance = self.distance_along_segment
        remaining_distance = look_ahead_distance
        
        while remaining_distance > 0 and temp_index < len(self.centerline) - 1:
            current_point = self.centerline[temp_index]
            next_point = self.centerline[temp_index + 1]
            
            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]
            segment_length = np.sqrt(dx**2 + dy**2)
            
            if segment_length < 1e-6:
                temp_index += 1
                continue
            
            remaining_in_segment = segment_length - temp_distance
            
            if remaining_distance <= remaining_in_segment:
                # Found look-ahead point
                fraction = (temp_distance + remaining_distance) / segment_length
                x = current_point[0] + fraction * dx
                y = current_point[1] + fraction * dy
                return (x, y)
            else:
                remaining_distance -= remaining_in_segment
                temp_index += 1
                temp_distance = 0.0
        
        # For closed tracks, wrap around
        if self.scenario == MotionScenario.FORMULA_STUDENT:
            temp_index = 0
            while remaining_distance > 0 and temp_index < self.centerline_index:
                current_point = self.centerline[temp_index]
                next_point = self.centerline[temp_index + 1]
                
                dx = next_point[0] - current_point[0]
                dy = next_point[1] - current_point[1]
                segment_length = np.sqrt(dx**2 + dy**2)
                
                if segment_length < 1e-6:
                    temp_index += 1
                    continue
                
                if remaining_distance <= segment_length:
                    fraction = remaining_distance / segment_length
                    x = current_point[0] + fraction * dx
                    y = current_point[1] + fraction * dy
                    return (x, y)
                else:
                    remaining_distance -= segment_length
                    temp_index += 1
        
        return None
    
    def reset(self) -> None:
        """Reset motion controller to start position"""
        self.centerline_index = 0
        self.distance_along_segment = 0.0
        
        # Reset to initial position based on scenario
        if self.scenario == MotionScenario.STRAIGHT_TRACK:
            initial_pos = np.array([0.0, 0.0, 0.0])
        else:
            initial_pos = np.array([30.0, 12.5, 0.0])
        
        self.state = VehicleState(
            position=initial_pos,
            orientation=np.array([0.0, 0.0, 0.0]),
            linear_velocity=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            linear_acceleration=np.array([0.0, 0.0, 0.0])
        )
        
        self.prev_velocity = np.array([0.0, 0.0, 0.0])
        self.prev_yaw = 0.0
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle