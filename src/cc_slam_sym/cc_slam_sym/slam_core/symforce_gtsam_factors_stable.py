#!/usr/bin/env python3
"""
Stable GTSAM wrappers for SymForce-generated factors
Addresses numerical stability issues with CustomFactor
"""

import numpy as np
import gtsam
from typing import List, Optional, Callable
import sys
from pathlib import Path
import sym

# Import SymForce-generated functions
sys.path.insert(0, str(Path(__file__).parent / "generated"))
try:
    from cone_color_factor_residual import cone_color_factor_residual
    from motion_model_residual import motion_model_residual
    SYMFORCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SymForce generated functions not available: {e}")
    SYMFORCE_AVAILABLE = False


def gtsam_pose2_to_sym(gtsam_pose: gtsam.Pose2) -> sym.Pose2:
    """Convert GTSAM Pose2 to sym.Pose2 for SymForce functions"""
    rot = sym.Rot2.from_angle(gtsam_pose.theta())
    trans = np.array([gtsam_pose.x(), gtsam_pose.y()])
    return sym.Pose2(rot, trans)


def color_string_to_float(color: str) -> float:
    """Convert color string to float encoding for SymForce factors"""
    color_map = {
        'yellow': 0.0,
        'blue': 1.0,
        'red': 2.0,
        'orange': 3.0,
        'unknown': -1.0
    }
    return color_map.get(color.lower(), -1.0)


def create_symforce_cone_factor(pose_key: int,
                               landmark_key: int,
                               observation: np.ndarray,
                               obs_color: str,
                               landmark_color: str,
                               position_noise: float = 0.1,
                               color_weight: float = 5.0,
                               use_bearing_range: bool = False) -> gtsam.CustomFactor:
    """
    Create a numerically stable cone observation factor
    
    For now, we use a simplified version without SymForce to avoid numerical issues
    """
    # Convert colors to float
    obs_color_float = color_string_to_float(obs_color)
    landmark_color_float = color_string_to_float(landmark_color)
    
    # Create simplified error function
    def error_func(this, values, jacobians):
        """Simplified error function for numerical stability"""
        try:
            # Get variables
            pose = values.atPose2(this.keys()[0])
            landmark_pt = values.atPoint2(this.keys()[1])
            
            # Transform landmark to robot frame
            landmark_robot = pose.transformTo(gtsam.Point2(landmark_pt[0], landmark_pt[1]))
            
            # Compute position error
            error_x = landmark_robot[0] - observation[0]
            error_y = landmark_robot[1] - observation[1]
            
            # Compute color error (0 if colors match, penalty otherwise)
            color_error = 0.0
            if abs(obs_color_float - landmark_color_float) > 0.5:
                color_error = color_weight
                
            # Return residual
            return np.array([error_x, error_y, color_error], dtype=np.float64)
            
        except Exception as e:
            print(f"Error in cone factor: {e}")
            return np.zeros(3, dtype=np.float64)
    
    # Create noise model
    noise_sigmas = np.array([
        position_noise,      # x position
        position_noise,      # y position
        1.0                 # color noise
    ])
    noise_model = gtsam.noiseModel.Diagonal.Sigmas(noise_sigmas)
    
    # Apply robust kernel for outlier rejection
    noise_model = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.0),
        noise_model
    )
    
    return gtsam.CustomFactor(noise_model, [pose_key, landmark_key], error_func)


def create_symforce_motion_factor(pose1_key: int,
                                 pose2_key: int,
                                 odometry: gtsam.Pose2,
                                 position_noise: float = 0.2,
                                 rotation_noise: float = 0.1,
                                 wheelbase: float = 0.3) -> gtsam.CustomFactor:
    """
    Create a numerically stable motion model factor
    
    For now, we use a simplified version without SymForce to avoid numerical issues
    """
    # Create simplified error function
    def error_func(this, values, jacobians):
        """Simplified error function for numerical stability"""
        try:
            # Get variables
            pose1 = values.atPose2(this.keys()[0])
            pose2 = values.atPose2(this.keys()[1])
            
            # Compute expected pose
            expected_pose2 = pose1.compose(odometry)
            
            # Compute error
            error_pose = expected_pose2.between(pose2)
            
            # Extract errors
            error_x = error_pose.x()
            error_y = error_pose.y()
            error_theta = error_pose.theta()
            
            # Simple Ackermann constraint - penalize lateral motion
            # Approximate lateral velocity from consecutive poses
            delta_pose = pose1.between(pose2)
            forward_vel = delta_pose.x()
            lateral_vel = delta_pose.y()
            
            # Lateral constraint error (should be near zero for Ackermann motion)
            lateral_error = lateral_vel * 10.0
            
            # Return residual
            return np.array([error_x, error_y, error_theta, lateral_error], dtype=np.float64)
            
        except Exception as e:
            print(f"Error in motion factor: {e}")
            return np.zeros(4, dtype=np.float64)
    
    # Create noise model
    noise_sigmas = np.array([
        position_noise,      # x position
        position_noise,      # y position
        rotation_noise,      # rotation
        0.1                 # lateral constraint
    ])
    noise_model = gtsam.noiseModel.Diagonal.Sigmas(noise_sigmas)
    
    return gtsam.CustomFactor(noise_model, [pose1_key, pose2_key], error_func)


# For backward compatibility, keep original function names
if SYMFORCE_AVAILABLE:
    # Try to use SymForce functions if they're stable enough
    def create_symforce_cone_factor_with_symforce(pose_key: int,
                                                  landmark_key: int,
                                                  observation: np.ndarray,
                                                  obs_color: str,
                                                  landmark_color: str,
                                                  position_noise: float = 0.1,
                                                  color_weight: float = 5.0,
                                                  use_bearing_range: bool = False):
        """Original SymForce-based implementation (may have numerical issues)"""
        obs_color_float = color_string_to_float(obs_color)
        landmark_color_float = color_string_to_float(landmark_color)
        
        if observation.shape == (2,):
            obs = observation.reshape((2, 1))
        else:
            obs = observation
            
        def error_func(this, values, jacobians):
            try:
                pose = values.atPose2(this.keys()[0])
                landmark = values.atPoint2(this.keys()[1])
                
                sym_pose = gtsam_pose2_to_sym(pose)
                landmark_array = np.array([landmark[0], landmark[1]]).reshape((2, 1))
                
                residual = cone_color_factor_residual(
                    robot_pose=sym_pose,
                    landmark_pos=landmark_array,
                    observation=obs,
                    observed_color=obs_color_float,
                    landmark_color=landmark_color_float,
                    color_weight=color_weight,
                    epsilon=1e-8
                )
                
                return np.asarray(residual, dtype=np.float64)
            except Exception as e:
                print(f"SymForce error: {e}")
                return np.zeros(3, dtype=np.float64)
        
        noise_sigmas = np.array([position_noise, position_noise, 1.0])
        noise_model = gtsam.noiseModel.Diagonal.Sigmas(noise_sigmas)
        noise_model = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.0),
            noise_model
        )
        
        return gtsam.CustomFactor(noise_model, [pose_key, landmark_key], error_func)