#!/usr/bin/env python3
"""
Simplified SymForce factors for SLAM
Direct symbolic computation without code generation
"""

import numpy as np
from typing import Tuple, Dict

# Try to import SymForce with proper configuration
try:
    import symforce
    # Try to configure epsilon
    try:
        if hasattr(symforce, 'get_epsilon') and symforce.get_epsilon() == 0:
            symforce.set_epsilon_to_symbol()
    except Exception:
        # Epsilon might already be set or API changed
        pass
    try:
        symforce.set_symbolic_api('sympy')
    except Exception:
        pass
    import symforce.symbolic as sf
    SYMFORCE_AVAILABLE = True
except Exception as e:
    print(f"Warning: SymForce not available: {e}")
    SYMFORCE_AVAILABLE = False


class SymforceFactors:
    """Collection of SymForce-based factor computations"""
    
    if not SYMFORCE_AVAILABLE:
        # Provide fallback implementations
        @staticmethod
        def cone_observation_residual_and_jacobian(*args, **kwargs):
            raise ImportError("SymForce not available")
            
        @staticmethod
        def motion_model_residual_and_jacobian(*args, **kwargs):
            raise ImportError("SymForce not available")
            
        @staticmethod
        def compute_path_smoothness_cost(poses, curvature_weight=1.0, acceleration_weight=1.0):
            # Simple implementation without SymForce
            if len(poses) < 3:
                return 0.0
            total_cost = 0.0
            for i in range(1, len(poses) - 1):
                p0, p1, p2 = poses[i-1], poses[i], poses[i+1]
                v1 = p1[:2] - p0[:2]
                v2 = p2[:2] - p1[:2]
                angle1 = np.arctan2(v1[1], v1[0])
                angle2 = np.arctan2(v2[1], v2[0])
                curvature = np.abs(np.arctan2(np.sin(angle2 - angle1), np.cos(angle2 - angle1)))
                dtheta1 = p1[2] - p0[2]
                dtheta2 = p2[2] - p1[2]
                angular_accel = np.abs(dtheta2 - dtheta1)
                total_cost += curvature_weight * curvature**2
                total_cost += acceleration_weight * angular_accel**2
            return total_cost
    else:
        @staticmethod
        def cone_observation_residual_and_jacobian(
            robot_pose: np.ndarray,  # [x, y, theta]
            landmark_pos: np.ndarray,  # [x, y]
            observation: np.ndarray,  # [x, y] in robot frame
            observed_color: float,
            landmark_color: float,
            color_weight: float = 1.0
        ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
            """
            Compute residual and Jacobians using SymForce symbolic computation
            
            Returns:
                residual: 3D residual [pos_x, pos_y, color]
                jacobians: Dict with 'pose' and 'landmark' Jacobians
            """
            # Create symbolic variables
            pose_x, pose_y, pose_theta = sf.symbols('pose_x pose_y pose_theta')
            landmark_x, landmark_y = sf.symbols('landmark_x landmark_y')
            obs_x, obs_y = sf.symbols('obs_x obs_y')
            obs_color, lm_color = sf.symbols('obs_color lm_color')
            
            # Build transformation matrix symbolically
            cos_theta = sf.cos(pose_theta)
            sin_theta = sf.sin(pose_theta)
            
            # Transform landmark to robot frame
            dx = landmark_x - pose_x
            dy = landmark_y - pose_y
            
            predicted_x = cos_theta * dx + sin_theta * dy
            predicted_y = -sin_theta * dx + cos_theta * dy
            
            # Position residual
            residual_x = predicted_x - obs_x
            residual_y = predicted_y - obs_y
            
            # Color residual
            color_diff = sf.Abs(obs_color - lm_color)
            residual_color = color_weight * sf.Min(color_diff, 1.0)
            
            # Create residual vector
            residual_vec = sf.Matrix([residual_x, residual_y, residual_color])
            
            # Compute Jacobians
            pose_vars = [pose_x, pose_y, pose_theta]
            landmark_vars = [landmark_x, landmark_y]
            
            jacobian_pose = residual_vec.jacobian(pose_vars)
            jacobian_landmark = residual_vec.jacobian(landmark_vars)
            
            # Substitute actual values
            subs_dict = {
                pose_x: robot_pose[0], pose_y: robot_pose[1], pose_theta: robot_pose[2],
                landmark_x: landmark_pos[0], landmark_y: landmark_pos[1],
                obs_x: observation[0], obs_y: observation[1],
                obs_color: observed_color, lm_color: landmark_color
            }
        
            # Evaluate numerically
            residual_num = np.array(residual_vec.subs(subs_dict)).astype(float).flatten()
            jacobian_pose_num = np.array(jacobian_pose.subs(subs_dict)).astype(float)
            jacobian_landmark_num = np.array(jacobian_landmark.subs(subs_dict)).astype(float)
        
            return residual_num, {
                'pose': jacobian_pose_num,
                'landmark': jacobian_landmark_num
            }
        
        @staticmethod
        def motion_model_residual_and_jacobian(
            pose1: np.ndarray,  # [x, y, theta]
            pose2: np.ndarray,  # [x, y, theta]
            odom: np.ndarray,   # [dx, dy, dtheta] relative motion
            wheelbase: float = 0.3  # For Ackermann constraints
        ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
            """
            Compute motion model residual with Ackermann constraints
            """
            # Create symbolic variables
            x1, y1, theta1 = sf.symbols('x1 y1 theta1')
            x2, y2, theta2 = sf.symbols('x2 y2 theta2')
            dx_odom, dy_odom, dtheta_odom = sf.symbols('dx dy dtheta')
            
            # Expected pose after motion
            cos_theta1 = sf.cos(theta1)
            sin_theta1 = sf.sin(theta1)
            
            # Transform odometry to world frame
            expected_x = x1 + cos_theta1 * dx_odom - sin_theta1 * dy_odom
            expected_y = y1 + sin_theta1 * dx_odom + cos_theta1 * dy_odom
            expected_theta = theta1 + dtheta_odom
            
            # Residual
            residual_x = x2 - expected_x
            residual_y = y2 - expected_y
            residual_theta = sf.atan2(sf.sin(theta2 - expected_theta), 
                                      sf.cos(theta2 - expected_theta))
            
            # Add Ackermann constraint penalty
            # Penalize lateral motion (holonomic constraint)
            lateral_motion = -sin_theta1 * (x2 - x1) + cos_theta1 * (y2 - y1)
            ackermann_penalty = 10.0 * lateral_motion
            
            # Create residual vector
            residual_vec = sf.Matrix([residual_x, residual_y, residual_theta, ackermann_penalty])
            
            # Compute Jacobians
            pose1_vars = [x1, y1, theta1]
            pose2_vars = [x2, y2, theta2]
            
            jacobian_pose1 = residual_vec.jacobian(pose1_vars)
            jacobian_pose2 = residual_vec.jacobian(pose2_vars)
            
            # Substitute values
            subs_dict = {
                x1: pose1[0], y1: pose1[1], theta1: pose1[2],
                x2: pose2[0], y2: pose2[1], theta2: pose2[2],
                dx_odom: odom[0], dy_odom: odom[1], dtheta_odom: odom[2]
            }
            
            # Evaluate
            residual_num = np.array(residual_vec.subs(subs_dict)).astype(float).flatten()
            jacobian_pose1_num = np.array(jacobian_pose1.subs(subs_dict)).astype(float)
            jacobian_pose2_num = np.array(jacobian_pose2.subs(subs_dict)).astype(float)
            
            return residual_num, {
                'pose1': jacobian_pose1_num,
                'pose2': jacobian_pose2_num
            }
        
        @staticmethod
        def compute_path_smoothness_cost(
            poses: np.ndarray,  # Nx3 array of poses
            curvature_weight: float = 1.0,
            acceleration_weight: float = 1.0
        ) -> float:
            """
            Compute path smoothness cost considering curvature and angular acceleration
            """
            if len(poses) < 3:
                return 0.0
                
            total_cost = 0.0
            
            for i in range(1, len(poses) - 1):
                # Get three consecutive poses
                p0 = poses[i-1]
                p1 = poses[i]
                p2 = poses[i+1]
                
                # Compute curvature
                v1 = p1[:2] - p0[:2]
                v2 = p2[:2] - p1[:2]
                
                # Angle change
                angle1 = np.arctan2(v1[1], v1[0])
                angle2 = np.arctan2(v2[1], v2[0])
                curvature = np.abs(np.arctan2(np.sin(angle2 - angle1), 
                                             np.cos(angle2 - angle1)))
                
                # Angular acceleration
                dtheta1 = p1[2] - p0[2]
                dtheta2 = p2[2] - p1[2]
                angular_accel = np.abs(dtheta2 - dtheta1)
                
                # Add to cost
                total_cost += curvature_weight * curvature**2
                total_cost += acceleration_weight * angular_accel**2
                
            return total_cost