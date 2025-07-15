#!/usr/bin/env python3
"""
Custom GTSAM factors for CC-SLAM-SYM
Implements factors that incorporate cone color information
"""

import gtsam
import numpy as np
from typing import List, Optional


def cone_color_to_int(color: str) -> int:
    """Convert cone color string to integer encoding
    
    Args:
        color: Color string ("yellow", "blue", "red", "unknown")
        
    Returns:
        Integer encoding (0: yellow, 1: blue, 2: red, 3: unknown)
    """
    color_map = {
        "yellow": 0,
        "blue": 1,
        "red": 2,
        "unknown": 3
    }
    return color_map.get(color.lower(), 3)


class ConeObservationFactor(gtsam.NonlinearFactor):
    """
    Custom factor for cone observations that includes color information.
    
    This factor penalizes color mismatches between observations and landmarks,
    providing additional constraints for robust data association.
    """
    
    def __init__(self, 
                 pose_key: int,
                 landmark_key: int, 
                 observed_position: np.ndarray,
                 observed_color: str,
                 landmark_color: str,
                 noise_model: gtsam.noiseModel.Base,
                 color_mismatch_penalty: float = 10.0,
                 color_confidence: float = 1.0):
        """
        Initialize cone observation factor
        
        Args:
            pose_key: Key for robot pose
            landmark_key: Key for landmark position
            observed_position: Observed cone position in robot frame [x, y]
            observed_color: Observed cone color
            landmark_color: Expected landmark color
            noise_model: 3D noise model (x, y, color)
            color_mismatch_penalty: Base penalty for color mismatch
            color_confidence: Confidence in landmark color classification (0-1)
        """
        # Initialize base class with keys
        gtsam.NonlinearFactor.__init__(self, noise_model, [pose_key, landmark_key])
        
        self.observed_position = gtsam.Point2(observed_position[0], observed_position[1])
        self.observed_color_int = cone_color_to_int(observed_color)
        self.landmark_color_int = cone_color_to_int(landmark_color)
        self.color_mismatch_penalty = color_mismatch_penalty
        self.color_confidence = color_confidence
        
    def error(self, values: gtsam.Values) -> np.ndarray:
        """
        Calculate error between predicted and observed cone position with color penalty
        
        Args:
            values: Current estimates for all variables
            
        Returns:
            3D error vector [position_error_x, position_error_y, color_error]
        """
        # Get pose and landmark estimates
        pose_key = self.keys()[0]
        landmark_key = self.keys()[1]
        
        pose = values.atPose2(pose_key)
        landmark_pos = values.atPoint2(landmark_key)
        
        # Transform landmark to robot frame
        predicted_position = pose.transformTo(landmark_pos)
        
        # Position error
        position_error = predicted_position - self.observed_position
        
        # Color error - only penalize if colors don't match and neither is unknown
        color_error = 0.0
        if (self.observed_color_int != 3 and  # Not unknown
            self.landmark_color_int != 3 and   # Not unknown  
            self.observed_color_int != self.landmark_color_int):
            # Scale penalty by color confidence
            # High confidence = full penalty, low confidence = reduced penalty
            color_error = self.color_mismatch_penalty * self.color_confidence
            
        # Return 3D error vector
        return np.array([position_error[0], position_error[1], color_error])


class RobustConeObservationFactor(ConeObservationFactor):
    """
    Robust version of cone observation factor using adaptive weights.
    
    This factor dynamically adjusts the color penalty based on the
    current state of the optimization to handle outliers better.
    """
    
    def __init__(self,
                 pose_key: int,
                 landmark_key: int,
                 observed_position: np.ndarray,
                 observed_color: str,
                 landmark_color: str,
                 noise_model: gtsam.noiseModel.Base,
                 initial_color_penalty: float = 10.0,
                 min_color_penalty: float = 1.0,
                 max_color_penalty: float = 100.0):
        """
        Initialize robust cone observation factor
        
        Args:
            pose_key: Key for robot pose
            landmark_key: Key for landmark position
            observed_position: Observed cone position in robot frame [x, y]
            observed_color: Observed cone color
            landmark_color: Expected landmark color
            noise_model: 3D noise model (x, y, color)
            initial_color_penalty: Initial penalty for color mismatch
            min_color_penalty: Minimum allowed color penalty
            max_color_penalty: Maximum allowed color penalty
        """
        super().__init__(pose_key, landmark_key, observed_position,
                        observed_color, landmark_color, noise_model,
                        initial_color_penalty)
        
        self.min_color_penalty = min_color_penalty
        self.max_color_penalty = max_color_penalty
        self.iteration_count = 0
        
    def error(self, values: gtsam.Values) -> np.ndarray:
        """
        Calculate error with adaptive color penalty
        
        Args:
            values: Current estimates for all variables
            
        Returns:
            3D error vector with adaptive color penalty
        """
        # Get base error
        base_error = super().error(values)
        
        # Adapt color penalty based on position error magnitude
        position_error_magnitude = np.linalg.norm(base_error[:2])
        
        # If position error is large, reduce color penalty to allow convergence
        # If position error is small, increase color penalty for precision
        if position_error_magnitude > 1.0:  # Large position error
            adaptive_penalty = self.min_color_penalty
        elif position_error_magnitude < 0.1:  # Small position error
            adaptive_penalty = self.max_color_penalty
        else:  # Interpolate
            t = (1.0 - position_error_magnitude) / 0.9
            adaptive_penalty = (self.min_color_penalty * (1 - t) + 
                              self.max_color_penalty * t)
        
        # Apply adaptive penalty only if there's a color mismatch
        if base_error[2] > 0:
            base_error[2] = adaptive_penalty
            
        return base_error


def create_cone_observation_factor(pose: gtsam.Pose2,
                                 landmark: gtsam.Point2,
                                 observation: np.ndarray,
                                 obs_color: str,
                                 landmark_color: str,
                                 position_noise: float = 0.1,
                                 color_weight: float = 0.5,
                                 color_confidence: float = 1.0,
                                 use_robust: bool = False) -> gtsam.NonlinearFactor:
    """
    Factory function to create appropriate cone observation factor
    
    Args:
        pose: Robot pose key
        landmark: Landmark position key  
        observation: Observed position in robot frame
        obs_color: Observed color
        landmark_color: Expected landmark color
        position_noise: Position measurement noise (meters)
        color_weight: Relative weight of color vs position
        color_confidence: Confidence in landmark color classification (0-1)
        use_robust: Whether to use robust adaptive version
        
    Returns:
        Configured cone observation factor
    """
    # Create noise model with position and color components
    noise_sigmas = np.array([
        position_noise,      # x position
        position_noise,      # y position  
        color_weight         # color component
    ])
    noise_model = gtsam.noiseModel.Diagonal.Sigmas(noise_sigmas)
    
    # Apply robust kernel for outlier rejection
    huber_parameter = 1.0
    robust_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(huber_parameter),
        noise_model
    )
    
    # Create appropriate factor type
    if use_robust:
        return RobustConeObservationFactor(
            pose, landmark, observation, obs_color, landmark_color,
            robust_noise, initial_color_penalty=10.0 / color_weight
        )
    else:
        return ConeObservationFactor(
            pose, landmark, observation, obs_color, landmark_color,
            robust_noise, 
            color_mismatch_penalty=10.0 / color_weight,
            color_confidence=color_confidence
        )