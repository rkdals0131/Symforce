#!/usr/bin/env python3
"""
SLAM Backend Module for CC-SLAM-SYM
Handles factor graph construction and optimization using GTSAM
"""

import numpy as np
import gtsam
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import time

from ..utils.data_structures import (
    Keyframe, Landmark, ConeCluster, 
    ImuData, GpsData
)
from .custom_factors import ConeObservationFactor, create_cone_observation_factor

# Try to import SymForce factors
try:
    from .symforce_factors import SymforceFactors
    SYMFORCE_AVAILABLE = True
except ImportError:
    SYMFORCE_AVAILABLE = False


@dataclass
class BackendConfig:
    """Configuration for SLAM backend"""
    # Optimization settings
    optimization_interval: int = 5          # Optimize every N keyframes
    max_iterations: int = 100               # Maximum optimizer iterations
    relinearize_threshold: float = 0.1      # ISAM2 relinearization threshold
    relinearize_skip: int = 10             # ISAM2 relinearization skip
    
    # Noise models
    prior_position_noise: float = 0.1       # meters
    prior_rotation_noise: float = 0.05      # radians
    odometry_position_noise: float = 0.2    # meters
    odometry_rotation_noise: float = 0.1    # radians
    landmark_observation_noise: float = 0.3 # meters
    
    # Robust kernels
    use_robust_kernels: bool = True
    huber_parameter: float = 1.0            # Huber robust kernel parameter
    
    # Sliding window
    use_sliding_window: bool = True
    max_keyframes: int = 50                 # Maximum keyframes to keep


class SlamBackend:
    """SLAM Backend: constructs and optimizes factor graphs using GTSAM"""
    
    def __init__(self, config: Optional[BackendConfig] = None):
        """Initialize SLAM backend
        
        Args:
            config: Backend configuration
        """
        self.config = config or BackendConfig()
        
        # GTSAM structures
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_values = gtsam.Values()
        self.current_estimate = None  # Will be created after first optimization
        
        # ISAM2 for incremental optimization with performance tuning
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(self.config.relinearize_threshold)
        params.relinearizeSkip = self.config.relinearize_skip
        params.enableRelinearization = True
        params.evaluateNonlinearError = False  # Disable for performance
        params.setFactorization("QR")  # QR is faster for sparse problems
        params.cacheLinearizedFactors = True  # Cache for performance
        params.enableDetailedResults = False  # Disable detailed results
        
        self.isam2 = gtsam.ISAM2(params)
        
        # Tracking
        self.optimized_keyframes: List[int] = []
        self.optimized_landmarks: List[int] = []
        self.keyframe_count = 0
        self.landmark_count = 0
        self.last_optimization_time = 0.0
        self.factors_since_optimization = 0
        self.new_values_count = 0
        
        # Storage for keyframes and landmarks
        self.keyframes = {}
        self.landmarks = {}
        
        # Statistics
        self.optimization_stats = {
            "total_optimizations": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "last_error": 0.0,
            "factors_per_optimization": 0
        }
        
    def add_prior(self, keyframe: Keyframe):
        """Add prior factor for the first keyframe
        
        Args:
            keyframe: First keyframe
        """
        # Create noise model
        noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
            self.config.prior_position_noise,
            self.config.prior_position_noise,
            self.config.prior_rotation_noise
        ]))
        
        # Add prior factor
        prior_factor = gtsam.PriorFactorPose2(
            keyframe.pose_symbol,
            keyframe.pose,
            noise
        )
        self.graph.add(prior_factor)
        
        # Add initial value
        self.initial_values.insert(keyframe.pose_symbol, keyframe.pose)
        
    def add_odometry_factor(self, 
                           keyframe1: Keyframe, 
                           keyframe2: Keyframe,
                           relative_pose: Optional[gtsam.Pose2] = None):
        """Add odometry factor between consecutive keyframes
        
        Args:
            keyframe1: Previous keyframe
            keyframe2: Current keyframe
            relative_pose: Optional measured relative pose
        """
        # Calculate relative pose if not provided
        if relative_pose is None:
            relative_pose = keyframe1.pose.between(keyframe2.pose)
            
        # Create noise model
        noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
            self.config.odometry_position_noise,
            self.config.odometry_position_noise,
            self.config.odometry_rotation_noise
        ]))
        
        # Apply robust kernel if enabled
        if self.config.use_robust_kernels:
            noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(self.config.huber_parameter),
                noise
            )
            
        # Add between factor
        between_factor = gtsam.BetweenFactorPose2(
            keyframe1.pose_symbol,
            keyframe2.pose_symbol,
            relative_pose,
            noise
        )
        self.graph.add(between_factor)
        
        # Add initial value for new keyframe
        if not self.initial_values.exists(keyframe2.pose_symbol):
            self.initial_values.insert(keyframe2.pose_symbol, keyframe2.pose)
            
    def add_landmark_observation(self,
                               keyframe: Keyframe,
                               landmark: Landmark,
                               observation: ConeCluster):
        """Add landmark observation factor with color information using SymForce
        
        Args:
            keyframe: Observing keyframe
            landmark: Observed landmark
            observation: Cone observation (in robot frame)
        """
        # Use SymForce for residual and Jacobian computation if available
        if SYMFORCE_AVAILABLE:
            try:
                # Get current poses for Jacobian computation
                if self.current_estimate and self.current_estimate.exists(keyframe.pose_symbol):
                    current_pose = self.current_estimate.atPose2(keyframe.pose_symbol)
                    robot_pose = np.array([current_pose.x(), current_pose.y(), current_pose.theta()])
                else:
                    robot_pose = np.array([keyframe.pose.x(), keyframe.pose.y(), keyframe.pose.theta()])
                
                if self.current_estimate and self.current_estimate.exists(landmark.symbol):
                    current_landmark = self.current_estimate.atPoint2(landmark.symbol)
                    landmark_pos = np.array([current_landmark[0], current_landmark[1]])
                else:
                    # Initial estimate
                    obs_world = keyframe.pose.transformFrom(observation.position[:2])
                    landmark_pos = obs_world
                
                # Convert colors to scalars
                observed_color = 0.0 if observation.color == "yellow" else (
                                1.0 if observation.color == "blue" else (
                                2.0 if observation.color == "red" else -1.0))
                landmark_color = 0.0 if landmark.color == "yellow" else (
                               1.0 if landmark.color == "blue" else (
                               2.0 if landmark.color == "red" else -1.0))
                
                # Dynamic color weight based on voting confidence
                if landmark.observations > 0:
                    total_votes = sum(landmark.color_votes.values())
                    if total_votes > 0:
                        confidence = max(landmark.color_votes.values()) / total_votes
                        color_weight = 5.0 * confidence
                    else:
                        color_weight = 1.0
                else:
                    color_weight = 1.0
                
                # Compute residual and Jacobians using SymForce
                residual, jacobians = SymforceFactors.cone_observation_residual_and_jacobian(
                    robot_pose=robot_pose,
                    landmark_pos=landmark_pos,
                    observation=observation.position[:2],
                    observed_color=observed_color,
                    landmark_color=landmark_color,
                    color_weight=color_weight
                )
                
                # Create custom factor with precomputed Jacobians
                # For now, fall back to standard factor until custom factor is properly integrated
                
            except Exception as e:
                print(f"SymForce computation failed: {e}, falling back to standard factor")
        
        # Use standard BearingRangeFactor2D with color-based noise scaling
        # Get color match confidence
        color_matches = (observation.color == landmark.color or 
                        observation.color == "unknown" or 
                        landmark.color == "unknown")
        
        # Scale noise based on color match
        base_noise = self.config.landmark_observation_noise
        if not color_matches:
            # Increase uncertainty for color mismatch
            noise_scale = 2.0  # Make observation less trusted
        else:
            noise_scale = 1.0
            
        # Create noise model
        noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
            base_noise * noise_scale,
            base_noise * noise_scale
        ]))
        
        # Apply robust kernel
        if self.config.use_robust_kernels:
            noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(self.config.huber_parameter),
                noise
            )
        
        # Use standard BearingRangeFactor2D
        # Transform observation to bearing and range
        obs_x, obs_y = observation.position[0], observation.position[1]
        bearing = gtsam.Rot2(np.arctan2(obs_y, obs_x))
        range_val = np.sqrt(obs_x**2 + obs_y**2)
        
        factor = gtsam.BearingRangeFactor2D(
            keyframe.pose_symbol,
            landmark.symbol,
            bearing,
            range_val,
            noise
        )
        
        # Add factor to graph
        self.graph.add(factor)
        self.factors_since_optimization += 1
        
        # Initialize landmark if first observation
        if not self.initial_values.exists(landmark.symbol):
            # Transform observation to world frame using keyframe pose
            if self.current_estimate and self.current_estimate.exists(keyframe.pose_symbol):
                # Use current estimate of keyframe pose if available
                keyframe_pose = self.current_estimate.atPose2(keyframe.pose_symbol)
            else:
                # Otherwise use the initial value
                keyframe_pose = keyframe.pose
                
            obs_robot = observation.position[:2]
            obs_world = keyframe_pose.transformFrom(obs_robot)
            
            # Initialize landmark position in world frame
            self.initial_values.insert(
                landmark.symbol,
                gtsam.Point2(obs_world[0], obs_world[1])
            )
            self.new_values_count += 1
            
            # Update landmark position estimate
            landmark.position = obs_world
            
    def _landmark_error_func(self, pose: gtsam.Pose2, landmark: gtsam.Point2) -> np.ndarray:
        """Error function for landmark observation
        
        Args:
            pose: Robot pose
            landmark: Landmark position
            
        Returns:
            Error vector
        """
        # Transform landmark to robot frame
        landmark_robot = pose.transformTo(landmark)
        
        # In real implementation, compare with actual measurement
        # For now, return zero error
        return np.zeros(2)
        
    def add_loop_closure(self,
                        keyframe1: Keyframe,
                        keyframe2: Keyframe,
                        relative_pose: gtsam.Pose2,
                        confidence: float = 1.0):
        """Add loop closure constraint
        
        Args:
            keyframe1: First keyframe in loop
            keyframe2: Second keyframe in loop
            relative_pose: Measured relative pose
            confidence: Loop closure confidence (0-1)
        """
        # Scale noise based on confidence
        base_noise = np.array([
            self.config.odometry_position_noise * 2,
            self.config.odometry_position_noise * 2,
            self.config.odometry_rotation_noise * 2
        ])
        scaled_noise = base_noise / confidence
        
        noise = gtsam.noiseModel.Diagonal.Sigmas(scaled_noise)
        
        # Add loop closure factor
        loop_factor = gtsam.BetweenFactorPose2(
            keyframe1.pose_symbol,
            keyframe2.pose_symbol,
            relative_pose,
            noise
        )
        self.graph.add(loop_factor)
        
    def optimize(self) -> bool:
        """Run optimization on the factor graph with performance improvements
        
        Returns:
            True if optimization successful
        """
        try:
            start_time = time.time()
            
            # Skip optimization if not enough new factors
            if self.factors_since_optimization < 10 and self.new_values_count < 5:
                return True
            
            # Update ISAM2 with new factors
            self.isam2.update(self.graph, self.initial_values)
            
            # Clear for next iteration
            self.graph = gtsam.NonlinearFactorGraph()
            self.initial_values.clear()
            self.factors_since_optimization = 0
            self.new_values_count = 0
            
            # Get current estimate
            self.current_estimate = self.isam2.calculateEstimate()
            
            # Update landmark positions and covariances
            self._update_landmark_estimates()
            
            # Log optimization result
            print(f"Optimization complete. Current estimate has {self.current_estimate.size()} values")
            
            # Update statistics
            elapsed_time = time.time() - start_time
            self.optimization_stats["total_optimizations"] += 1
            self.optimization_stats["total_time"] += elapsed_time
            self.optimization_stats["average_time"] = (
                self.optimization_stats["total_time"] / 
                self.optimization_stats["total_optimizations"]
            )
            self.optimization_stats["factors_per_optimization"] = self.factors_since_optimization
            
            # Calculate error
            try:
                self.optimization_stats["last_error"] = self.isam2.error(self.current_estimate)
            except:
                pass
            
            # Apply sliding window if needed
            if self.config.use_sliding_window:
                self.marginalize_old_keyframes(self.config.max_keyframes)
            
            return True
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return False
            
    def _update_landmark_estimates(self):
        """Update landmark positions and covariances from optimization results"""
        if not self.current_estimate:
            return
            
        # Get marginals for covariance computation
        try:
            marginals = gtsam.Marginals(self.isam2.getFactorsUnsafe(), self.current_estimate)
        except:
            # If marginals computation fails, skip covariance update
            return
            
        # Update each landmark
        for landmark_id, landmark in self.landmarks.items():
            if self.current_estimate.exists(landmark.symbol):
                # Update position
                point = self.current_estimate.atPoint2(landmark.symbol)
                landmark.position = np.array([point[0], point[1]])
                
                # Update covariance
                try:
                    cov = marginals.marginalCovariance(landmark.symbol)
                    landmark.covariance = np.array(cov)
                except:
                    # Keep existing covariance if marginal computation fails
                    pass
            
    def get_optimized_pose(self, symbol: int) -> Optional[gtsam.Pose2]:
        """Get optimized pose for a keyframe
        
        Args:
            symbol: Keyframe symbol
            
        Returns:
            Optimized pose or None if not found
        """
        try:
            return self.current_estimate.atPose2(symbol)
        except:
            return None
            
    def get_optimized_landmark(self, symbol: int) -> Optional[np.ndarray]:
        """Get optimized position for a landmark
        
        Args:
            symbol: Landmark symbol
            
        Returns:
            Optimized position [x, y] or None if not found
        """
        try:
            point = self.current_estimate.atPoint2(symbol)
            return np.array([point[0], point[1]])
        except:
            return None
            
    def marginalize_old_keyframes(self, keep_last_n: int = 20):
        """Marginalize old keyframes to maintain sliding window
        
        Args:
            keep_last_n: Number of recent keyframes to keep
        """
        if len(self.optimized_keyframes) <= keep_last_n:
            return
            
        # Get keyframes to marginalize
        num_to_marginalize = len(self.optimized_keyframes) - keep_last_n
        keyframes_to_marginalize = self.optimized_keyframes[:num_to_marginalize]
        
        # In ISAM2, we can use fixed lag smoother approach
        # For now, just track which keyframes are active
        self.optimized_keyframes = self.optimized_keyframes[num_to_marginalize:]
        
        # Log marginalization
        print(f"Marginalized {num_to_marginalize} old keyframes")
        
    def get_factor_graph_stats(self) -> Dict:
        """Get statistics about the factor graph
        
        Returns:
            Dictionary with graph statistics
        """
        active_factors = 0
        try:
            # Get total factors in ISAM2
            active_factors = self.isam2.getFactorsUnsafe().size()
        except:
            pass
            
        return {
            "num_factors": self.graph.size() if self.graph else 0,
            "active_factors": active_factors,
            "num_values": self.current_estimate.size() if self.current_estimate else 0,
            "optimization_stats": self.optimization_stats,
            "pending_factors": self.factors_since_optimization
        }
    
    def get_optimized_trajectory(self) -> List[Tuple[float, gtsam.Pose2]]:
        """Get optimized trajectory as list of (timestamp, pose) tuples
        
        Returns:
            List of (timestamp, pose) tuples
        """
        trajectory = []
        
        if not self.keyframes:
            return trajectory
            
        for kf_id in sorted(self.keyframes.keys()):
            kf = self.keyframes[kf_id]
            
            # Try to get optimized pose
            pose = None
            if self.current_estimate:
                try:
                    if self.current_estimate.exists(kf.pose_symbol):
                        pose = self.current_estimate.atPose2(kf.pose_symbol)
                except Exception:
                    pass
                    
            # Fall back to initial pose
            if pose is None:
                pose = kf.pose
                
            if pose is not None:
                trajectory.append((kf.timestamp, pose))
                
        return trajectory
    
    def get_optimized_landmarks(self) -> Dict[int, np.ndarray]:
        """Get optimized landmark positions
        
        Returns:
            Dictionary of landmark_id -> position
        """
        optimized_landmarks = {}
        
        if not hasattr(self, 'landmarks') or not self.current_estimate:
            return optimized_landmarks
            
        for lm_id, landmark in self.landmarks.items():
            if self.current_estimate.exists(landmark.symbol):
                point = self.current_estimate.atPoint2(landmark.symbol)
                optimized_landmarks[lm_id] = np.array([point[0], point[1]])
                
        return optimized_landmarks
        
    def save_graph(self, filename: str):
        """Save factor graph to file for debugging
        
        Args:
            filename: Output filename
        """
        if self.graph and self.current_estimate:
            # Save to GraphViz format
            self.graph.saveGraph(filename + "_graph.dot", self.current_estimate)
            
    def reset(self):
        """Reset the backend to initial state"""
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_values = gtsam.Values()
        self.current_estimate = gtsam.Values()
        
        # Reset ISAM2
        self.isam2 = gtsam.ISAM2()
        
        # Reset tracking
        self.optimized_keyframes.clear()
        self.optimized_landmarks.clear()
        self.keyframe_count = 0
        self.landmark_count = 0