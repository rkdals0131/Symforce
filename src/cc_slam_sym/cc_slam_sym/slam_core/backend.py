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

# Import SymForce-GTSAM integrated factors
from .symforce_gtsam_factors_stable import (
    create_symforce_cone_factor,
    create_symforce_motion_factor,
    color_string_to_float
)


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
    
    # Outlier rejection
    chi2_threshold: float = 9.0             # Chi-squared threshold for outlier rejection (99% for 2 DOF)
    remove_outliers: bool = True            # Enable outlier removal after optimization
    
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
            "factors_per_optimization": 0,
            "outliers_removed": 0
        }
        
        # Track factors for outlier removal
        self.factor_keys = []  # Store factor indices
        
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
        """Add odometry factor between consecutive keyframes using SymForce
        
        Args:
            keyframe1: Previous keyframe
            keyframe2: Current keyframe
            relative_pose: Optional measured relative pose
        """
        # Calculate relative pose if not provided
        if relative_pose is None:
            relative_pose = keyframe1.pose.between(keyframe2.pose)
            
        # Check for unreasonable motion (vehicle constraints)
        dt = keyframe2.timestamp - keyframe1.timestamp
        if dt > 0:
            # Extract motion components
            dx = relative_pose.x()
            dy = relative_pose.y()
            dtheta = relative_pose.theta()
            
            # Calculate velocities
            linear_vel = np.sqrt(dx**2 + dy**2) / dt
            angular_vel = abs(dtheta) / dt
            
            # Vehicle constraints (reasonable for a car)
            max_linear_vel = 10.0  # m/s (36 km/h)
            max_angular_vel = 2.0  # rad/s (about 115 deg/s)
            
            # Scale noise if motion seems unreasonable
            motion_scale = 1.0
            if linear_vel > max_linear_vel:
                motion_scale *= (linear_vel / max_linear_vel)
                print(f"Warning: High linear velocity {linear_vel:.2f} m/s")
            if angular_vel > max_angular_vel:
                motion_scale *= (angular_vel / max_angular_vel)
                print(f"Warning: High angular velocity {angular_vel:.2f} rad/s")
        else:
            motion_scale = 1.0
            
        # Create SymForce motion factor with Ackermann constraints
        factor = create_symforce_motion_factor(
            keyframe1.pose_symbol,
            keyframe2.pose_symbol,
            relative_pose,
            position_noise=self.config.odometry_position_noise * motion_scale,
            rotation_noise=self.config.odometry_rotation_noise * motion_scale,
            wheelbase=0.3  # Formula Student car wheelbase
        )
        self.graph.add(factor)
        
        # Add initial value for new keyframe
        if not self.initial_values.exists(keyframe2.pose_symbol):
            self.initial_values.insert(keyframe2.pose_symbol, keyframe2.pose)
            
    def add_landmark_observation(self,
                               keyframe: Keyframe,
                               landmark: Landmark,
                               observation: ConeCluster):
        """Add landmark observation factor using SymForce-generated functions
        
        Args:
            keyframe: Observing keyframe
            landmark: Observed landmark
            observation: Cone observation (in robot frame)
        """
        # Get color confidence from landmark voting
        if landmark.observation_count > 0:
            total_votes = sum(landmark.color_votes.values())
            if total_votes > 0:
                confidence = max(landmark.color_votes.values()) / total_votes
                color_weight = 5.0 * confidence  # Scale weight by confidence
            else:
                color_weight = 1.0
        else:
            color_weight = 1.0
            
        # Create SymForce cone observation factor
        factor = create_symforce_cone_factor(
            pose_key=keyframe.pose_symbol,
            landmark_key=landmark.symbol,
            observation=observation.position[:2],
            obs_color=observation.color,
            landmark_color=landmark.color,
            position_noise=self.config.landmark_observation_noise,
            color_weight=color_weight,
            use_bearing_range=False  # Use Cartesian formulation
        )
        
        # Add factor to graph
        self.graph.add(factor)
        self.factors_since_optimization += 1
        
        # Track factor for potential outlier removal
        factor_key = self.graph.size() - 1
        self.factor_keys.append((factor_key, 'observation', keyframe.pose_symbol, landmark.symbol))
        
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
        """Add loop closure constraint using SymForce motion factor
        
        Args:
            keyframe1: First keyframe in loop
            keyframe2: Second keyframe in loop
            relative_pose: Measured relative pose
            confidence: Loop closure confidence (0-1)
        """
        # Scale noise based on confidence (lower confidence = higher noise)
        position_noise = self.config.odometry_position_noise * 2 / confidence
        rotation_noise = self.config.odometry_rotation_noise * 2 / confidence
        
        # Create SymForce motion factor for loop closure
        factor = create_symforce_motion_factor(
            keyframe1.pose_symbol,
            keyframe2.pose_symbol,
            relative_pose,
            position_noise=position_noise,
            rotation_noise=rotation_noise,
            wheelbase=0.3  # Same as odometry
        )
        self.graph.add(factor)
        
    def optimize(self) -> bool:
        """Run optimization on the factor graph with performance improvements
        
        Returns:
            True if optimization successful
        """
        try:
            start_time = time.time()
            
            # Skip optimization if not enough new factors
            # Only require a minimal number to avoid empty optimizations
            if self.factors_since_optimization < 2:
                return True
            
            # Use batch optimization instead of ISAM2 due to CustomFactor issues
            # Build combined values
            combined_values = gtsam.Values()
            
            # Add current estimates
            if self.current_estimate:
                for key in self.current_estimate.keys():
                    try:
                        if chr(gtsam.Symbol(key).chr()) == 'x':
                            combined_values.insert(key, self.current_estimate.atPose2(key))
                        elif chr(gtsam.Symbol(key).chr()) == 'l':
                            combined_values.insert(key, self.current_estimate.atPoint2(key))
                    except:
                        pass
            
            # Add new values
            for key in self.initial_values.keys():
                if not combined_values.exists(key):
                    try:
                        if chr(gtsam.Symbol(key).chr()) == 'x':
                            combined_values.insert(key, self.initial_values.atPose2(key))
                        elif chr(gtsam.Symbol(key).chr()) == 'l':
                            combined_values.insert(key, self.initial_values.atPoint2(key))
                    except:
                        pass
            
            # Batch optimization
            optimizer_params = gtsam.LevenbergMarquardtParams()
            optimizer_params.setVerbosity("SILENT")
            optimizer_params.setMaxIterations(100)
            optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, combined_values, optimizer_params)
            self.current_estimate = optimizer.optimize()
            
            # Clear for next iteration
            self.graph = gtsam.NonlinearFactorGraph()
            self.initial_values.clear()
            self.factors_since_optimization = 0
            self.new_values_count = 0
            self.factor_keys.clear()
            
            # Get current estimate
            self.current_estimate = self.isam2.calculateEstimate()
            
            # Remove outliers if enabled
            if self.config.remove_outliers:
                self._remove_outliers()
            
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
                self.optimization_stats["last_error"] = self.graph.error(self.current_estimate)
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
            
    def _remove_outliers(self):
        """Remove factors with high error (outliers) from the graph"""
        if not self.current_estimate:
            return
            
        try:
            # Get all factors from ISAM2
            graph = self.isam2.getFactorsUnsafe()
            
            factors_to_remove = []
            
            # Check each factor's error
            for i in range(graph.size()):
                factor = graph.at(i)
                if factor is None:
                    continue
                    
                try:
                    # Calculate unwhitened error for this factor
                    error = factor.unwhitenedError(self.current_estimate)
                    chi2_error = np.dot(error, error)
                    
                    # For bearing-range factors, we have 2 DOF
                    # Chi-squared threshold at 99% confidence for 2 DOF is ~9.21
                    if chi2_error > self.config.chi2_threshold:
                        factors_to_remove.append(i)
                        
                except Exception:
                    # Skip factors that can't compute error
                    continue
            
            # Remove outlier factors
            if factors_to_remove:
                # Create new graph without outliers
                new_graph = gtsam.NonlinearFactorGraph()
                for i in range(graph.size()):
                    if i not in factors_to_remove:
                        factor = graph.at(i)
                        if factor is not None:
                            new_graph.add(factor)
                
                # Re-initialize ISAM2 with cleaned graph
                # Need to recreate params since ISAM2.params() might not work
                params = gtsam.ISAM2Params()
                params.setRelinearizeThreshold(self.config.relinearize_threshold)
                params.relinearizeSkip = self.config.relinearize_skip
                params.enableRelinearization = True
                params.evaluateNonlinearError = False
                params.setFactorization("QR")
                params.cacheLinearizedFactors = True
                params.enableDetailedResults = False
                
                self.isam2 = gtsam.ISAM2(params)
                self.isam2.update(new_graph, self.current_estimate)
                
                # Update statistics
                self.optimization_stats["outliers_removed"] += len(factors_to_remove)
                print(f"Removed {len(factors_to_remove)} outlier factors")
                
        except Exception as e:
            print(f"Outlier removal failed: {e}")
    
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
        self.factor_keys.clear()