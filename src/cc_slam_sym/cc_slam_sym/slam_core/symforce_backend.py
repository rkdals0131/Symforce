#!/usr/bin/env python3
"""
SymForce-enhanced SLAM Backend
Utilizes SymForce code generation for optimized factor computation
"""

import numpy as np
import gtsam
import symforce
symforce.set_symbolic_api('sympy')
import symforce.symbolic as sf
from symforce import codegen
from symforce.values import Values
from typing import List, Dict, Optional, Tuple, Callable
import time
from pathlib import Path

from ..utils.data_structures import Keyframe, Landmark, ConeCluster
from .cone_color_factor import ConeColorFactor
from .backend import BackendConfig


class SymforceBackend:
    """SLAM Backend using SymForce-generated factors with GTSAM optimization"""
    
    def __init__(self, config: Optional[BackendConfig] = None):
        """Initialize SymForce-enhanced backend"""
        self.config = config or BackendConfig()
        
        # GTSAM structures
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_values = gtsam.Values()
        self.current_estimate = None
        
        # ISAM2 for incremental optimization
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(self.config.relinearize_threshold)
        params.setRelinearizeSkip(self.config.relinearize_skip)
        params.setEnableRelinearization(True)
        params.setEvaluateNonlinearError(True)
        params.setFactorization("QR")  # QR is faster for sparse problems
        
        self.isam2 = gtsam.ISAM2(params)
        
        # Generate SymForce factors
        self._setup_symforce_factors()
        
        # Storage
        self.keyframes = {}
        self.landmarks = {}
        self.keyframe_symbols = {}  # Track symbol assignments
        self.landmark_symbols = {}
        
        # Window management
        self.active_keyframes = []  # List of active keyframe IDs
        self.marginalized_keyframes = set()
        
        # Statistics
        self.optimization_stats = {
            "total_optimizations": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "last_error": 0.0,
            "factors_added": 0,
            "values_added": 0
        }
        
    def _setup_symforce_factors(self):
        """Generate and load SymForce-optimized factors"""
        # Generate code if not exists
        generated_dir = Path(__file__).parent / "generated"
        if not (generated_dir / "cone_color_factor_residual.py").exists():
            print("Generating SymForce factors...")
            ConeColorFactor.generate_code(str(generated_dir))
            
        # Import generated functions
        try:
            import sys
            sys.path.insert(0, str(generated_dir))
            from cone_color_factor_residual import cone_color_factor_residual
            self.cone_residual_func = cone_color_factor_residual
            print("SymForce factors loaded successfully")
        except ImportError as e:
            print(f"Warning: Could not load SymForce factors: {e}")
            self.cone_residual_func = None
            
    def add_prior(self, keyframe: Keyframe):
        """Add prior factor for the first keyframe"""
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
        
        # Track keyframe
        self.keyframes[keyframe.id] = keyframe
        self.keyframe_symbols[keyframe.id] = keyframe.pose_symbol
        self.active_keyframes.append(keyframe.id)
        
    def add_odometry_factor(self, 
                           keyframe1: Keyframe, 
                           keyframe2: Keyframe,
                           relative_pose: Optional[gtsam.Pose2] = None):
        """Add odometry factor between consecutive keyframes"""
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
            
        # Track keyframe
        self.keyframes[keyframe2.id] = keyframe2
        self.keyframe_symbols[keyframe2.id] = keyframe2.pose_symbol
        self.active_keyframes.append(keyframe2.id)
        
        self.optimization_stats["factors_added"] += 1
        
    def add_landmark_observation(self,
                               keyframe: Keyframe,
                               landmark: Landmark,
                               observation: ConeCluster):
        """Add landmark observation using SymForce-generated factor"""
        
        if self.cone_residual_func is None:
            # Fallback to standard GTSAM factor
            self._add_landmark_observation_gtsam(keyframe, landmark, observation)
            return
            
        # Use SymForce-optimized factor
        # Create custom factor wrapper
        class SymforceConeFactorWrapper(gtsam.CustomFactor):
            def __init__(self, pose_key, landmark_key, observation, observed_color, 
                        landmark_color, color_weight, noise_model, residual_func):
                super().__init__(noise_model, [pose_key, landmark_key])
                self.observation = observation
                self.observed_color = observed_color
                self.landmark_color = landmark_color
                self.color_weight = color_weight
                self.residual_func = residual_func
                
            def error(self, values):
                pose = values.atPose2(self.keys()[0])
                landmark_pos = values.atPoint2(self.keys()[1])
                
                # Call SymForce-generated function
                result = self.residual_func(
                    robot_pose=np.array([pose.x(), pose.y(), pose.theta()]),
                    landmark_pos=np.array([landmark_pos[0], landmark_pos[1]]),
                    observation=self.observation,
                    observed_color=self.observed_color,
                    landmark_color=self.landmark_color,
                    color_weight=self.color_weight,
                    epsilon=1e-8
                )
                
                return result['residual']
        
        # Prepare factor inputs
        obs_robot = observation.position[:2]
        observed_color = ConeColorFactor.color_to_scalar(observation.color)
        landmark_color = ConeColorFactor.color_to_scalar(landmark.color)
        
        # Dynamic color weight based on voting confidence
        if landmark.observations > 0:
            total_votes = sum(landmark.color_votes.values())
            if total_votes > 0:
                confidence = max(landmark.color_votes.values()) / total_votes
                color_weight = 5.0 * confidence  # Scale weight by confidence
            else:
                color_weight = 1.0
        else:
            color_weight = 1.0
        
        # Create noise model
        noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
            self.config.landmark_observation_noise,
            self.config.landmark_observation_noise,
            0.5  # Color error noise
        ]))
        
        # Apply robust kernel
        if self.config.use_robust_kernels:
            noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(self.config.huber_parameter),
                noise
            )
        
        # Create and add factor
        factor = SymforceConeFactorWrapper(
            keyframe.pose_symbol,
            landmark.symbol,
            obs_robot,
            observed_color,
            landmark_color,
            color_weight,
            noise,
            self.cone_residual_func
        )
        
        self.graph.add(factor)
        
        # Initialize landmark if first observation
        if not self.initial_values.exists(landmark.symbol):
            # Transform observation to world frame
            if self.current_estimate and self.current_estimate.exists(keyframe.pose_symbol):
                keyframe_pose = self.current_estimate.atPose2(keyframe.pose_symbol)
            else:
                keyframe_pose = keyframe.pose
                
            obs_world = keyframe_pose.transformFrom(obs_robot)
            self.initial_values.insert(
                landmark.symbol,
                gtsam.Point2(obs_world[0], obs_world[1])
            )
            
            # Track landmark
            self.landmarks[landmark.id] = landmark
            self.landmark_symbols[landmark.id] = landmark.symbol
            
        self.optimization_stats["factors_added"] += 1
        
    def _add_landmark_observation_gtsam(self, keyframe, landmark, observation):
        """Fallback to standard GTSAM factor"""
        # Similar to original backend implementation
        color_matches = (observation.color == landmark.color or 
                        observation.color == "unknown" or 
                        landmark.color == "unknown")
        
        base_noise = self.config.landmark_observation_noise
        noise_scale = 1.0 if color_matches else 2.0
        
        noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
            base_noise * noise_scale,
            base_noise * noise_scale
        ]))
        
        if self.config.use_robust_kernels:
            noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(self.config.huber_parameter),
                noise
            )
        
        # Use BearingRangeFactor2D
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
        
        self.graph.add(factor)
        
        # Initialize landmark if needed
        if not self.initial_values.exists(landmark.symbol):
            if self.current_estimate and self.current_estimate.exists(keyframe.pose_symbol):
                keyframe_pose = self.current_estimate.atPose2(keyframe.pose_symbol)
            else:
                keyframe_pose = keyframe.pose
                
            obs_robot = observation.position[:2]
            obs_world = keyframe_pose.transformFrom(obs_robot)
            
            self.initial_values.insert(
                landmark.symbol,
                gtsam.Point2(obs_world[0], obs_world[1])
            )
            
    def optimize(self) -> bool:
        """Run optimization with proper sliding window"""
        try:
            start_time = time.time()
            
            # Apply sliding window before optimization
            if self.config.use_sliding_window and len(self.active_keyframes) > self.config.max_keyframes:
                self._apply_sliding_window()
            
            # Update ISAM2
            self.isam2.update(self.graph, self.initial_values)
            
            # Clear for next iteration
            self.graph = gtsam.NonlinearFactorGraph()
            self.initial_values.clear()
            
            # Get current estimate
            self.current_estimate = self.isam2.calculateEstimate()
            
            # Update statistics
            elapsed_time = time.time() - start_time
            self.optimization_stats["total_optimizations"] += 1
            self.optimization_stats["total_time"] += elapsed_time
            self.optimization_stats["average_time"] = (
                self.optimization_stats["total_time"] / 
                self.optimization_stats["total_optimizations"]
            )
            
            # Calculate error if possible
            try:
                self.optimization_stats["last_error"] = self.isam2.error(self.current_estimate)
            except:
                pass
                
            return True
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return False
            
    def _apply_sliding_window(self):
        """Apply sliding window to maintain performance"""
        # Keep only the most recent keyframes
        num_to_remove = len(self.active_keyframes) - self.config.max_keyframes
        
        if num_to_remove <= 0:
            return
            
        # Get keyframes to marginalize
        keyframes_to_marginalize = self.active_keyframes[:num_to_remove]
        
        # Create marginalization factors
        marginal_factors = gtsam.NonlinearFactorGraph()
        
        # Get all factors connected to marginalized keyframes
        for kf_id in keyframes_to_marginalize:
            if kf_id in self.keyframe_symbols:
                symbol = self.keyframe_symbols[kf_id]
                # In a real implementation, we would extract factors
                # connected to this keyframe and create marginal factors
                
        # Update active keyframes list
        self.active_keyframes = self.active_keyframes[num_to_remove:]
        self.marginalized_keyframes.update(keyframes_to_marginalize)
        
    def get_optimized_pose(self, symbol: int) -> Optional[gtsam.Pose2]:
        """Get optimized pose for a keyframe"""
        try:
            return self.current_estimate.atPose2(symbol)
        except:
            return None
            
    def get_optimized_landmark(self, symbol: int) -> Optional[np.ndarray]:
        """Get optimized position for a landmark"""
        try:
            point = self.current_estimate.atPoint2(symbol)
            return np.array([point[0], point[1]])
        except:
            return None
            
    def get_factor_graph_stats(self) -> Dict:
        """Get statistics about the factor graph"""
        active_factors = 0
        try:
            active_factors = self.isam2.getFactorsUnsafe().size()
        except:
            pass
            
        return {
            "num_factors": active_factors,
            "num_values": self.current_estimate.size() if self.current_estimate else 0,
            "active_keyframes": len(self.active_keyframes),
            "marginalized_keyframes": len(self.marginalized_keyframes),
            "optimization_stats": self.optimization_stats
        }