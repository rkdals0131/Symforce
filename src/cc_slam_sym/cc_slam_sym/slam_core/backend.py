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
from ..utils.async_optimization import AsyncOptimizer


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
    use_robust_kernels: bool = False        # Currently disabled - causes convergence issues
    huber_parameter: float = 1.345          # Huber robust kernel parameter (95% efficiency)
    robust_kernel_type: str = "huber"       # Options: "huber", "cauchy", "tukey"
    
    # Outlier rejection
    chi2_threshold: float = 9.0             # Chi-squared threshold for outlier rejection (99% for 2 DOF)
    remove_outliers: bool = True            # Enable outlier removal after optimization
    
    # Sliding window
    use_sliding_window: bool = True
    max_keyframes: int = 50                 # Maximum keyframes to keep


class SlamBackend:
    """SLAM Backend: constructs and optimizes factor graphs using GTSAM"""
    
    def __init__(self, config: Optional[BackendConfig] = None, logger=None):
        """Initialize SLAM backend
        
        Args:
            config: Backend configuration
            logger: Optional ROS logger for debug output
        """
        self.config = config or BackendConfig()
        self.logger = logger  # For publishing optimization logs
        
        # GTSAM structures
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_values = gtsam.Values()
        self.current_estimate = gtsam.Values()  # Initialize empty instead of None
        
        # Pattern detection for structural constraints
        if self.logger:
            self.logger.info("[BACKEND_INIT_V5] Initializing pattern detector...")
        from .cone_pattern_factor import ConePatternDetector
        self.pattern_detector = ConePatternDetector(logger=self.logger)
        self.detected_patterns = []  # Store detected patterns to avoid duplicates
        if self.logger:
            self.logger.info(f"[BACKEND_INIT_V5] Pattern detector initialized: {self.pattern_detector}")
            self.logger.info(f"[BACKEND_INIT_V5] Backend add_pattern_factors method: {self.add_pattern_factors}")
        
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
        
        # Initialize AsyncOptimizer for non-blocking optimization
        self.async_optimizer = AsyncOptimizer(max_queue_size=3, logger=logger)
        self.pending_optimization = False
        self.last_optimization_request_id = -1
        
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
        
        # GPS reference point (first GPS fix)
        self.gps_reference_utm = None
        self.gps_reference_pose = None
        
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
        self.factors_since_optimization += 1  # Track prior factor
        
        # Add initial value
        self.initial_values.insert(keyframe.pose_symbol, keyframe.pose)
        self.new_values_count += 1  # Track new value
        
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
                if self.logger:
                    self.logger.warning(f"[SLAM_HIGH_VELOCITY] linear={linear_vel:.2f}m/s")
            if angular_vel > max_angular_vel:
                motion_scale *= (angular_vel / max_angular_vel)
                if self.logger:
                    self.logger.warning(f"[SLAM_HIGH_ANGULAR_VELOCITY] angular={angular_vel:.2f}rad/s")
        else:
            motion_scale = 1.0
            
        # Create SymForce motion factor with Ackermann constraints
        position_noise = self.config.odometry_position_noise * motion_scale
        rotation_noise = self.config.odometry_rotation_noise * motion_scale
        
        if self.logger:
            self.logger.debug(f"[SLAM_ADD_ODOMETRY] {keyframe1.pose_symbol}->{keyframe2.pose_symbol}, motion=[{relative_pose.x():.3f},{relative_pose.y():.3f},{relative_pose.theta():.3f}], pos_noise={position_noise:.3f}, rot_noise={rotation_noise:.3f}")
        
        factor = create_symforce_motion_factor(
            keyframe1.pose_symbol,
            keyframe2.pose_symbol,
            relative_pose,
            position_noise=position_noise,
            rotation_noise=rotation_noise,
            wheelbase=1.3  # Formula Student car wheelbase
        )
        self.graph.add(factor)
        self.factors_since_optimization += 1  # INCREMENT FOR ODOMETRY FACTORS TOO!
        
        # Add initial value for new keyframe
        if not self.initial_values.exists(keyframe2.pose_symbol):
            self.initial_values.insert(keyframe2.pose_symbol, keyframe2.pose)
            self.new_values_count += 1  # Track new values too
            
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
                color_weight = 0.0  # Disable color penalty for perfect simulation
            else:
                color_weight = 0.0
        else:
            color_weight = 0.0
            
        # Create SymForce cone observation factor
        if self.logger:
            self.logger.debug(f"[SLAM_ADD_OBSERVATION] pose={keyframe.pose_symbol}, landmark={landmark.symbol}, obs_noise={self.config.landmark_observation_noise}, obs_pos={observation.position[:2].tolist()}")
        
        factor = create_symforce_cone_factor(
            pose_key=keyframe.pose_symbol,
            landmark_key=landmark.symbol,
            observation=observation.position[:2],
            obs_color=observation.color,
            landmark_color=landmark.color,
            position_noise=self.config.landmark_observation_noise,
            color_weight=color_weight,
            use_bearing_range=self.config.use_robust_kernels  # Use as flag for robust kernels
        )
        
        # Add factor to graph
        self.graph.add(factor)
        self.factors_since_optimization += 1
        
        # Track factor for potential outlier removal
        factor_key = self.graph.size() - 1
        self.factor_keys.append((factor_key, 'observation', keyframe.pose_symbol, landmark.symbol))
        
        # Initialize landmark if first observation
        if not self.initial_values.exists(landmark.symbol):
            # Transform observation to world frame using best available keyframe pose
            keyframe_pose = None
            
            # Priority 1: Use optimized pose from current estimate if available
            if self.current_estimate and self.current_estimate.exists(keyframe.pose_symbol):
                keyframe_pose = self.current_estimate.atPose2(keyframe.pose_symbol)
                if self.logger:
                    self.logger.debug(f"[SLAM_LANDMARK_INIT] Using optimized pose for landmark {landmark.symbol}")
            
            # Priority 2: Use initial keyframe pose as fallback
            if keyframe_pose is None and keyframe.pose is not None:
                keyframe_pose = keyframe.pose
                if self.logger:
                    self.logger.debug(f"[SLAM_LANDMARK_INIT] Using initial pose for landmark {landmark.symbol}")
            
            # Priority 3: Error if no pose available
            if keyframe_pose is None:
                raise ValueError(f"No pose available for landmark {landmark.symbol} initialization")
                
            obs_robot = observation.position[:2]
            obs_world = keyframe_pose.transformFrom(obs_robot)
            
            # Check for existing landmarks at this position before initializing
            duplicate_found = False
            for lm_id, existing_lm in self.landmarks.items():
                if lm_id != landmark.id and self.initial_values.exists(existing_lm.symbol):
                    try:
                        existing_pos = self.initial_values.atPoint2(existing_lm.symbol)
                        dist = np.sqrt((existing_pos[0] - obs_world[0])**2 + (existing_pos[1] - obs_world[1])**2)
                        
                        if dist < 0.5:  # 50cm threshold
                            if self.logger:
                                self.logger.warning(f"[SLAM_DUPLICATE_PREVENTED] New landmark {landmark.id} too close to existing {lm_id} "
                                                  f"(dist={dist:.3f}m), skipping initialization")
                            duplicate_found = True
                            # Don't initialize this landmark, merge with existing
                            return
                    except:
                        pass
            
            if not duplicate_found:
                # Initialize landmark position in world frame
                self.initial_values.insert(
                    landmark.symbol,
                    gtsam.Point2(obs_world[0], obs_world[1])
                )
                self.new_values_count += 1
                
                # Update landmark position estimate
                landmark.position = obs_world
            
            if self.logger:
                self.logger.info(f"[SLAM_LANDMARK_CREATED] landmark={landmark.symbol}, world_pos=[{obs_world[0]:.3f},{obs_world[1]:.3f}], keyframe_pose=[{keyframe_pose.x():.3f},{keyframe_pose.y():.3f},{keyframe_pose.theta():.3f}], robot_obs=[{obs_robot[0]:.3f},{obs_robot[1]:.3f}]")
            
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
    
    def test_backend_v5(self):
        """Simple test method to verify backend is accessible"""
        if self.logger:
            self.logger.info("[BACKEND_TEST_V5] Backend test method called successfully!")
        return True
    
    def add_pattern_factors(self, keyframe: Keyframe):
        """Detect and add structural pattern factors for cone observations
        
        Args:
            keyframe: Keyframe with cone observations
        """
        # V5 DEBUG: FORCE LOGGING for debugging
        import logging
        force_logger = logging.getLogger('slam_debug')
        force_logger.info(f"[FORCE_BACKEND_PATTERN_V5] add_pattern_factors CALLED for KF {keyframe.id}")
        force_logger.info(f"[FORCE_BACKEND_PATTERN_V5] self.logger = {self.logger}")
        
        # V5 DEBUG: Log to verify method is entered
        if self.logger:
            self.logger.info(f"[BACKEND_PATTERN_V5] add_pattern_factors CALLED for KF {keyframe.id}")
            self.logger.info(f"[BACKEND_PATTERN_V5] keyframe has {len(keyframe.observations) if hasattr(keyframe, 'observations') else 'NO'} observations")
            self.logger.info(f"[BACKEND_PATTERN_V5] keyframe type: {type(keyframe)}, has observations: {hasattr(keyframe, 'observations')}")
            self.logger.info(f"[SLAM_PATTERN_DETECTION_V5] ENTRY: Called for KF {keyframe.id} with {len(keyframe.observations)} observations")
        else:
            force_logger.error("[FORCE_BACKEND_PATTERN_V5] LOGGER IS NONE!")
            
        if not hasattr(keyframe, 'observations'):
            if self.logger:
                self.logger.error("[BACKEND_PATTERN_V5] ERROR: Keyframe has no observations attribute!")
            return
            
        if len(keyframe.observations) < 3:
            if self.logger:
                self.logger.info(f"[BACKEND_PATTERN_V5] Skipping - only {len(keyframe.observations)} observations")
                self.logger.info(f"[SLAM_PATTERN_DETECTION_V5] SKIP: Not enough observations ({len(keyframe.observations)} < 3)")
            return
            
        # Get cone positions in robot frame
        # V5 FIX: Ensure we extract only 2D positions
        cone_positions = []
        for obs in keyframe.observations:
            if hasattr(obs, 'position') and len(obs.position) >= 2:
                cone_positions.append([float(obs.position[0]), float(obs.position[1])])
        cone_positions = np.array(cone_positions)
        
        if self.logger:
            self.logger.info(f"[SLAM_PATTERN_DETECTION_V5] PROCESSING: {len(cone_positions)} cones from KF {keyframe.id}")
        
        # Detect patterns
        patterns = self.pattern_detector.detect_patterns(cone_positions)
        
        if self.logger:
            self.logger.info(f"[SLAM_PATTERN_RESULT_V3] DETECT RESULT: Found {len(patterns)} patterns from {len(cone_positions)} cone positions")
            for i, pattern in enumerate(patterns):
                self.logger.info(f"[SLAM_PATTERN_V3] Pattern {i}: type={pattern.pattern_type.value}, "
                               f"indices={pattern.cone_indices}, confidence={pattern.confidence:.2f}")
        
        if not patterns:
            if self.logger:
                self.logger.debug("[SLAM_PATTERN_DETECTION] No patterns detected")
            return
            
        if self.logger:
            self.logger.info(f"[SLAM_PATTERN_DETECTION] Found {len(patterns)} patterns")
            
        # Process each detected pattern
        for pattern in patterns:
            # Map observation indices to landmark IDs
            landmark_keys = []
            all_landmarks_exist = True
            
            # V5 FIX: Use association result to map observations to landmarks
            if not hasattr(keyframe, 'association_result') or keyframe.association_result is None:
                if self.logger:
                    self.logger.debug(f"[SLAM_PATTERN_DETECTION_V5] No association result for keyframe {keyframe.id}")
                continue
                
            # Create a mapping from observation index to landmark
            obs_to_landmark = {}
            for obs_idx, lm_idx in keyframe.association_result.matched_pairs:
                # Get the landmark from the local landmarks list
                local_landmarks = list(self.landmarks.values())
                if lm_idx < len(local_landmarks):
                    obs_to_landmark[obs_idx] = local_landmarks[lm_idx]
            
            for cone_idx in pattern.cone_indices:
                if cone_idx in obs_to_landmark:
                    landmark = obs_to_landmark[cone_idx]
                    landmark_keys.append(landmark.symbol)
                else:
                    if self.logger:
                        self.logger.debug(f"[SLAM_PATTERN_DETECTION_V5] No landmark match for observation index {cone_idx}")
                    all_landmarks_exist = False
                    break
                        
            # Only add pattern factor if all landmarks exist
            if all_landmarks_exist and len(landmark_keys) >= 3:
                # Check if this pattern was already added (avoid duplicates)
                pattern_signature = tuple(sorted(landmark_keys))
                if pattern_signature not in self.detected_patterns:
                    if self.logger:
                        self.logger.debug(f"[SLAM_PATTERN_DETECTION] Adding new pattern with landmarks: {[int(k) for k in landmark_keys]}")
                    # Create appropriate noise model based on pattern type
                    from .cone_pattern_factor import ConePatternFactor, PatternType
                    
                    if pattern.pattern_type == PatternType.CORNER_90:
                        noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # Angle, distance ratio
                    elif pattern.pattern_type == PatternType.CURVE:
                        noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # Center x,y, radius
                    elif pattern.pattern_type == PatternType.STRAIGHT:
                        noise_dim = 2 + len(landmark_keys)  # Direction + perpendicular distances
                        noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(noise_dim) * 0.1)
                    else:
                        continue
                        
                    # Add pattern factor
                    pattern_factor = ConePatternFactor(landmark_keys, pattern, noise)
                    self.graph.add(pattern_factor)
                    self.detected_patterns.append(pattern_signature)
                    
                    if self.logger:
                        self.logger.info(f"[SLAM_PATTERN_FACTOR_V3] ADDED FACTOR: {pattern.pattern_type.value} pattern "
                                       f"with {len(landmark_keys)} landmarks, confidence={pattern.confidence:.2f}, "
                                       f"signature={[int(k) for k in pattern_signature]}")
                        self.logger.info(f"[SLAM_PATTERN_V3] Total patterns stored: {len(self.detected_patterns)}")
                else:
                    if self.logger:
                        self.logger.debug(f"[SLAM_PATTERN_DETECTION] Pattern already exists: {[int(k) for k in sorted(landmark_keys)]}")
            else:
                if self.logger:
                    self.logger.debug(f"[SLAM_PATTERN_DETECTION] Skipping pattern - all_landmarks_exist={all_landmarks_exist}, num_landmarks={len(landmark_keys)}")
    
    def add_gps_factor(self, keyframe: Keyframe, gps_data: GpsData):
        """Add GPS position factor to constrain absolute position
        
        Args:
            keyframe: Keyframe to constrain
            gps_data: GPS measurement in UTM coordinates
        """
        if not keyframe.pose_symbol:
            return
            
        # Initialize GPS reference on first measurement
        if self.gps_reference_utm is None:
            self.gps_reference_utm = np.array([gps_data.utm_x, gps_data.utm_y])
            self.gps_reference_pose = keyframe.pose_symbol
            # Add strong prior at GPS reference
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))  # 10cm position, 0.1 rad rotation
            prior_factor = gtsam.PriorFactorPose2(
                keyframe.pose_symbol,
                gtsam.Pose2(0.0, 0.0, 0.0),  # Reference at origin
                prior_noise
            )
            self.graph.add(prior_factor)
            if self.logger:
                self.logger.info(f"[SLAM_GPS_REFERENCE] Set GPS reference at UTM [{gps_data.utm_x:.2f}, {gps_data.utm_y:.2f}]")
            return
            
        # Convert GPS to relative position from reference
        gps_relative = np.array([gps_data.utm_x, gps_data.utm_y]) - self.gps_reference_utm
        
        # Create GPS factor with appropriate noise
        # RTK GPS typical accuracy: 1-2cm horizontal
        gps_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.02, 0.02]))  # 2cm standard deviation
        
        # Create position constraint factor using CustomFactor
        # This constrains the position of the keyframe to match GPS measurement
        target_point = gtsam.Point2(gps_relative[0], gps_relative[1])
        
        position_factor = gtsam.CustomFactor(
            gps_noise,
            [keyframe.pose_symbol],
            lambda pose: self._gps_residual(pose, target_point)
        )
        
        self.graph.add(position_factor)
        self.factors_since_optimization += 1
        
        if self.logger:
            self.logger.debug(f"[SLAM_GPS_FACTOR] Added GPS constraint for keyframe {keyframe.id} at relative position [{gps_relative[0]:.2f}, {gps_relative[1]:.2f}]")
    
    def _gps_residual(self, pose: gtsam.Pose2, target: gtsam.Point2) -> np.ndarray:
        """Compute residual between pose position and GPS target"""
        return np.array([pose.x() - target[0], pose.y() - target[1]])
        
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
            wheelbase=1.3  # Formula Student car wheelbase
        )
        self.graph.add(factor)
        
    def optimize(self, use_async: bool = True) -> bool:
        """Run optimization on the factor graph with performance improvements
        
        Returns:
            True if optimization successful
        """
        try:
            start_time = time.time()
            
            # Check if we have anything to optimize
            if self.logger:
                self.logger.debug(f"[SLAM_OPTIMIZATION_CHECK] factors_since_last={self.factors_since_optimization}, new_values={self.new_values_count}, graph_size={self.graph.size() if self.graph else 0}")
            
            # Skip optimization if no new factors
            if self.factors_since_optimization == 0:
                if self.logger:
                    self.logger.debug("[SLAM_OPTIMIZATION_SKIP] No new factors to optimize")
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
            
            # Debug: Check initial error
            initial_error = self.graph.error(combined_values)
            if self.logger:
                self.logger.debug(f"[SLAM_OPTIMIZATION_START] initial_error={initial_error:.6f}, graph_factors={self.graph.size()}, values={combined_values.size()}")
            
            # Count factor types and compute individual errors
            odometry_factors = 0
            observation_factors = 0
            odometry_error = 0.0
            observation_error = 0.0
            
            for i in range(self.graph.size()):
                factor = self.graph.at(i)
                if factor:
                    try:
                        factor_error = factor.error(combined_values)
                        keys = factor.keys()
                        if len(keys) == 2:
                            key1_type = chr(gtsam.Symbol(keys[0]).chr())
                            key2_type = chr(gtsam.Symbol(keys[1]).chr())
                            
                            if key1_type == 'x' and key2_type == 'x':
                                odometry_factors += 1
                                odometry_error += factor_error
                            elif (key1_type == 'x' and key2_type == 'l') or (key1_type == 'l' and key2_type == 'x'):
                                observation_factors += 1
                                observation_error += factor_error
                    except:
                        pass
            
            if self.logger:
                self.logger.debug(f"[SLAM_FACTOR_BREAKDOWN] odometry_factors={odometry_factors}, odometry_error={odometry_error:.6f}, observation_factors={observation_factors}, observation_error={observation_error:.6f}")
            
            if use_async:
                # Request async optimization
                success = self.async_optimizer.request_optimization(self.graph, combined_values)
                if success:
                    self.pending_optimization = True
                    if self.logger:
                        self.logger.info("[SLAM_ASYNC_OPT] Optimization requested")
                    return True  # Optimization started
                else:
                    if self.logger:
                        self.logger.warning("[SLAM_ASYNC_OPT] Failed to queue optimization")
                    return False
            else:
                # Synchronous optimization (fallback)
                optimizer_params = gtsam.LevenbergMarquardtParams()
                optimizer_params.setVerbosity("ERROR")  # Show optimization progress
                optimizer_params.setMaxIterations(100)
                optimizer_params.setRelativeErrorTol(1e-5)
                optimizer_params.setAbsoluteErrorTol(1e-5)
                
                try:
                    optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, combined_values, optimizer_params)
                    self.current_estimate = optimizer.optimize()
                except Exception as opt_error:
                    if self.logger:
                        self.logger.error(f"[SLAM_OPTIMIZATION_ERROR] error_type={type(opt_error).__name__}, error_msg={str(opt_error)}, graph_size={self.graph.size()}, values_size={combined_values.size()}")
                    raise opt_error
            
            # Debug: Verify current_estimate is properly populated
            if self.logger:
                if self.current_estimate:
                    self.logger.debug(f"[SLAM_OPTIMIZATION_SUCCESS] current_estimate_size={self.current_estimate.size()}")
                else:
                    self.logger.error("[SLAM_OPTIMIZATION_FAILED] current_estimate is None")
            
            # Debug: Check final error and changes
            final_error = self.graph.error(self.current_estimate)
            if self.logger:
                self.logger.info(f"[SLAM_OPTIMIZATION_COMPLETE] initial_error={initial_error:.6f}, final_error={final_error:.6f}, error_reduction={initial_error - final_error:.6f}")
            
            # Check how much poses actually changed
            pose_changes = []
            for key in combined_values.keys():
                if chr(gtsam.Symbol(key).chr()) == 'x':
                    try:
                        initial_pose = combined_values.atPose2(key)
                        final_pose = self.current_estimate.atPose2(key)
                        
                        dx = final_pose.x() - initial_pose.x()
                        dy = final_pose.y() - initial_pose.y()
                        dtheta = final_pose.theta() - initial_pose.theta()
                        
                        change = np.sqrt(dx**2 + dy**2)
                        pose_changes.append(change)
                        
                        if change > 0.01:  # Only log significant changes
                            if self.logger:
                                self.logger.debug(f"[SLAM_POSE_CORRECTION] {gtsam.Symbol(key).string()} moved {change:.3f}m (dx={dx:.3f}, dy={dy:.3f}, dtheta={dtheta:.3f})")
                    except:
                        pass
            
            if pose_changes:
                if self.logger:
                    self.logger.info(f"[SLAM_POSE_CHANGES] mean={np.mean(pose_changes):.3f}m, max={np.max(pose_changes):.3f}m, total_poses_moved={len([x for x in pose_changes if x > 0.01])}")
            else:
                if self.logger:
                    self.logger.warning("[SLAM_NO_POSE_CHANGES] Optimization ran but no poses were corrected!")
            
            # IMPORTANT: Do NOT clear initial_values - they might be needed for new landmarks
            # Instead, update initial_values with optimized values
            for key in self.current_estimate.keys():
                try:
                    if chr(gtsam.Symbol(key).chr()) == 'x':
                        self.initial_values.update(key, self.current_estimate.atPose2(key))
                    elif chr(gtsam.Symbol(key).chr()) == 'l':
                        self.initial_values.update(key, self.current_estimate.atPoint2(key))
                except:
                    pass
            
            # Keep the graph for continuous optimization
            # self.graph = gtsam.NonlinearFactorGraph()  # DON'T clear graph
            
            self.factors_since_optimization = 0
            self.new_values_count = 0
            # self.factor_keys.clear()  # Keep factor tracking for outlier removal
            
            # Keep the optimized estimate - DON'T overwrite with ISAM2
            # self.current_estimate = self.isam2.calculateEstimate()  # This was discarding optimization results!
            
            # Verify current_estimate is still valid after cleanup
            if self.logger:
                self.logger.debug(f"[SLAM_CLEANUP] current_estimate_size={self.current_estimate.size() if self.current_estimate else 0}")
            
            # Remove outliers if enabled
            if self.config.remove_outliers:
                self._remove_outliers()
            
            # Update landmark positions and covariances
            self._update_landmark_estimates()
            
            # Log optimization result
            if self.logger:
                self.logger.debug(f"[SLAM_OPTIMIZATION_VALUES] estimate_size={self.current_estimate.size()}")
            
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
                
                # Rebuild factor graph with only recent keyframes and landmarks
                if len(self.keyframes) >= self.config.max_keyframes:
                    self._rebuild_factor_graph_for_sliding_window()
            
            return True
            
        except Exception as e:
            if self.logger:
                import traceback
                self.logger.error(f"[SLAM_OPTIMIZATION_FAILED_V4] error_type={type(e).__name__}, error_msg={str(e)}")
                self.logger.error(f"[SLAM_OPTIMIZATION_TRACE_V4] {traceback.format_exc()}")
            return False
            
    def _update_landmark_estimates(self):
        """Update landmark positions and covariances from optimization results"""
        if not self.current_estimate:
            return
            
        # Get marginals for covariance computation
        try:
            # Use the factor graph directly since we're using batch optimization
            marginals = gtsam.Marginals(self.graph, self.current_estimate)
        except Exception as e:
            # If marginals computation fails, skip covariance update
            if self.logger:
                self.logger.warning(f"[SLAM_MARGINALS_ERROR] Failed to compute marginals: {e}")
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
        if len(self.keyframes) <= keep_last_n:
            return
            
        # Get keyframes to marginalize
        keyframe_ids = sorted(self.keyframes.keys())
        num_to_marginalize = len(keyframe_ids) - keep_last_n
        keyframes_to_marginalize = keyframe_ids[:num_to_marginalize]
        
        # Remove old keyframes from storage
        for kf_id in keyframes_to_marginalize:
            if kf_id in self.keyframes:
                del self.keyframes[kf_id]
        
        # Also remove old landmarks that are no longer observed
        # (This is a simplified approach - in full SLAM you'd properly marginalize)
        if len(keyframes_to_marginalize) > 0:
            # Get remaining keyframe IDs
            remaining_keyframes = set(self.keyframes.keys())
            landmarks_to_remove = []
            
            for lm_id, landmark in self.landmarks.items():
                # Check if landmark is still observed by remaining keyframes
                still_observed = False
                for kf_id in remaining_keyframes:
                    keyframe = self.keyframes[kf_id]
                    for obs in keyframe.observations:
                        if obs.track_id == landmark.track_id:
                            still_observed = True
                            break
                    if still_observed:
                        break
                
                if not still_observed:
                    landmarks_to_remove.append(lm_id)
            
            # Remove unobserved landmarks
            for lm_id in landmarks_to_remove:
                del self.landmarks[lm_id]
                
            if self.logger:
                self.logger.info(f"[SLAM_MARGINALIZED] keyframes={num_to_marginalize}, landmarks={len(landmarks_to_remove)}")
        
        # Update optimized keyframes list
        self.optimized_keyframes = [kf_id for kf_id in self.optimized_keyframes 
                                   if kf_id in self.keyframes]
    
    def _rebuild_factor_graph_for_sliding_window(self):
        """Rebuild factor graph with only recent keyframes and landmarks for sliding window"""
        if self.logger:
            self.logger.debug("[SLAM_REBUILD_GRAPH] Rebuilding factor graph for sliding window")
        
        # Create new factor graph
        new_graph = gtsam.NonlinearFactorGraph()
        new_initial_values = gtsam.Values()
        
        # Sort keyframes by ID
        keyframe_ids = sorted(self.keyframes.keys())
        
        # Add prior for the oldest remaining keyframe
        if keyframe_ids:
            first_kf = self.keyframes[keyframe_ids[0]]
            
            # Use optimized pose if available, otherwise initial pose
            if self.current_estimate and self.current_estimate.exists(first_kf.pose_symbol):
                pose = self.current_estimate.atPose2(first_kf.pose_symbol)
            else:
                pose = first_kf.pose
            
            # Add prior factor
            noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
                self.config.prior_position_noise,
                self.config.prior_position_noise,
                self.config.prior_rotation_noise
            ]))
            prior_factor = gtsam.PriorFactorPose2(first_kf.pose_symbol, pose, noise)
            new_graph.add(prior_factor)
            new_initial_values.insert(first_kf.pose_symbol, pose)
        
        # Add odometry factors between consecutive keyframes
        for i in range(len(keyframe_ids) - 1):
            kf1 = self.keyframes[keyframe_ids[i]]
            kf2 = self.keyframes[keyframe_ids[i + 1]]
            
            # Get poses
            pose1 = self.current_estimate.atPose2(kf1.pose_symbol) if self.current_estimate and self.current_estimate.exists(kf1.pose_symbol) else kf1.pose
            pose2 = self.current_estimate.atPose2(kf2.pose_symbol) if self.current_estimate and self.current_estimate.exists(kf2.pose_symbol) else kf2.pose
            
            # Calculate relative pose
            relative_pose = pose1.between(pose2)
            
            # Add motion factor
            from .symforce_gtsam_factors_stable import create_symforce_motion_factor
            factor = create_symforce_motion_factor(
                kf1.pose_symbol,
                kf2.pose_symbol,
                relative_pose,
                position_noise=self.config.odometry_position_noise,
                rotation_noise=self.config.odometry_rotation_noise,
                wheelbase=1.3  # Formula Student car wheelbase
            )
            new_graph.add(factor)
            
            # Add pose to initial values
            if not new_initial_values.exists(kf2.pose_symbol):
                new_initial_values.insert(kf2.pose_symbol, pose2)
        
        # Add landmark observation factors
        for lm_id, landmark in self.landmarks.items():
            # Get optimized or initial landmark position
            if self.current_estimate and self.current_estimate.exists(landmark.symbol):
                lm_pos = self.current_estimate.atPoint2(landmark.symbol)
                new_initial_values.insert(landmark.symbol, lm_pos)
            else:
                lm_pos = gtsam.Point2(landmark.position[0], landmark.position[1])
                new_initial_values.insert(landmark.symbol, lm_pos)
            
            # Add observation factors for this landmark
            for kf_id in keyframe_ids:
                keyframe = self.keyframes[kf_id]
                for obs in keyframe.observations:
                    if obs.track_id == landmark.track_id:
                        # Add observation factor
                        from .symforce_gtsam_factors_stable import create_symforce_cone_factor
                        factor = create_symforce_cone_factor(
                            pose_key=keyframe.pose_symbol,
                            landmark_key=landmark.symbol,
                            observation=obs.position[:2],
                            obs_color=obs.color,
                            landmark_color=landmark.color,
                            position_noise=self.config.landmark_observation_noise,
                            color_weight=0.0,  # Disabled for perfect simulation
                            use_bearing_range=False
                        )
                        new_graph.add(factor)
        
        # Replace the old graph and initial values
        self.graph = new_graph
        self.initial_values = new_initial_values
        self.factor_keys = []  # Reset factor tracking
        
        if self.logger:
            self.logger.debug(f"[SLAM_GRAPH_REBUILT] factors={new_graph.size()}, variables={new_initial_values.size()}")
        
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
            
    def _remove_outliers(self, angular_velocity: float = 0.0):
        """Remove factors with high error (outliers) from the graph
        
        Args:
            angular_velocity: Current angular velocity for adaptive thresholding
        """
        if not self.current_estimate or not self.graph:
            return
            
        try:
            factors_to_remove = []
            
            # Check each factor's error in our main graph
            for i in range(self.graph.size()):
                factor = self.graph.at(i)
                if factor is None:
                    continue
                    
                try:
                    # Calculate unwhitened error for this factor
                    error = factor.unwhitenedError(self.current_estimate)
                    chi2_error = np.dot(error, error)
                    
                    # Adaptive threshold based on angular velocity
                    # During turns, we expect higher errors due to uncertainty
                    angular_velocity_scale = 1.0 + abs(angular_velocity) * 0.5  # Up to 2x at 2 rad/s
                    adaptive_threshold = self.config.chi2_threshold * angular_velocity_scale
                    
                    # For observation factors, we have 3 DOF (x, y, color)
                    # Chi-squared threshold at 99% confidence for 3 DOF is ~11.34
                    # For motion factors, we have 4 DOF (x, y, theta, lateral)
                    if chi2_error > adaptive_threshold:
                        factors_to_remove.append(i)
                        
                except Exception as e:
                    # Skip factors that can't compute error
                    if self.logger:
                        self.logger.debug(f"[SLAM_OUTLIER_CHECK] Could not compute error for factor {i}: {e}")
                    continue
            
            # Remove outlier factors
            if factors_to_remove:
                # Create new graph without outliers
                new_graph = gtsam.NonlinearFactorGraph()
                for i in range(self.graph.size()):
                    if i not in factors_to_remove:
                        factor = self.graph.at(i)
                        if factor is not None:
                            new_graph.add(factor)
                
                # Replace the graph with the cleaned version
                self.graph = new_graph
                
                # Update statistics
                self.optimization_stats["outliers_removed"] += len(factors_to_remove)
                if self.logger:
                    self.logger.info(f"[SLAM_OUTLIERS_REMOVED] count={len(factors_to_remove)}, remaining_factors={self.graph.size()}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"[SLAM_OUTLIER_REMOVAL_FAILED] error={e}")
    
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
    
    def check_async_result(self) -> bool:
        """Check if async optimization completed and apply results"""
        if not self.pending_optimization:
            return False
            
        # Try to get result without blocking
        result = self.async_optimizer.get_result(timeout=0.0)
        if result is None:
            return False
            
        self.pending_optimization = False
        
        if result.success:
            # Apply optimized values
            self.current_estimate = result.optimized_values
            
            if self.logger:
                self.logger.info(f"[SLAM_ASYNC_OPT] Applied optimization result: "
                               f"error={result.error:.6f}, time={result.computation_time*1000:.1f}ms")
                
            # Update statistics
            self.optimization_stats['optimization_count'] += 1
            self.optimization_stats['total_time'] += result.computation_time
            self.last_optimization_time = time.time()
            
            # Update landmark estimates from optimization
            self._update_landmark_estimates()
            
            return True
        else:
            if self.logger:
                self.logger.error("[SLAM_ASYNC_OPT] Optimization failed")
            return False