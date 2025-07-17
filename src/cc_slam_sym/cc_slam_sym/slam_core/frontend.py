#!/usr/bin/env python3
"""
SLAM Frontend Module for CC-SLAM-SYM
Handles odometry processing, keyframe selection, and landmark initialization
"""

import numpy as np
import gtsam
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
import time

from ..utils.data_structures import (
    ConeCluster, Landmark, Keyframe, OdometryData, 
    ConeColor, LandmarkType
)
from .data_association import DataAssociation, AssociationConfig, AssociationResult
from .local_map import LocalMap, LocalMapConfig


@dataclass
class FrontendConfig:
    """Configuration for SLAM frontend"""
    # Keyframe selection thresholds
    keyframe_distance_threshold: float = 2.0     # meters
    keyframe_rotation_threshold: float = 0.5     # radians (~30 degrees)
    keyframe_time_threshold: float = 1.0         # seconds
    
    # Data association
    association_config: AssociationConfig = field(default_factory=AssociationConfig)
    
    # Landmark initialization
    min_observations_for_landmark: int = 2
    max_landmark_init_distance: float = 20.0     # Maximum distance to initialize landmark
    
    # Noise models
    odometry_position_noise: float = 0.1         # meters
    odometry_rotation_noise: float = 0.05        # radians
    landmark_position_noise: float = 0.2         # meters


class SlamFrontend:
    """SLAM Frontend: processes sensor data and prepares it for the backend"""
    
    def __init__(self, config: Optional[FrontendConfig] = None, logger=None):
        """Initialize SLAM frontend
        
        Args:
            config: Frontend configuration
            logger: Optional ROS logger for debug output
        """
        self.config = config or FrontendConfig()
        self.logger = logger
        
        # Pass logger to data association
        self.data_association = DataAssociation(self.config.association_config, logger)
        
        # State tracking
        self.current_pose = gtsam.Pose2(0.0, 0.0, 0.0)
        self.last_keyframe_pose = gtsam.Pose2(0.0, 0.0, 0.0)
        self.last_keyframe_time = 0.0
        
        # Odometry prediction for data association
        self.last_odom_pose = gtsam.Pose2(0.0, 0.0, 0.0)
        self.predicted_pose = gtsam.Pose2(0.0, 0.0, 0.0)
        
        # Verify pose initialization
        assert self.current_pose is not None, "Current pose not initialized"
        
        # Landmarks and keyframes
        self.landmarks: Dict[int, Landmark] = {}
        self.keyframes: List[Keyframe] = []
        self.next_landmark_id = 0
        self.next_keyframe_id = 0
        
        # Candidate landmarks (not yet confirmed)
        self.candidate_landmarks: Dict[int, List[ConeCluster]] = {}
        
        # Local map for efficient data association
        local_map_config = LocalMapConfig(
            max_keyframes=20,
            max_landmarks=100,
            temporal_window=10.0,
            spatial_radius=20.0
        )
        self.local_map = LocalMap(local_map_config)
        
    def process_odometry(self, odom_data: OdometryData) -> gtsam.Pose2:
        """Process odometry data and update current pose estimate
        
        Args:
            odom_data: Odometry measurement
            
        Returns:
            Updated pose estimate
        """
        # Store previous pose for motion prediction
        self.last_odom_pose = self.current_pose
        
        # Convert odometry to relative motion
        if odom_data.delta_pose is not None:
            # Incremental odometry
            delta_pose = gtsam.Pose2(
                odom_data.delta_pose[0],
                odom_data.delta_pose[1],
                odom_data.delta_pose[2]
            )
            self.current_pose = self.current_pose.compose(delta_pose)
            
            # Predict next pose for data association
            self.predicted_pose = self.current_pose.compose(delta_pose)
        else:
            # Absolute odometry
            self.current_pose = gtsam.Pose2(
                odom_data.position[0],
                odom_data.position[1],
                odom_data.orientation
            )
            
            # Calculate motion for prediction
            if self.last_odom_pose:
                motion = self.last_odom_pose.between(self.current_pose)
                self.predicted_pose = self.current_pose.compose(motion)
            else:
                self.predicted_pose = self.current_pose
        
        # Update local map with current pose
        self.local_map.update_current_pose(self.current_pose, odom_data.timestamp)
            
        return self.current_pose
    
    def update_pose_from_backend(self, optimized_pose: gtsam.Pose2):
        """Update current pose with optimized estimate from backend
        
        Args:
            optimized_pose: Optimized pose from SLAM backend
        """
        print(f"Frontend: Updating pose from backend optimization")
        print(f"  Before: x={self.current_pose.x():.3f}, y={self.current_pose.y():.3f}, theta={self.current_pose.theta():.3f}")
        print(f"  After:  x={optimized_pose.x():.3f}, y={optimized_pose.y():.3f}, theta={optimized_pose.theta():.3f}")
        
        # Update current pose with optimized estimate
        self.current_pose = optimized_pose
        
        # Update last keyframe pose if it exists
        if self.keyframes:
            latest_keyframe = self.keyframes[-1]
            if latest_keyframe.pose_symbol:
                # Don't update keyframe pose - it will be handled by backend
                pass
        
    def should_create_keyframe(self, current_time: float) -> bool:
        """Check if a new keyframe should be created
        
        Args:
            current_time: Current timestamp
            
        Returns:
            True if keyframe should be created
        """
        # Time-based criterion
        time_diff = current_time - self.last_keyframe_time
        if time_diff > self.config.keyframe_time_threshold:
            return True
            
        # Distance-based criterion
        relative_pose = self.last_keyframe_pose.between(self.current_pose)
        translation = np.sqrt(relative_pose.x()**2 + relative_pose.y()**2)
        if translation > self.config.keyframe_distance_threshold:
            return True
            
        # Rotation-based criterion
        rotation = abs(relative_pose.theta())
        if rotation > self.config.keyframe_rotation_threshold:
            return True
            
        return False
        
    def create_keyframe(self, 
                       timestamp: float,
                       observations: List[ConeCluster],
                       association_result: Optional[AssociationResult] = None) -> Optional[Keyframe]:
        """Create a new keyframe
        
        Args:
            timestamp: Keyframe timestamp
            observations: Cone observations at this keyframe
            association_result: Optional association result for this keyframe
            
        Returns:
            Created keyframe or None if creation failed
        """
        # Ensure current pose is valid
        if self.current_pose is None:
            raise ValueError("Current pose is None when creating keyframe")
            
        # Create keyframe
        keyframe = Keyframe(
            id=self.next_keyframe_id,
            timestamp=timestamp,
            pose_symbol=gtsam.symbol('x', self.next_keyframe_id),
            pose=self.current_pose,
            observations=observations,
            association_result=association_result  # Store association result
        )
        
        # Verify keyframe pose
        if keyframe.pose is None:
            raise ValueError(f"Keyframe {keyframe.id} created with None pose")
        
        # Update tracking
        self.keyframes.append(keyframe)
        self.next_keyframe_id += 1
        self.last_keyframe_pose = self.current_pose
        self.last_keyframe_time = timestamp
        
        # Add to local map
        self.local_map.add_keyframe(keyframe)
        
        return keyframe
        
    def process_cone_observations(self,
                                observations: List[ConeCluster],
                                timestamp: float) -> AssociationResult:
        """Process cone observations and update landmarks
        
        Args:
            observations: List of cone observations (in robot frame)
            timestamp: Current timestamp
            
        Returns:
            Data association result
        """
        # Use local map for efficient data association
        local_landmarks = self.local_map.get_nearby_landmarks()
        
        # Perform data association with observations in robot frame
        association_result = self.data_association.associate(
            observations,  # Pass observations in robot frame
            local_landmarks,
            self.current_pose,  # Current pose
            self.predicted_pose  # Predicted pose for better matching
        )
        
        # Update existing landmarks with matched observations
        for obs_idx, lm_idx in association_result.matched_pairs:
            landmark = local_landmarks[lm_idx]
            observation = observations[obs_idx]
            # Transform observation to world frame for landmark update
            obs_world_pos = self.current_pose.transformFrom(observation.position[:2])
            world_obs = ConeCluster(
                timestamp=observation.timestamp,
                position=np.array([obs_world_pos[0], obs_world_pos[1], observation.position[2]]),
                color=observation.color,
                confidence=observation.confidence,
                track_id=observation.track_id,
                covariance=observation.covariance
            )
            landmark.update_with_observation(world_obs, timestamp)
            
        # Process unmatched observations (potential new landmarks)
        for obs_idx in association_result.unmatched_observations:
            observation = observations[obs_idx]
            # Transform to world frame
            obs_world_pos = self.current_pose.transformFrom(observation.position[:2])
            world_obs = ConeCluster(
                timestamp=observation.timestamp,
                position=np.array([obs_world_pos[0], obs_world_pos[1], observation.position[2]]),
                color=observation.color,
                confidence=observation.confidence,
                track_id=observation.track_id,
                covariance=observation.covariance
            )
            self._process_unmatched_observation(world_obs, timestamp)
            
        return association_result
        
    def _transform_to_world(self, observations: List[ConeCluster]) -> List[ConeCluster]:
        """Transform cone observations from robot frame to world frame
        
        Args:
            observations: Observations in robot frame
            
        Returns:
            Observations in world frame
        """
        world_observations = []
        
        for obs in observations:
            # Transform position
            robot_point = np.array([obs.position[0], obs.position[1]])
            world_point = self.current_pose.transformFrom(robot_point)
            
            # Create transformed observation
            world_obs = ConeCluster(
                timestamp=obs.timestamp,
                position=np.array([world_point[0], world_point[1], obs.position[2]]),
                color=obs.color,
                confidence=obs.confidence,
                track_id=obs.track_id,
                covariance=obs.covariance
            )
            world_observations.append(world_obs)
            
        return world_observations
        
    def _process_unmatched_observation(self, observation: ConeCluster, timestamp: float):
        """Process an unmatched observation (potential new landmark)
        
        Args:
            observation: Unmatched cone observation (in world frame)
            timestamp: Current timestamp
        """
        # Check if observation is within reasonable distance from current robot position
        robot_to_obs = observation.position[:2] - np.array([self.current_pose.x(), self.current_pose.y()])
        distance = np.linalg.norm(robot_to_obs)
        if distance > self.config.max_landmark_init_distance:
            return
            
        # Track candidate landmarks
        track_id = observation.track_id
        if track_id not in self.candidate_landmarks:
            self.candidate_landmarks[track_id] = []
        self.candidate_landmarks[track_id].append(observation)
        
        # Check if enough observations to create landmark
        if len(self.candidate_landmarks[track_id]) >= self.config.min_observations_for_landmark:
            self._create_landmark_from_candidates(track_id, timestamp)
            
    def _create_landmark_from_candidates(self, track_id: int, timestamp: float):
        """Create a new landmark from candidate observations
        
        Args:
            track_id: Tracking ID of the candidate
            timestamp: Current timestamp
        """
        candidates = self.candidate_landmarks[track_id]
        
        # Average position and determine color
        positions = np.array([c.position[:2] for c in candidates])
        avg_position = np.mean(positions, axis=0)
        
        # Determine most common color
        colors = [c.color for c in candidates]
        color = max(set(colors), key=colors.count)
        
        # Determine landmark type based on color
        type_map = {
            "yellow": LandmarkType.CONE_YELLOW,
            "blue": LandmarkType.CONE_BLUE,
            "red": LandmarkType.CONE_RED,
        }
        landmark_type = type_map.get(color, LandmarkType.CONE_YELLOW)
        
        # Create landmark
        landmark = Landmark(
            id=self.next_landmark_id,
            symbol=gtsam.symbol('l', self.next_landmark_id),
            position=avg_position,
            color=color,
            type=landmark_type,
            observation_count=len(candidates),
            first_seen_timestamp=candidates[0].timestamp,
            last_seen_timestamp=timestamp,
            confidence=np.mean([c.confidence for c in candidates]),
            track_id=track_id  # Store the original track ID
        )
        
        # Add to landmarks
        self.landmarks[self.next_landmark_id] = landmark
        self.next_landmark_id += 1
        
        # Add to local map
        self.local_map.add_landmark(landmark)
        
        # Update data association tracking for loop closure detection
        self.data_association.update_landmark_tracking(
            landmark.id, 
            timestamp, 
            self.current_pose
        )
        
        # Clear candidates
        del self.candidate_landmarks[track_id]
        
        # Log landmark creation
        if self.logger:
            self.logger.info(f"[SLAM_LANDMARK_CREATED] id={landmark.id}, track_id={track_id}, position=[{avg_position[0]:.2f},{avg_position[1]:.2f}], color={color}, observations={len(candidates)}")
        
        # Check for nearby landmarks (potential duplicates)
        nearby_landmarks = self.local_map.get_nearby_landmarks(position=avg_position, radius=2.0)
        for nearby_lm in nearby_landmarks:
            if nearby_lm.id != landmark.id and nearby_lm.color == color:
                dist = np.linalg.norm(nearby_lm.position - avg_position)
                if dist < 1.0 and self.logger:
                    self.logger.warning(f"[SLAM_DUPLICATE_WARNING] New landmark {landmark.id} created {dist:.3f}m from existing landmark {nearby_lm.id} (color={color})")
        
        
    def get_odometry_noise_model(self) -> gtsam.noiseModel.Diagonal:
        """Get noise model for odometry factors
        
        Returns:
            GTSAM noise model
        """
        return gtsam.noiseModel.Diagonal.Sigmas(np.array([
            self.config.odometry_position_noise,
            self.config.odometry_position_noise,
            self.config.odometry_rotation_noise
        ]))
        
    def get_landmark_noise_model(self) -> gtsam.noiseModel.Diagonal:
        """Get noise model for landmark observation factors
        
        Returns:
            GTSAM noise model
        """
        return gtsam.noiseModel.Diagonal.Sigmas(np.array([
            self.config.landmark_position_noise,
            self.config.landmark_position_noise
        ]))
        
    def get_state_summary(self) -> Dict:
        """Get summary of frontend state
        
        Returns:
            Dictionary with state information
        """
        return {
            "current_pose": {
                "x": self.current_pose.x(),
                "y": self.current_pose.y(),
                "theta": self.current_pose.theta()
            },
            "num_keyframes": len(self.keyframes),
            "num_landmarks": len(self.landmarks),
            "num_candidates": len(self.candidate_landmarks)
        }