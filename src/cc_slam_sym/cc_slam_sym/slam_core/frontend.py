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
    
    def __init__(self, config: Optional[FrontendConfig] = None):
        """Initialize SLAM frontend
        
        Args:
            config: Frontend configuration
        """
        self.config = config or FrontendConfig()
        self.data_association = DataAssociation(self.config.association_config)
        
        # State tracking
        self.current_pose = gtsam.Pose2(0.0, 0.0, 0.0)
        self.last_keyframe_pose = gtsam.Pose2(0.0, 0.0, 0.0)
        self.last_keyframe_time = 0.0
        
        # Verify pose initialization
        assert self.current_pose is not None, "Current pose not initialized"
        
        # Landmarks and keyframes
        self.landmarks: Dict[int, Landmark] = {}
        self.keyframes: List[Keyframe] = []
        self.next_landmark_id = 0
        self.next_keyframe_id = 0
        
        # Candidate landmarks (not yet confirmed)
        self.candidate_landmarks: Dict[int, List[ConeCluster]] = {}
        
    def process_odometry(self, odom_data: OdometryData) -> gtsam.Pose2:
        """Process odometry data and update current pose estimate
        
        Args:
            odom_data: Odometry measurement
            
        Returns:
            Updated pose estimate
        """
        # Convert odometry to relative motion
        if odom_data.delta_pose is not None:
            # Incremental odometry
            delta_pose = gtsam.Pose2(
                odom_data.delta_pose[0],
                odom_data.delta_pose[1],
                odom_data.delta_pose[2]
            )
            self.current_pose = self.current_pose.compose(delta_pose)
        else:
            # Absolute odometry
            self.current_pose = gtsam.Pose2(
                odom_data.position[0],
                odom_data.position[1],
                odom_data.orientation
            )
            
        return self.current_pose
        
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
                       observations: List[ConeCluster]) -> Optional[Keyframe]:
        """Create a new keyframe
        
        Args:
            timestamp: Keyframe timestamp
            observations: Cone observations at this keyframe
            
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
            observations=observations
        )
        
        # Verify keyframe pose
        if keyframe.pose is None:
            raise ValueError(f"Keyframe {keyframe.id} created with None pose")
        
        # Update tracking
        self.keyframes.append(keyframe)
        self.next_keyframe_id += 1
        self.last_keyframe_pose = self.current_pose
        self.last_keyframe_time = timestamp
        
        return keyframe
        
    def process_cone_observations(self,
                                observations: List[ConeCluster],
                                timestamp: float) -> AssociationResult:
        """Process cone observations and update landmarks
        
        Args:
            observations: List of cone observations
            timestamp: Current timestamp
            
        Returns:
            Data association result
        """
        # Transform observations to world frame
        world_observations = self._transform_to_world(observations)
        
        # Perform data association
        landmark_list = list(self.landmarks.values())
        association_result = self.data_association.associate(
            world_observations,
            landmark_list,
            np.array([self.current_pose.x(), self.current_pose.y(), self.current_pose.theta()])
        )
        
        # Update existing landmarks with matched observations
        for obs_idx, lm_idx in association_result.matched_pairs:
            landmark = landmark_list[lm_idx]
            observation = world_observations[obs_idx]
            landmark.update_with_observation(observation, timestamp)
            
        # Process unmatched observations (potential new landmarks)
        for obs_idx in association_result.unmatched_observations:
            self._process_unmatched_observation(world_observations[obs_idx], timestamp)
            
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
        # Check if observation is within reasonable distance
        distance = np.sqrt(observation.position[0]**2 + observation.position[1]**2)
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
            confidence=np.mean([c.confidence for c in candidates])
        )
        
        # Add to landmarks
        self.landmarks[self.next_landmark_id] = landmark
        self.next_landmark_id += 1
        
        # Clear candidates
        del self.candidate_landmarks[track_id]
        
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