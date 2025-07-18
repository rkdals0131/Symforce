#!/usr/bin/env python3
"""
Local Map Module for CC-SLAM-SYM
Maintains a sliding window of recent landmarks for efficient data association
Inspired by GLIM's submap approach
"""

import numpy as np
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
import time

from ..utils.data_structures import Landmark, Keyframe
import gtsam


@dataclass
class LocalMapConfig:
    """Configuration for local map management"""
    max_keyframes: int = 20  # Maximum keyframes in local map
    max_landmarks: int = 200  # Maximum landmarks in local map
    temporal_window: float = 10.0  # Time window in seconds
    spatial_radius: float = 30.0  # Spatial radius in meters
    
    # Marginalization thresholds
    keyframe_removal_distance: float = 50.0  # Distance to remove old keyframes
    keyframe_removal_time: float = 30.0  # Time to remove old keyframes


class LocalMap:
    """Manages a local window of the map for efficient processing"""
    
    def __init__(self, config: Optional[LocalMapConfig] = None):
        """Initialize local map
        
        Args:
            config: Local map configuration
        """
        self.config = config or LocalMapConfig()
        
        # Local window storage
        self.local_keyframes: Dict[int, Keyframe] = {}
        self.local_landmarks: Dict[int, Landmark] = {}
        
        # Current robot pose (for distance calculations)
        self.current_pose: Optional[gtsam.Pose2] = None
        self.current_time: float = 0.0
        
        # Active landmark IDs (for quick lookup)
        self.active_landmark_ids: Set[int] = set()
        
    def update_current_pose(self, pose: gtsam.Pose2, timestamp: float):
        """Update current robot pose for distance calculations
        
        Args:
            pose: Current robot pose
            timestamp: Current timestamp
        """
        self.current_pose = pose
        self.current_time = timestamp
        
    def add_keyframe(self, keyframe: Keyframe):
        """Add a keyframe to the local map
        
        Args:
            keyframe: Keyframe to add
        """
        self.local_keyframes[keyframe.id] = keyframe
        
        # Marginalize old keyframes if needed
        self._marginalize_old_keyframes()
        
    def add_landmark(self, landmark: Landmark):
        """Add or update a landmark in the local map
        
        Args:
            landmark: Landmark to add/update
        """
        self.local_landmarks[landmark.id] = landmark
        self.active_landmark_ids.add(landmark.id)
        
        # Check if we need to remove distant landmarks
        self._remove_distant_landmarks()
    
    def update_all_landmarks(self, landmarks: Dict[int, Landmark]):
        """Update all landmarks with optimized positions from backend
        
        Args:
            landmarks: Dictionary of all landmarks with updated positions
        """
        # Update existing landmarks in local map
        for lm_id, updated_landmark in landmarks.items():
            if lm_id in self.local_landmarks:
                # Update position and covariance
                self.local_landmarks[lm_id].position = updated_landmark.position.copy()
                if hasattr(updated_landmark, 'covariance') and updated_landmark.covariance is not None:
                    self.local_landmarks[lm_id].covariance = updated_landmark.covariance.copy()
                    
        # Add any new landmarks that might have been created
        for lm_id, landmark in landmarks.items():
            if lm_id not in self.local_landmarks:
                self.add_landmark(landmark)
        
    def get_nearby_landmarks(self, 
                           position: Optional[np.ndarray] = None,
                           radius: Optional[float] = None) -> List[Landmark]:
        """Get landmarks within a certain radius
        
        Args:
            position: Query position (uses current pose if None)
            radius: Search radius (uses config value if None)
            
        Returns:
            List of nearby landmarks
        """
        if position is None and self.current_pose is not None:
            position = np.array([self.current_pose.x(), self.current_pose.y()])
        elif position is None:
            return list(self.local_landmarks.values())
            
        if radius is None:
            radius = self.config.spatial_radius
            
        nearby = []
        for landmark in self.local_landmarks.values():
            dist = np.linalg.norm(landmark.position - position)
            if dist <= radius:
                nearby.append(landmark)
                
        return nearby
        
    def get_recent_landmarks(self, time_window: Optional[float] = None) -> List[Landmark]:
        """Get landmarks observed within a time window
        
        Args:
            time_window: Time window in seconds (uses config value if None)
            
        Returns:
            List of recent landmarks
        """
        if time_window is None:
            time_window = self.config.temporal_window
            
        cutoff_time = self.current_time - time_window
        
        recent = []
        for landmark in self.local_landmarks.values():
            if landmark.last_seen_timestamp >= cutoff_time:
                recent.append(landmark)
                
        return recent
        
    def _marginalize_old_keyframes(self):
        """Remove keyframes that are too old or too far"""
        if not self.current_pose:
            return
            
        to_remove = []
        
        for kf_id, keyframe in self.local_keyframes.items():
            # Time-based removal
            time_diff = self.current_time - keyframe.timestamp
            if time_diff > self.config.keyframe_removal_time:
                to_remove.append(kf_id)
                continue
                
            # Distance-based removal
            kf_position = np.array([keyframe.pose.x(), keyframe.pose.y()])
            current_position = np.array([self.current_pose.x(), self.current_pose.y()])
            distance = np.linalg.norm(kf_position - current_position)
            
            if distance > self.config.keyframe_removal_distance:
                to_remove.append(kf_id)
                
        # Remove old keyframes
        for kf_id in to_remove:
            del self.local_keyframes[kf_id]
            
        # Keep only the most recent N keyframes
        if len(self.local_keyframes) > self.config.max_keyframes:
            sorted_kfs = sorted(self.local_keyframes.items(), 
                              key=lambda x: x[1].timestamp)
            n_to_remove = len(self.local_keyframes) - self.config.max_keyframes
            
            for kf_id, _ in sorted_kfs[:n_to_remove]:
                del self.local_keyframes[kf_id]
                
    def _remove_distant_landmarks(self):
        """Remove landmarks that are too far from current position"""
        if not self.current_pose:
            return
            
        current_position = np.array([self.current_pose.x(), self.current_pose.y()])
        to_remove = []
        
        # Check distance for each landmark
        for lm_id, landmark in self.local_landmarks.items():
            distance = np.linalg.norm(landmark.position - current_position)
            
            # Remove if too far and not recently observed
            if (distance > self.config.spatial_radius * 2 and 
                self.current_time - landmark.last_seen_timestamp > 5.0):
                to_remove.append(lm_id)
                
        # Remove distant landmarks
        for lm_id in to_remove:
            del self.local_landmarks[lm_id]
            self.active_landmark_ids.discard(lm_id)
            
        # Keep only the most recent N landmarks if exceeded
        if len(self.local_landmarks) > self.config.max_landmarks:
            sorted_lms = sorted(self.local_landmarks.items(),
                              key=lambda x: x[1].last_seen_timestamp)
            n_to_remove = len(self.local_landmarks) - self.config.max_landmarks
            
            for lm_id, _ in sorted_lms[:n_to_remove]:
                del self.local_landmarks[lm_id]
                self.active_landmark_ids.discard(lm_id)
                
    def get_covisible_landmarks(self, keyframe: Keyframe) -> List[Landmark]:
        """Get landmarks that are potentially visible from a keyframe
        
        Args:
            keyframe: Query keyframe
            
        Returns:
            List of potentially visible landmarks
        """
        kf_position = np.array([keyframe.pose.x(), keyframe.pose.y()])
        
        # Use a larger radius for covisibility check
        covisible_radius = self.config.spatial_radius * 1.5
        
        covisible = []
        for landmark in self.local_landmarks.values():
            # Check if within viewing distance
            dist = np.linalg.norm(landmark.position - kf_position)
            if dist <= covisible_radius:
                # Additional check: was it observed recently?
                time_diff = abs(landmark.last_seen_timestamp - keyframe.timestamp)
                if time_diff < self.config.temporal_window:
                    covisible.append(landmark)
                    
        return covisible
        
    def get_state_summary(self) -> Dict:
        """Get summary of local map state
        
        Returns:
            Dictionary with state information
        """
        return {
            "num_keyframes": len(self.local_keyframes),
            "num_landmarks": len(self.local_landmarks),
            "active_landmarks": len(self.active_landmark_ids),
            "oldest_keyframe_age": self._get_oldest_keyframe_age(),
            "spatial_extent": self._calculate_spatial_extent()
        }
        
    def _get_oldest_keyframe_age(self) -> float:
        """Get age of oldest keyframe in seconds"""
        if not self.local_keyframes:
            return 0.0
            
        oldest_time = min(kf.timestamp for kf in self.local_keyframes.values())
        return self.current_time - oldest_time
        
    def _calculate_spatial_extent(self) -> float:
        """Calculate spatial extent of local map"""
        if not self.local_landmarks:
            return 0.0
            
        positions = np.array([lm.position for lm in self.local_landmarks.values()])
        if len(positions) < 2:
            return 0.0
            
        # Calculate bounding box diagonal
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)
        return np.linalg.norm(max_pos - min_pos)