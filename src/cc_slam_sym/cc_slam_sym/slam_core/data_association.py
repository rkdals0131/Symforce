#!/usr/bin/env python3
"""
Data Association Module for CC-SLAM-SYM
Handles matching between cone observations and landmarks in the map
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy.spatial import KDTree
import gtsam

from ..utils.data_structures import ConeCluster, Landmark, ConeColor, LandmarkType


@dataclass
class AssociationResult:
    """Result of data association process"""
    matched_pairs: List[Tuple[int, int]]  # (observation_idx, landmark_idx)
    unmatched_observations: List[int]     # Indices of new cone detections
    unmatched_landmarks: List[int]        # Indices of potentially lost landmarks
    
    
@dataclass
class AssociationConfig:
    """Configuration for data association"""
    max_distance_threshold: float = 2.0   # Maximum distance for association (meters)
    color_match_required: bool = True     # Require color match for association
    use_mahalanobis: bool = False        # Use Mahalanobis distance (future)
    min_observations_for_landmark: int = 2  # Min observations before creating landmark
    

class DataAssociation:
    """Nearest neighbor data association for cone matching"""
    
    def __init__(self, config: Optional[AssociationConfig] = None):
        """Initialize data association module
        
        Args:
            config: Configuration parameters
        """
        self.config = config or AssociationConfig()
        self._observation_counts: Dict[int, int] = {}  # Track observation counts
        
    def associate(self, 
                  observations: List[ConeCluster], 
                  landmarks: List[Landmark],
                  robot_pose: Optional[np.ndarray] = None) -> AssociationResult:
        """Associate cone observations with existing landmarks
        
        Args:
            observations: List of current cone observations
            landmarks: List of existing landmarks in the map
            robot_pose: Current robot pose [x, y, theta] (optional, for future use)
            
        Returns:
            AssociationResult containing matched pairs and unmatched items
        """
        if not observations:
            # No observations, all landmarks are unmatched
            return AssociationResult(
                matched_pairs=[],
                unmatched_observations=[],
                unmatched_landmarks=list(range(len(landmarks)))
            )
            
        if not landmarks:
            # No landmarks yet, all observations are new
            return AssociationResult(
                matched_pairs=[],
                unmatched_observations=list(range(len(observations))),
                unmatched_landmarks=[]
            )
            
        # Extract positions from observations and landmarks
        obs_positions = np.array([[obs.position[0], obs.position[1]] 
                                  for obs in observations])
        lm_positions = np.array([[lm.position[0], lm.position[1]] 
                                 for lm in landmarks])
        
        # Build KD-tree for efficient nearest neighbor search
        kdtree = KDTree(lm_positions)
        
        # Find nearest landmarks for each observation
        distances, nearest_indices = kdtree.query(
            obs_positions, 
            k=1,  # Find single nearest neighbor
            distance_upper_bound=self.config.max_distance_threshold
        )
        
        # Track which landmarks have been matched
        matched_landmark_set = set()
        matched_pairs = []
        unmatched_observations = []
        
        # Process each observation
        for obs_idx, (dist, lm_idx) in enumerate(zip(distances, nearest_indices)):
            # Check if within distance threshold
            if dist < self.config.max_distance_threshold:
                # Check color match if required
                if self.config.color_match_required:
                    obs_color = observations[obs_idx].color
                    lm_color = landmarks[lm_idx].color
                    
                    # Handle unknown colors - allow matching with any color
                    if obs_color == "unknown" or lm_color == "unknown":
                        color_matches = True
                    else:
                        color_matches = (obs_color == lm_color)
                else:
                    color_matches = True
                    
                # Add match if color matches and landmark not already matched
                if color_matches and lm_idx not in matched_landmark_set:
                    matched_pairs.append((obs_idx, lm_idx))
                    matched_landmark_set.add(lm_idx)
                else:
                    unmatched_observations.append(obs_idx)
            else:
                unmatched_observations.append(obs_idx)
                
        # Find unmatched landmarks
        unmatched_landmarks = [i for i in range(len(landmarks)) 
                              if i not in matched_landmark_set]
        
        return AssociationResult(
            matched_pairs=matched_pairs,
            unmatched_observations=unmatched_observations,
            unmatched_landmarks=unmatched_landmarks
        )
        
    def compute_mahalanobis_distance(self,
                                   obs_position: np.ndarray,
                                   lm_position: np.ndarray,
                                   covariance: np.ndarray) -> float:
        """Compute Mahalanobis distance between observation and landmark
        
        Args:
            obs_position: Observation position [x, y]
            lm_position: Landmark position [x, y]
            covariance: 2x2 covariance matrix
            
        Returns:
            Mahalanobis distance
        """
        diff = obs_position - lm_position
        inv_cov = np.linalg.inv(covariance)
        distance = np.sqrt(diff.T @ inv_cov @ diff)
        return distance
        
    def validate_association(self,
                           observation: ConeCluster,
                           landmark: Landmark,
                           distance: float) -> bool:
        """Validate if an association is reasonable
        
        Args:
            observation: Cone observation
            landmark: Existing landmark
            distance: Euclidean distance between them
            
        Returns:
            True if association is valid
        """
        # Basic distance check
        if distance > self.config.max_distance_threshold:
            return False
            
        # Color consistency check
        if self.config.color_match_required:
            if observation.color != "unknown" and landmark.color != "unknown":
                if observation.color != landmark.color:
                    return False
                    
        # Additional validation can be added here:
        # - Chi-square test for Mahalanobis distance
        # - Track consistency over time
        # - Geometric constraints (e.g., track width)
        
        return True
        
    def update_observation_counts(self, unmatched_observations: List[int]):
        """Track how many times we've seen unmatched observations
        
        This helps decide when to create new landmarks
        
        Args:
            unmatched_observations: Indices of unmatched observations
        """
        for idx in unmatched_observations:
            self._observation_counts[idx] = self._observation_counts.get(idx, 0) + 1
            
    def should_create_landmark(self, observation_idx: int) -> bool:
        """Check if an unmatched observation should become a new landmark
        
        Args:
            observation_idx: Index of the observation
            
        Returns:
            True if landmark should be created
        """
        count = self._observation_counts.get(observation_idx, 0)
        return count >= self.config.min_observations_for_landmark