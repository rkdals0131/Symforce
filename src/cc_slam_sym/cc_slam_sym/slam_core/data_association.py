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
import time

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
    use_mahalanobis: bool = True         # Use Mahalanobis distance
    min_observations_for_landmark: int = 2  # Min observations before creating landmark
    base_chi2_threshold: float = 0.90        # Base chi-squared confidence level
    loop_closure_threshold_scale: float = 2.0 # Scale factor for loop closure scenarios
    min_landmarks_for_loop_closure: int = 3  # Min landmarks to consider loop closure
    

class DataAssociation:
    """Nearest neighbor data association for cone matching"""
    
    def __init__(self, config: Optional[AssociationConfig] = None, logger=None):
        """Initialize data association module
        
        Args:
            config: Configuration parameters
            logger: Optional ROS logger for debug output
        """
        self.config = config or AssociationConfig()
        self._observation_counts: Dict[int, int] = {}  # Track observation counts
        self.logger = logger  # For logging debug output
        
        # Track association history for loop closure detection
        self.landmark_creation_times = {}  # landmark_id -> timestamp
        self.landmark_creation_poses = {}  # landmark_id -> robot_pose
        self.last_optimization_time = time.time()
        self.distance_traveled = 0.0
        self.last_pose = None
        
    def associate(self, 
                  observations: List[ConeCluster], 
                  landmarks: List[Landmark],
                  robot_pose: Optional[gtsam.Pose2] = None,
                  predicted_pose: Optional[gtsam.Pose2] = None) -> AssociationResult:
        """Associate cone observations with existing landmarks using Mahalanobis distance
        
        Args:
            observations: List of current cone observations (in robot frame)
            landmarks: List of existing landmarks in the map
            robot_pose: Current robot pose (SE2)
            
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
            
        # Detect potential loop closure based on landmark age
        is_potential_loop_closure = self._detect_loop_closure(landmarks, robot_pose)
        
        # Chi-squared threshold for 2-DOF - adaptive based on loop closure
        from scipy.stats import chi2
        base_threshold = chi2.ppf(self.config.base_chi2_threshold, df=2)
        
        if is_potential_loop_closure:
            chi2_threshold = base_threshold * self.config.loop_closure_threshold_scale
            if self.logger:
                self.logger.info(f"[SLAM_LOOP_CLOSURE] ACTIVATED - threshold={chi2_threshold:.3f} (base={base_threshold:.3f})")
        else:
            chi2_threshold = base_threshold
            if self.logger:
                self.logger.debug(f"[SLAM_ASSOCIATION] Using standard chi-squared threshold: {chi2_threshold:.3f}")
        
        # Track which landmarks have been matched
        matched_landmark_set = set()
        matched_pairs = []
        unmatched_observations = []
        
        # If we have mahalanobis distance enabled
        if self.config.use_mahalanobis and robot_pose is not None:
            # Use predicted pose if available for better matching
            pose_for_association = predicted_pose if predicted_pose is not None else robot_pose
            if self.logger:
                self.logger.debug(f"[SLAM_ASSOCIATION_START] observations={len(observations)}, landmarks={len(landmarks)}, pose=[{pose_for_association.x():.3f},{pose_for_association.y():.3f},{pose_for_association.theta():.3f}]")
            
            # Extract positions for KD-tree (for efficient candidate search)
            lm_positions = np.array([[lm.position[0], lm.position[1]] 
                                     for lm in landmarks])
            kdtree = KDTree(lm_positions)
            
            # Process each observation
            for obs_idx, observation in enumerate(observations):
                # Transform observation to world frame using predicted pose
                obs_world = pose_for_association.transformFrom(observation.position[:2])
                
                # Find candidate landmarks within a reasonable radius
                candidate_indices = kdtree.query_ball_point(
                    obs_world, 
                    r=self.config.max_distance_threshold * 2  # Larger radius for candidates
                )
                
                best_match = None
                min_mahalanobis_dist_sq = float('inf')
                
                for lm_idx in candidate_indices:
                    if lm_idx in matched_landmark_set:
                        continue
                        
                    landmark = landmarks[lm_idx]
                    
                    # Check color compatibility
                    if self.config.color_match_required:
                        if (observation.color != "unknown" and 
                            landmark.color != "unknown" and 
                            observation.color != landmark.color):
                            continue
                    
                    # Calculate innovation (observation - prediction)
                    innovation = obs_world - landmark.position
                    
                    # Combine covariances: observation + landmark
                    # Transform observation covariance to world frame
                    R = pose_for_association.rotation().matrix()  # 2x2 rotation matrix
                    obs_cov_world = R @ observation.covariance[:2, :2] @ R.T
                    
                    # Innovation covariance
                    S = landmark.covariance + obs_cov_world
                    
                    # Mahalanobis distance squared
                    try:
                        S_inv = np.linalg.inv(S)
                        mahalanobis_dist_sq = innovation.T @ S_inv @ innovation
                    except np.linalg.LinAlgError:
                        # Singular matrix, skip this association
                        continue
                    
                    # Enhanced debug logging
                    euclidean_dist = np.linalg.norm(innovation)
                    if self.logger:
                        self.logger.debug(f"[SLAM_ASSOCIATION_CANDIDATE] obs_{obs_idx}->lm_{lm_idx}: euclidean={euclidean_dist:.3f}m, mahalanobis²={mahalanobis_dist_sq:.3f}, threshold={chi2_threshold:.3f}, colors={observation.color}/{landmark.color}")
                    
                    # Debug data for logging
                    log_data = {
                        'obs_idx': obs_idx,
                        'lm_id': landmark.id,
                        'lm_idx': lm_idx,
                        'euclidean_dist': euclidean_dist,
                        'mahalanobis_dist_sq': float(mahalanobis_dist_sq),
                        'threshold': float(chi2_threshold),
                        'is_loop_closure': is_potential_loop_closure
                    }
                    
                    # Check if within statistical threshold and better than current best
                    if (mahalanobis_dist_sq < chi2_threshold and 
                        mahalanobis_dist_sq < min_mahalanobis_dist_sq):
                        min_mahalanobis_dist_sq = mahalanobis_dist_sq
                        best_match = lm_idx
                        if self.logger:
                            self.logger.debug(f"[SLAM_ASSOCIATION_CANDIDATE] obs_{obs_idx}->lm_{lm_idx}: BEST candidate so far")
                        # Mark as candidate
                    else:
                        if mahalanobis_dist_sq >= chi2_threshold:
                            if self.logger:
                                self.logger.debug(f"[SLAM_ASSOCIATION_REJECTED] obs_{obs_idx}->lm_{lm_idx}: exceeds threshold")
                            # Rejected due to threshold
                        else:
                            if self.logger:
                                self.logger.debug(f"[SLAM_ASSOCIATION_REJECTED] obs_{obs_idx}->lm_{lm_idx}: not best match")
                            # Rejected as not best match
                    # Log data handled above with logger
                
                if best_match is not None:
                    matched_pairs.append((obs_idx, best_match))
                    matched_landmark_set.add(best_match)
                    if self.logger:
                        self.logger.debug(f"[SLAM_ASSOCIATION_SUCCESS] obs_{obs_idx}->lm_{best_match}, "
                          f"Mahalanobis²={min_mahalanobis_dist_sq:.3f}")
                    # Success logged above
                else:
                    unmatched_observations.append(obs_idx)
                    if self.logger:
                        self.logger.debug(f"[SLAM_ASSOCIATION_FAILED] obs_{obs_idx}: no valid associations")
                    
        else:
            # Fall back to simple Euclidean distance matching
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
    
    def _detect_loop_closure(self, landmarks: List[Landmark], robot_pose: gtsam.Pose2) -> bool:
        """Detect if we're potentially in a loop closure scenario
        
        Loop closure is likely if:
        1. We see multiple old landmarks (created > 10s ago)
        2. These landmarks were created when robot was far away
        3. We've traveled a significant distance since last optimization
        
        Args:
            landmarks: Current visible landmarks
            robot_pose: Current robot pose
            
        Returns:
            True if loop closure is likely
        """
        if not landmarks or not robot_pose:
            return False
            
        current_time = time.time()
        old_landmark_count = 0
        
        # Update distance traveled
        if self.last_pose is not None:
            delta_pose = self.last_pose.between(robot_pose)
            self.distance_traveled += np.sqrt(delta_pose.x()**2 + delta_pose.y()**2)
        self.last_pose = robot_pose
        
        # Check each landmark
        for landmark in landmarks:
            creation_time = self.landmark_creation_times.get(landmark.id, current_time)
            time_since_creation = current_time - creation_time
            
            # Old landmark (seen more than 10 seconds ago)
            if time_since_creation > 10.0:
                old_landmark_count += 1
                
                # Check distance from creation pose
                if landmark.id in self.landmark_creation_poses:
                    creation_pose = self.landmark_creation_poses[landmark.id]
                    distance_from_creation = np.sqrt(
                        (robot_pose.x() - creation_pose.x())**2 + 
                        (robot_pose.y() - creation_pose.y())**2
                    )
                    
                    if distance_from_creation > 20.0:  # Traveled far since creation
                        if self.logger:
                            self.logger.debug(f"[SLAM_OLD_LANDMARK] lm_{landmark.id}: created {time_since_creation:.1f}s ago, "
                              f"{distance_from_creation:.1f}m away")
        
        # Determine if loop closure
        is_loop_closure = (old_landmark_count >= self.config.min_landmarks_for_loop_closure or
                          self.distance_traveled > 50.0)
        
        if is_loop_closure:
            if self.logger:
                self.logger.info(f"[SLAM_LOOP_CLOSURE_DETECTED] old_landmarks={old_landmark_count}, "
                  f"traveled {self.distance_traveled:.1f}m")
            
            # Loop closure event logged above
            
        return is_loop_closure
    
    def update_landmark_tracking(self, landmark_id: int, timestamp: float, robot_pose: gtsam.Pose2):
        """Update tracking information when a new landmark is created
        
        Args:
            landmark_id: ID of the newly created landmark
            timestamp: Creation timestamp
            robot_pose: Robot pose at creation time
        """
        self.landmark_creation_times[landmark_id] = timestamp
        self.landmark_creation_poses[landmark_id] = robot_pose
        
    def reset_optimization_tracking(self):
        """Reset tracking after optimization"""
        self.last_optimization_time = time.time()
        self.distance_traveled = 0.0
        
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