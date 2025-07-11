#!/usr/bin/env python3
"""
Core data structures for CC-SLAM-SYM project.
Based on docs/data_structures.md
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import gtsam
import symforce

# Color enum for cone types
class ConeColor(Enum):
    YELLOW = "yellow"
    BLUE = "blue"
    RED = "red"
    ORANGE = "orange"
    UNKNOWN = "unknown"

class LandmarkType(Enum):
    CONE_YELLOW = "cone_yellow"
    CONE_BLUE = "cone_blue"
    CONE_RED = "cone_red"
    CONE_ORANGE = "cone_orange"
    START_FINISH_LINE = "start_finish_line"

@dataclass
class ConeCluster:
    """Raw cone observation from perception module"""
    timestamp: float
    position: np.ndarray  # 3D position in robot frame [x, y, z]
    color: str  # "yellow", "blue", "red", "orange"
    confidence: float = 0.0  # Detection confidence [0.0, 1.0]
    track_id: int = -1  # Tracking ID from perception
    covariance: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.1)
    
    def is_valid(self) -> bool:
        """Check if observation is valid"""
        return self.confidence > 0.5 and self.track_id >= 0
    
    def distance_to(self, other: 'ConeCluster') -> float:
        """Euclidean distance to another cone"""
        return np.linalg.norm(self.position - other.position)

@dataclass
class Landmark:
    """Map landmark (cone in the global map)"""
    id: int
    symbol: gtsam.Symbol  # GTSAM symbol (L0, L1, ...)
    position: np.ndarray  # 2D position in map frame [x, y]
    color: str
    type: LandmarkType
    
    # Statistics
    observation_count: int = 0
    first_seen_timestamp: float = 0.0
    last_seen_timestamp: float = 0.0
    confidence: float = 0.0
    
    # Uncertainty
    covariance: np.ndarray = field(default_factory=lambda: np.eye(2) * 0.1)
    
    def update_with_observation(self, obs: ConeCluster, current_time: float):
        """Update landmark statistics with new observation"""
        self.observation_count += 1
        self.last_seen_timestamp = current_time
        if self.first_seen_timestamp == 0.0:
            self.first_seen_timestamp = current_time
        
        # Update confidence (simple averaging)
        self.confidence = (self.confidence * (self.observation_count - 1) + obs.confidence) / self.observation_count
    
    def should_remove(self, current_time: float, timeout: float = 30.0) -> bool:
        """Check if landmark should be removed (not seen for too long)"""
        return (current_time - self.last_seen_timestamp) > timeout
    
    def to_gtsam(self) -> gtsam.Point2:
        """Convert to GTSAM Point2"""
        return gtsam.Point2(self.position[0], self.position[1])

@dataclass
class ImuData:
    """IMU measurement data"""
    timestamp: float
    linear_acceleration: np.ndarray  # [ax, ay, az] m/s^2
    angular_velocity: np.ndarray  # [wx, wy, wz] rad/s
    
    # Covariances
    accel_covariance: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.01)
    gyro_covariance: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.001)

@dataclass
class GpsData:
    """GPS/RTK measurement data"""
    timestamp: float
    
    # Position (lat/lon/alt)
    latitude: float
    longitude: float
    altitude: float
    
    # UTM coordinates
    utm_x: float
    utm_y: float
    utm_zone: str
    
    # Velocity
    velocity_enu: np.ndarray = field(default_factory=lambda: np.zeros(3))  # East-North-Up
    
    # Precision
    position_covariance: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.1)
    velocity_covariance: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.1)
    
    # Fix type
    fix_type: int = 0  # 0: NO_FIX, 1: SINGLE, 2: DIFFERENTIAL, 4: RTK_FIXED, 5: RTK_FLOAT
    
    def to_local(self, origin_lat: float, origin_lon: float) -> np.ndarray:
        """Convert to local ENU coordinates relative to origin"""
        # Simplified conversion - in real implementation use proper geodetic conversion
        lat_diff = self.latitude - origin_lat
        lon_diff = self.longitude - origin_lon
        
        # Approximate meters per degree
        meters_per_deg_lat = 111319.5
        meters_per_deg_lon = meters_per_deg_lat * np.cos(np.radians(origin_lat))
        
        x = lon_diff * meters_per_deg_lon
        y = lat_diff * meters_per_deg_lat
        
        return np.array([x, y])
    
    def is_valid(self) -> bool:
        """Check if GPS has valid RTK fix"""
        return self.fix_type >= 5  # RTK_FLOAT or better

@dataclass
class Keyframe:
    """SLAM keyframe"""
    id: int
    timestamp: float
    pose_symbol: gtsam.Symbol  # GTSAM pose symbol (X0, X1, ...)
    
    # State
    pose: gtsam.Pose2  # SE(2) pose (x, y, theta)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [vx, vy, vtheta]
    
    # Sensor data
    observations: List[ConeCluster] = field(default_factory=list)
    imu_data: Optional[ImuData] = None
    gps_data: Optional[GpsData] = None
    
    # Connections
    connected_keyframes: List[int] = field(default_factory=list)
    observed_landmarks: List[int] = field(default_factory=list)
    
    def should_be_keyframe(self, last_kf: 'Keyframe', 
                          trans_threshold: float = 1.0, 
                          rot_threshold: float = 0.2) -> bool:
        """Check if this should be a new keyframe"""
        # Translation distance
        trans_dist = np.linalg.norm(
            [self.pose.x() - last_kf.pose.x(), 
             self.pose.y() - last_kf.pose.y()]
        )
        
        # Rotation distance
        rot_dist = abs(self.pose.theta() - last_kf.pose.theta())
        
        return trans_dist > trans_threshold or rot_dist > rot_threshold
    
    def predict_next_pose(self, dt: float) -> gtsam.Pose2:
        """Predict next pose using constant velocity model"""
        dx = self.velocity[0] * dt
        dy = self.velocity[1] * dt
        dtheta = self.velocity[2] * dt
        
        # Create relative pose
        delta = gtsam.Pose2(dx, dy, dtheta)
        
        # Compose with current pose
        return self.pose.compose(delta)

@dataclass
class DataAssociationResult:
    """Result of data association between observations and landmarks"""
    @dataclass
    class Match:
        landmark_id: int
        observation_idx: int
        distance: float
        mahalanobis_distance: float
        color_match_score: float
    
    matches: List[Match] = field(default_factory=list)
    new_landmark_indices: List[int] = field(default_factory=list)
    outlier_indices: List[int] = field(default_factory=list)
    
    # Statistics
    average_match_distance: float = 0.0
    num_color_mismatches: int = 0

@dataclass
class LoopClosure:
    """Loop closure constraint"""
    query_keyframe_id: int
    match_keyframe_id: int
    timestamp: float
    
    # Transform
    relative_pose: gtsam.Pose2
    covariance: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.1)
    
    # Validation info
    score: float = 0.0
    num_matched_landmarks: int = 0
    landmark_matches: List[Tuple[int, int]] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if loop closure is valid"""
        return self.score > 0.8 and self.num_matched_landmarks >= 5
    
    def to_factor(self) -> gtsam.BetweenFactorPose2:
        """Convert to GTSAM between factor"""
        noise = gtsam.noiseModel.Gaussian.Covariance(self.covariance)
        return gtsam.BetweenFactorPose2(
            gtsam.symbol('x', self.query_keyframe_id),
            gtsam.symbol('x', self.match_keyframe_id),
            self.relative_pose,
            noise
        )

class SlamState:
    """Overall SLAM system state"""
    def __init__(self):
        # Current state
        self.current_pose = gtsam.Pose2()
        self.current_velocity = np.zeros(3)
        self.current_timestamp = 0.0
        
        # Map data
        self.landmarks: Dict[int, Landmark] = {}
        self.keyframes: Dict[int, Keyframe] = {}
        
        # Graph info
        self.num_factors = 0
        self.num_variables = 0
        self.optimization_error = 0.0
        
        # Statistics
        self.total_keyframes = 0
        self.total_landmarks = 0
        self.active_landmarks = 0
        self.loop_closures = 0
        self.mapping_time = 0.0
        self.optimization_time = 0.0
    
    def reset(self):
        """Reset SLAM state"""
        self.__init__()
    
    def save_to_file(self, filename: str):
        """Save state to file"""
        # TODO: Implement serialization
        pass
    
    def load_from_file(self, filename: str):
        """Load state from file"""
        # TODO: Implement deserialization
        pass

# Helper functions for GTSAM symbols
def pose_symbol(idx: int) -> gtsam.Symbol:
    """Create pose symbol (X0, X1, ...)"""
    return gtsam.symbol('x', idx)

def landmark_symbol(idx: int) -> gtsam.Symbol:
    """Create landmark symbol (L0, L1, ...)"""
    return gtsam.symbol('l', idx)

def velocity_symbol(idx: int) -> gtsam.Symbol:
    """Create velocity symbol (V0, V1, ...)"""
    return gtsam.symbol('v', idx)

def bias_symbol(idx: int) -> gtsam.Symbol:
    """Create IMU bias symbol (B0, B1, ...)"""
    return gtsam.symbol('b', idx)