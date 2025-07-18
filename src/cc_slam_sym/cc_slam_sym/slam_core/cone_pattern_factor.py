#!/usr/bin/env python3
"""
Cone Pattern Factor for CC-SLAM-SYM
Adds structural constraints based on cone patterns (corners, curves, etc.)
"""

import numpy as np
from typing import List, Tuple, Optional
import gtsam
from dataclasses import dataclass
from enum import Enum

class PatternType(Enum):
    """Types of cone patterns"""
    CORNER_90 = "corner_90"      # 90-degree corner
    CORNER_ACUTE = "corner_acute" # Acute angle corner
    CURVE = "curve"               # Curved section
    STRAIGHT = "straight"         # Straight line
    CHICANE = "chicane"          # S-curve pattern

@dataclass
class ConePattern:
    """Detected cone pattern"""
    pattern_type: PatternType
    cone_indices: List[int]      # Indices of cones in this pattern
    confidence: float            # Pattern detection confidence
    parameters: dict             # Pattern-specific parameters (radius for curve, angle for corner, etc.)

class ConePatternDetector:
    """Detects geometric patterns in cone observations"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.min_cones_for_pattern = {
            PatternType.CORNER_90: 3,
            PatternType.CORNER_ACUTE: 3,
            PatternType.CURVE: 4,
            PatternType.STRAIGHT: 3,
            PatternType.CHICANE: 5
        }
    
    def detect_patterns(self, cone_positions: np.ndarray) -> List[ConePattern]:
        """Detect all patterns in a set of cone positions
        
        Args:
            cone_positions: Nx2 array of cone positions in robot frame
            
        Returns:
            List of detected patterns
        """
        patterns = []
        n_cones = len(cone_positions)
        
        if n_cones < 3:
            return patterns
            
        # Detect corners (3 consecutive cones)
        for i in range(n_cones - 2):
            corner = self._check_corner(cone_positions[i:i+3])
            if corner:
                patterns.append(corner)
                
        # Detect curves (4+ consecutive cones)
        for i in range(n_cones - 3):
            for j in range(i + 4, min(i + 8, n_cones + 1)):  # Check up to 8 cones
                curve = self._check_curve(cone_positions[i:j])
                if curve:
                    patterns.append(curve)
                    
        # Detect straight lines (3+ consecutive cones)
        for i in range(n_cones - 2):
            for j in range(i + 3, min(i + 6, n_cones + 1)):
                straight = self._check_straight(cone_positions[i:j])
                if straight:
                    patterns.append(straight)
                    
        return patterns
    
    def _check_corner(self, positions: np.ndarray) -> Optional[ConePattern]:
        """Check if 3 cones form a corner"""
        if len(positions) != 3:
            return None
            
        # V5 FIX: Ensure positions are 2D
        positions_2d = positions[:, :2] if positions.shape[1] > 2 else positions
            
        # Calculate vectors
        v1 = positions_2d[1] - positions_2d[0]
        v2 = positions_2d[2] - positions_2d[1]
        
        # Calculate angle
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 0.1 or norm2 < 0.1:  # Too short vectors
            return None
            
        angle = np.arccos(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1, 1))
        angle_deg = np.degrees(angle)
        
        if self.logger:
            self.logger.debug(f"[SLAM_PATTERN_CORNER] Checking corner: angle={angle_deg:.1f}°, v1_len={norm1:.2f}m, v2_len={norm2:.2f}m")
        
        # Check for 90-degree corner (±30 degrees tolerance - more lenient)
        if 60 <= angle_deg <= 120:
            return ConePattern(
                pattern_type=PatternType.CORNER_90,
                cone_indices=[0, 1, 2],
                confidence=1.0 - abs(90 - angle_deg) / 30.0,
                parameters={'angle': angle_deg}
            )
        # Check for acute corner
        elif 20 <= angle_deg <= 60:
            return ConePattern(
                pattern_type=PatternType.CORNER_ACUTE,
                cone_indices=[0, 1, 2],
                confidence=0.8,
                parameters={'angle': angle_deg}
            )
            
        return None
    
    def _check_curve(self, positions: np.ndarray) -> Optional[ConePattern]:
        """Check if cones form a curve using circle fitting"""
        if len(positions) < 4:
            return None
            
        # V5 FIX: Ensure positions are 2D
        positions_2d = positions[:, :2] if positions.shape[1] > 2 else positions
            
        # Fit circle to points
        center, radius, residual = self._fit_circle(positions_2d)
        
        if self.logger:
            self.logger.debug(f"[SLAM_PATTERN_CURVE] Circle fit: radius={radius:.2f}m, residual={residual:.3f}, n_points={len(positions)}")
        
        # Check if fit is good (low residual) - more lenient threshold
        if residual < 0.5 and radius > 2.0:  # Average distance from circle < 50cm, radius > 2m
            return ConePattern(
                pattern_type=PatternType.CURVE,
                cone_indices=list(range(len(positions))),
                confidence=1.0 - residual * 2,  # Scale to 0-1
                parameters={'center': center, 'radius': radius}
            )
            
        return None
    
    def _check_straight(self, positions: np.ndarray) -> Optional[ConePattern]:
        """Check if cones form a straight line"""
        if len(positions) < 3:
            return None
            
        # V5 FIX: Ensure positions are 2D
        positions_2d = positions[:, :2] if positions.shape[1] > 2 else positions
            
        # Fit line and calculate residuals
        mean_pos = np.mean(positions_2d, axis=0)
        centered = positions_2d - mean_pos
        
        # PCA to find main direction
        _, _, vt = np.linalg.svd(centered.T, full_matrices=False)
        direction = vt[0]
        # V5 FIX: Ensure direction is 2D
        if len(direction) > 2:
            direction = direction[:2]
        
        # Calculate perpendicular distances
        distances = []
        for pos in centered:
            proj = np.dot(pos, direction) * direction
            perp_dist = np.linalg.norm(pos - proj)
            distances.append(perp_dist)
            
        avg_dist = np.mean(distances)
        max_dist = np.max(distances) if distances else float('inf')
        
        if self.logger:
            self.logger.debug(f"[SLAM_PATTERN_STRAIGHT] Line fit: avg_dist={avg_dist:.3f}m, max_dist={max_dist:.3f}m, n_points={len(positions)}")
        
        # Check if points are collinear - more lenient threshold
        if avg_dist < 0.3:  # Average perpendicular distance < 30cm
            return ConePattern(
                pattern_type=PatternType.STRAIGHT,
                cone_indices=list(range(len(positions))),
                confidence=1.0 - avg_dist * 3.33,  # Scale to 0-1
                parameters={'direction': direction, 'mean_position': mean_pos}
            )
            
        return None
    
    def _fit_circle(self, positions: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Fit a circle to 2D points using least squares
        
        Returns:
            center: Circle center [x, y]
            radius: Circle radius
            residual: Average distance from points to circle
        """
        if len(positions) < 3:
            return np.array([0, 0]), 1.0, float('inf')
            
        # Use centroid-based circle fitting for better stability
        centroid = np.mean(positions, axis=0)
        centered_positions = positions - centroid
        
        # Calculate radius as average distance from centroid
        distances = np.linalg.norm(centered_positions, axis=1)
        radius = np.mean(distances)
        
        if radius < 0.1:  # Too small radius
            return centroid, radius, float('inf')
        
        # Calculate residual
        residual = np.std(distances)  # Standard deviation as residual
        
        return centroid, radius, residual


class ConePatternFactor(gtsam.CustomFactor):
    """Custom factor for cone pattern constraints"""
    
    def __init__(self, 
                 landmark_keys: List[int],
                 pattern: ConePattern,
                 noise_model: gtsam.noiseModel.Base):
        """Initialize cone pattern factor
        
        Args:
            landmark_keys: GTSAM keys for landmarks in the pattern
            pattern: Detected cone pattern
            noise_model: Noise model for the constraint
        """
        super().__init__(noise_model, landmark_keys)
        self.pattern = pattern
        self.landmark_keys = landmark_keys
        
    def error(self, values: gtsam.Values) -> np.ndarray:
        """Calculate error based on pattern type"""
        try:
            # Extract landmark positions
            positions = []
            for key in self.landmark_keys:
                try:
                    point = values.atPoint2(key)
                    # Ensure we get a 2D position
                    if hasattr(point, '__len__'):
                        positions.append(np.array([float(point[0]), float(point[1])]))
                    else:
                        # Point2 might return x(), y() methods
                        positions.append(np.array([float(point.x()), float(point.y())]))
                except Exception as e:
                    # If extraction fails, use origin
                    positions.append(np.array([0.0, 0.0]))
            positions = np.array(positions)
            
            if self.pattern.pattern_type == PatternType.CORNER_90:
                return self._corner_90_error(positions)
            elif self.pattern.pattern_type == PatternType.CURVE:
                return self._curve_error(positions)
            elif self.pattern.pattern_type == PatternType.STRAIGHT:
                return self._straight_error(positions)
            else:
                return np.zeros(self.dim())
        except Exception as e:
            print(f"[PATTERN_FACTOR_ERROR_V5] Exception in error(): {e}")
            # Return zero error of correct dimension on any error
            return np.zeros(self.dim())
    
    def _corner_90_error(self, positions: np.ndarray) -> np.ndarray:
        """Error for 90-degree corner constraint"""
        if len(positions) < 3:
            return np.array([0.0, 0.0])
            
        # V4 FIX: Ensure positions are 2D before calculating vectors
        p0 = np.array(positions[0]).flatten()[:2]
        p1 = np.array(positions[1]).flatten()[:2]
        p2 = np.array(positions[2]).flatten()[:2]
        
        # Calculate vectors
        v1 = p1 - p0
        v2 = p2 - p1
        
        # Ensure vectors are 2D
        v1 = np.array(v1).flatten()[:2]
        v2 = np.array(v2).flatten()[:2]
        
        # Check vector lengths
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return np.array([0.0, 0.0])
        
        # Normalize 
        v1_norm = v1 / norm1
        v2_norm = v2 / norm2
        
        # Error is deviation from 90 degrees (dot product should be 0)
        angle_error = float(np.dot(v1_norm, v2_norm))
        
        # Also enforce similar distances
        distance_error = (norm1 - norm2) / (norm1 + norm2 + 1e-6)  # Normalized difference
        
        return np.array([angle_error * 10, distance_error * 5], dtype=float)  # Scale errors
    
    def _curve_error(self, positions: np.ndarray) -> np.ndarray:
        """Error for curve constraint"""
        # Fit circle to current positions
        center, radius, _ = ConePatternDetector()._fit_circle(positions)
        
        # Expected values from pattern detection
        expected_center = self.pattern.parameters['center']
        expected_radius = self.pattern.parameters['radius']
        
        # Errors
        center_error = center - expected_center
        radius_error = (radius - expected_radius) / expected_radius
        
        return np.concatenate([center_error, [radius_error * 5]])
    
    def _straight_error(self, positions: np.ndarray) -> np.ndarray:
        """Error for straight line constraint"""
        # V5 FIX: Ensure positions are 2D
        positions_2d = positions[:, :2] if positions.shape[1] > 2 else positions
        
        # Expected direction
        expected_dir = self.pattern.parameters['direction']
        
        # Calculate current direction
        mean_pos = np.mean(positions_2d, axis=0)
        centered = positions_2d - mean_pos
        _, _, vt = np.linalg.svd(centered.T, full_matrices=False)
        current_dir = vt[0]
        # V5 FIX: Ensure direction is 2D
        if len(current_dir) > 2:
            current_dir = current_dir[:2]
        
        # Ensure consistent direction
        if np.dot(current_dir, expected_dir) < 0:
            current_dir = -current_dir
        
        # Direction error
        dir_error = current_dir - expected_dir
        
        # Collinearity error (perpendicular distances)
        perp_distances = []
        for pos in centered:
            proj = np.dot(pos, current_dir) * current_dir
            perp_dist = np.linalg.norm(pos - proj)
            perp_distances.append(perp_dist)
            
        return np.concatenate([dir_error * 10, perp_distances])
    
    def dim(self) -> int:
        """Return dimension of error vector"""
        if self.pattern.pattern_type == PatternType.CORNER_90:
            return 2
        elif self.pattern.pattern_type == PatternType.CURVE:
            return 3
        elif self.pattern.pattern_type == PatternType.STRAIGHT:
            return 2 + len(self.landmark_keys)
        else:
            return 1