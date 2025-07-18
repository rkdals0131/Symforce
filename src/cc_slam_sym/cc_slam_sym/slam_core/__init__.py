#!/usr/bin/env python3
"""
SLAM Core Module

Contains the main SLAM algorithms:
- Backend: Factor graph optimization using GTSAM with SymForce-generated factors
- Frontend: Data association, feature extraction
- SymForce Integration: Optimized factor computation with analytical Jacobians
"""

# Import data association module
try:
    from .data_association import DataAssociation, AssociationConfig, AssociationResult
except ImportError:
    pass

# Import frontend module
try:
    from .frontend import SlamFrontend, FrontendConfig
except ImportError:
    pass

# Import backend module
try:
    from .backend import SlamBackend, BackendConfig
except ImportError:
    pass

# Import local map
try:
    from .local_map import LocalMap, LocalMapConfig
except ImportError:
    pass

# Import SymForce-GTSAM integration
try:
    from .symforce_gtsam_factors_stable import (
        create_symforce_cone_factor,
        create_symforce_motion_factor,
        color_string_to_float
    )
except ImportError:
    pass

# Import cone color factor for code generation
try:
    from .cone_color_factor import ConeColorFactor
except (ImportError, AttributeError):
    # SymForce may not be available or configured
    pass

# Import cone pattern factor for structural constraints
try:
    from .cone_pattern_factor import ConePatternDetector, ConePattern, PatternType, ConePatternFactor
except ImportError:
    pass