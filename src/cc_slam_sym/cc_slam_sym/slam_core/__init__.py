#!/usr/bin/env python3
"""
SLAM Core Module

Contains the main SLAM algorithms:
- Backend: Factor graph optimization using GTSAM and SymForce  
- Frontend: Data association, feature extraction
- Loop Closure: Place recognition and loop detection
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

# Import SLAM system
try:
    from .slam_system import SlamSystem, SlamSystemConfig
except ImportError:
    pass

# Import visualizer
try:
    from .slam_visualizer import SlamVisualizer, SlamAnimator
except ImportError:
    pass

# Import cone color factor if available
try:
    from .cone_color_factor import ConeColorFactor
except (ImportError, AttributeError):
    pass