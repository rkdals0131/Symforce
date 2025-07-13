#!/usr/bin/env python3
"""
SLAM Core Module

Contains the main SLAM algorithms:
- Backend: Factor graph optimization using GTSAM and SymForce  
- Frontend: Data association, feature extraction
- Loop Closure: Place recognition and loop detection
"""

from .cone_color_factor import ConeColorFactor