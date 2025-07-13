#!/usr/bin/env python3
"""
Simulation Module

Contains simulation components for testing SLAM algorithms:
- Dummy publisher: Simulates sensor data with realistic noise models
- Cone definitions: Ground truth track layouts
- Sensor error models: Detection errors, drift, bias simulation
"""

from .dummy_publisher_node import DummyPublisher
from .cone_definitions import *