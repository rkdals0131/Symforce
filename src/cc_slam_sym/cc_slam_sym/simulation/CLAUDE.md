# simulation Module

## Module Overview
This module provides simulation capabilities for testing the SLAM system without real sensor data.

## Components

### dummy_publisher_node.py
- Publishes simulated cone detections and odometry
- Generates realistic Formula Student track layouts
- Adds configurable noise to simulate real sensors

### sensor_simulator.py
- Simulates various sensor behaviors with realistic noise models
- Includes occlusion, false positives, and measurement uncertainty
- Configurable through YAML parameters

### cone_definitions.py
- Defines cone types and colors for Formula Student
- Maps between color strings and numeric representations

### motion_controller.py
- Simulates vehicle motion along predefined paths with mathematically smooth dynamics
- Uses continuous spline-based trajectories for smooth motion without jitter
- Implements ContinuousTrajectory class with arc-length parameterization
- Supports both straight track and Formula Student elliptical scenarios

## Status
✅ Fully functional simulation environment
✅ Realistic noise and error models
✅ Configurable through YAML files
✅ Works independently of SLAM backend choice

## Usage
The simulator provides ground truth data alongside noisy measurements, making it ideal for:
- Testing SLAM algorithm performance
- Debugging data association issues  
- Evaluating different backend implementations
