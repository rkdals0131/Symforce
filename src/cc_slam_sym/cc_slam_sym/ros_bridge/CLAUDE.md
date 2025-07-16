# ros_bridge Module

## Module Overview
This module provides the ROS2 interface layer for the SLAM system, handling communication between ROS topics and the SLAM core.

## Components

### slam_ros_node.py
- Main ROS2 node that orchestrates the SLAM system
- Subscribes to odometry and cone detection topics
- Publishes poses, paths, landmarks, and visualization markers
- Currently using standard backend.py (NOT SymForce-enhanced)

### data_converter.py  
- Converts between ROS messages and internal data structures
- Handles cone detections, odometry, and IMU data conversion
- Fully functional and actively used

## Current Issues
- Node is hardcoded to use backend.py instead of symforce_backend.py
- No configuration option to switch between backends
- Performance monitoring shows degradation after processing many frames

## Integration Points
- Receives: `/odometry`, `/cone_detections` 
- Publishes: `/slam/pose`, `/slam/path`, `/slam/landmarks`, `/slam/markers`
- TF: Publishes map->odom transform

## Status
✅ Fully functional ROS2 interface
❌ Not using SymForce-optimized backend
⚠️ Performance issues due to backend limitations
