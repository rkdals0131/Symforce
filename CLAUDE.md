# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a ROS2 workspace implementing a high-performance SLAM (Simultaneous Localization and Mapping) system with multi-threaded graph optimization. The main package `cc_slam_graph` integrates g2o for backend optimization and supports both RViz and Matplotlib visualization.

## Essential Commands

### Build Commands
```bash
# Build the ROS2 package from workspace root
colcon build --packages-select cc_slam_graph

# For development with live code changes
colcon build --packages-select cc_slam_graph --symlink-install

# Build external GTSAM library (if needed)
cd lib/gtsam && mkdir build && cd build && cmake .. && make
```

### Run Commands
```bash
# Source the workspace first
source install/setup.bash

# Run simulation data publisher
ros2 run cc_slam_graph dummy_cone_publisher_node

# Run main SLAM node
ros2 run cc_slam_graph high_performance_slam_node
```

### Test Commands
```bash
# Run all tests
colcon test --packages-select cc_slam_graph
colcon test-result --verbose

# Run specific linting tests
ament_flake8 cc_slam_graph
ament_pep257 cc_slam_graph
```

## High-Level Architecture

### Core Components

1. **high_performance_slam_node.py**: Main ROS2 node orchestrating the SLAM system
   - Manages multi-threaded processing (up to 16 cores)
   - Handles ROS2 communication and TF publishing
   - Integrates visualization pipelines

2. **slam_backend_engine.py**: Core SLAM backend using g2o
   - Implements pose-graph optimization
   - Manages keyframes and landmarks
   - Handles data association and optimization triggers

3. **slam_visualizer.py**: Dual visualization system
   - RViz markers for 3D visualization
   - Matplotlib for 2D trajectory plots

### External Libraries

- **GTSAM** (lib/gtsam/): Modified fork with Sim(3) Lie group support for factor graphs
- **nano-pgo** (lib/nano-pgo/): Educational pose-graph optimization using SymForce for auto-generated Jacobians

### Configuration System

All SLAM parameters are centralized in `config/slam_config.yaml`:
- Backend settings (keyframe thresholds, optimization parameters)
- Multi-threading configuration
- Visualization options
- Performance monitoring settings
- Sensor fusion placeholders (GPS/IMU ready)

### Development Roadmap

The project has a comprehensive development plan (20250608.md) targeting:
- Global loop closure detection
- Full graph re-optimization
- Chi-squared based outlier removal
- ROS service interfaces for external control
- IMU/GPS tight coupling integration

### Key Design Patterns

1. **Multi-threaded Processing**: Parallel processing of keyframes and landmarks with thread-safe queues
2. **Modular Architecture**: Clear separation between data ingestion, backend processing, and visualization
3. **Configurable Everything**: All parameters externalized to YAML configuration
4. **Standard ROS2 Integration**: Full TF tree publishing, standard message types

### Performance Considerations

- Target: 10Hz backend processing rate
- Batch optimization for efficiency
- Resource monitoring and adaptive configuration
- Huber kernel for robust optimization