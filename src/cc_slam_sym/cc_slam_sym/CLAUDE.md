# cc_slam_sym Package Status

## Package Overview
CC-SLAM-SYM is a Formula Student Driverless SLAM system using SymForce for optimized factor computation with GTSAM backend. Successfully integrates SymForce-generated factors with batch optimization.

## Current Implementation Status

### Working Components
- **ros_bridge**: ROS2 interface with data conversion
- **simulation**: Dummy cone publisher with realistic noise simulation  
- **frontend**: Keyframe and landmark management
- **backend**: GTSAM batch optimization with SymForce factors ✅
- **utils**: Data structures and concurrent containers
- **slam_core/generated**: SymForce-generated optimized functions ✅

### Recent Fixes (2025-07-16)
1. **SymForce Integration**: Now fully operational with generated factors
2. **Batch Optimization**: Replaced ISAM2 due to CustomFactor compatibility 
3. **Stable Performance**: No numerical errors or accumulation issues
4. **Color-aware Factors**: Properly utilizing color information in optimization

## System Architecture

### Factor Types
1. **Motion Model Factor**: Ackermann-constrained odometry with SymForce
2. **Cone Observation Factor**: Color-aware landmark observations
3. **Prior Factor**: Initial pose constraint

### Key Design Decisions
1. **Batch Optimization**: Using LevenbergMarquardt instead of ISAM2 for stability
2. **Stable Factors**: Simplified residual functions to avoid numerical issues
3. **Color Weighting**: Dynamic weight adjustment based on color confidence

## Module Structure
- **slam_core/**: Contains both used and unused backend implementations
- **ros_bridge/**: Fully functional ROS2 interface
- **simulation/**: Working simulator for testing
- **utils/**: Core data structures and utilities

## Performance Target
- Must handle 10Hz backend processing rate
- Requires optimized Jacobian computation via SymForce
- Need proper sliding window management