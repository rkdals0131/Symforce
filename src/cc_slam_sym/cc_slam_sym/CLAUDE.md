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

### Recent Fixes (2025-07-17)
1. **SymForce Integration**: Now fully operational with generated factors
2. **Batch Optimization**: Replaced ISAM2 due to CustomFactor compatibility 
3. **Noise Model Correction**: Fixed simulation/SLAM config mismatch for perfect input
4. **Performance Optimization**: Disabled robust kernels and color penalty for GT simulation
5. **Graph Size Control**: Reduced sliding window to 20 keyframes for better performance
6. **Landmark Drift Investigation**: Identified systematic issues in data association and factor graph construction

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

## Current Status: Loop Closure and Duplicate Landmark Fixes

### Recent Fixes (2025-07-17 continued)
1. **Fixed Critical Optimization Bug**: Backend was skipping optimization due to incorrect factor counting
2. **Enhanced Association Debugging**: Detailed logging of Mahalanobis distances and association decisions
3. **Adaptive Loop Closure Thresholds**: Automatically relaxes association threshold when revisiting areas
4. **Landmark Creation Tracking**: Tracks when/where landmarks were created for loop closure detection
5. **Fixed Association Result Usage**: Keyframes now properly use association results for observation factors
6. **Fixed ISAM2 Reference Bug**: Backend _update_landmark_estimates() was using non-existent ISAM2 object
7. **Optimized Keyframe Density**: Reduced spacing from 2m to 1m to prevent excessive information condensation

### Loop Closure Implementation
- **Detection**: Based on seeing old landmarks (>10s) or traveling >50m
- **Threshold Adaptation**: Chi-squared threshold scales from 4.605 to 9.21 during loop closure
- **Tracking**: Maintains landmark creation times and poses for revisit detection
- **Integration**: Resets tracking after each optimization

### Expected Behavior
- System prints "LOOP CLOSURE MODE ACTIVATED" when revisiting areas
- Successfully associates with existing landmarks instead of creating duplicates
- Optimization runs every 5 keyframes and corrects accumulated drift
- No duplicate landmarks within 2m radius

### Testing Strategy
1. Run figure-8 or loop trajectory
2. Monitor association logs for loop closure activation
3. Verify no duplicate landmarks are created
4. Check that poses are corrected by optimization

### Files Created
- `loop_closure_debug_strategy.md`: Comprehensive debugging and solution guide
- Enhanced logging in `data_association.py` and `backend.py`
- Debug monitoring nodes in `cc_slam_sym/debug/`:
  - `association_monitor_node.py`: Tracks data association and loop closure
  - `optimization_monitor_node.py`: Monitors optimization behavior
  - `landmark_monitor_node.py`: Detects duplicate landmarks and clusters
- `debug_monitor_launch.py`: Launch file for all debug nodes

### Debug Logging System
- **Purpose**: Tagged logging for easy filtering in rqt_console
- **Tags**: Prefixed messages like `[SLAM_OPTIMIZATION_ERROR]`, `[SLAM_LOOP_CLOSURE]`
- **Tool**: Use `rqt_console` with regex filters
- **Documentation**: See `docs/slam_debug_logging_guide.md`

### Usage
```bash
# Start rqt_console
ros2 run rqt_console rqt_console

# Run SLAM
ros2 launch cc_slam_sym slam_launch.py

# In rqt_console, use filters like:
# - SLAM_OPTIMIZATION_ERROR  (optimization failures)
# - SLAM_LOOP_CLOSURE       (loop closure events)
# - SLAM_DUPLICATE          (duplicate landmarks)
```

## Performance Target
- Must handle 10Hz backend processing rate
- Requires optimized Jacobian computation via SymForce
- Need proper sliding window management
- **Current Priority**: Verify rigid constraints are enforced

## Critical Fixes Applied (2025-07-17)

### Rigid Constraint Enforcement
- Observation factors now correctly transform landmarks to robot frame
- Very tight noise model (0.1x) enforces rigid relative positions
- Simplified motion model for reliability
- Enhanced debugging to track constraint violations

### Expected Behavior
- Landmarks should maintain exact relative position to observing poses
- Optimization should trigger every 10 keyframes
- No "[LARGE_ERROR]" messages should appear
- System should converge to consistent map

## System Architecture Notes
- This is a **cone-based visual SLAM**, NOT LiDAR SLAM
- Tracks colored cones (yellow/blue/red) for Formula Student
- Input: 100Hz odometry + 20Hz cone observations + 8Hz GPS (RTK)
- Keyframe creation: Every 1m/0.2rad/0.3s (optimized for less information loss)
- Optimization: Every 10 keyframes with 15-keyframe sliding window
- Reduced information condensation from 16:1 to 8:1 at low speeds

## Corner Divergence Fix (2025-07-18)

### Adaptive Motion-Aware Data Association
- Chi-squared thresholds now scale with angular velocity
- Association search radius increases during turns
- Outlier rejection adapts to motion state
- Prevents false rejections during high uncertainty periods

### GPS Factor Integration
- RTK GPS provides absolute position constraints
- 2cm accuracy prevents long-term drift
- GPS factors added to keyframes at 8Hz
- First GPS measurement sets reference frame

### Implementation Details
- `data_association.py`: Added angular_velocity parameter for adaptive thresholds
- `backend.py`: Added add_gps_factor() method and GPS reference tracking
- `frontend.py`: Tracks angular velocity from odometry
- GPS simulation publishes on `/gps/odom` topic
- SLAM node subscribes and adds GPS constraints to factor graph

### Expected Improvements
- No divergence at track corners
- Maintained global consistency via GPS
- Better association during rapid viewpoint changes
- Stable optimization convergence

## V3 Debug Enhancement (2025-07-18)

### Pattern Factor Debug
- Added V3 logging throughout pattern detection pipeline
- Pattern detector logs detection results with type and confidence
- Backend logs when patterns are added to factor graph
- Visualization logs pattern count before drawing

### Debug Approach
- Use tagged logging (e.g., [SLAM_PATTERN_V3]) for easy filtering
- Monitor pattern detection: corners (60-120°), curves, straight lines
- Track pattern signatures to avoid duplicates
- Verify pattern factors are visualized as purple lines

### Key Files Modified
- backend.py: Enhanced pattern detection logging
- slam_ros_node.py: V3 initialization message and pattern viz logging
- Build with --symlink-install for immediate updates

## V4 Optimization Fix (2025-07-18)

### Issues Fixed
1. **Pattern Factor Dimension Error**: Fixed "shapes (2,) and (3,) not aligned"
   - Added defensive coding to ensure all vectors are 2D
   - Multiple flatten()[:2] operations to guarantee 2D vectors
   
2. **Optimization Failure Debugging**: 
   - Enhanced error logging with full traceback
   - Disabled async optimization for clearer debugging
   - Added [SLAM_OPTIMIZATION_FAILED_V4] tag

3. **Build Issues**:
   - V3 messages weren't showing → confirmed need for rebuild
   - Using --symlink-install for immediate updates

### Debug Messages
- [SLAM_PATTERN_DETECTION_V4]: Pattern detection attempts
- [SLAM_OPTIMIZATION_FAILED_V4]: Detailed optimization errors
- VERSION 4 WITH OPTIMIZATION FIX: Initialization confirmation
