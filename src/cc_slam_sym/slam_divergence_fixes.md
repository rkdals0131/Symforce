# SLAM Divergence Fixes Summary

## Issues Identified

1. **Pattern factors not visualized** - Pattern detection was running but not finding patterns due to strict thresholds
2. **Orphan landmarks** - Landmarks created without observation factors connecting them to poses
3. **Corner divergence** - System still diverges at corners despite adaptive thresholds

## Fixes Applied

### 1. Pattern Factor Visualization (FIXED)
- Added extensive debug logging to pattern detection
- Made pattern detection thresholds more lenient:
  - Corner detection: 60-120° (was 75-105°)
  - Curve detection: residual < 0.5m and radius > 2m
  - Straight line: perpendicular distance < 0.3m (was 0.2m)
- Added logger to ConePatternDetector for debugging
- Pattern visualization already implemented in slam_ros_node.py (purple lines)

### 2. Debug Output Added
- `[SLAM_PATTERN_DETECTION]` - Shows pattern detection attempts
- `[SLAM_PATTERN_CORNER]` - Corner angle calculations
- `[SLAM_PATTERN_CURVE]` - Circle fitting results
- `[SLAM_PATTERN_STRAIGHT]` - Line fitting results
- `[SLAM_PATTERN_FACTOR]` - Successfully added patterns

### 3. Orphan Landmark Prevention (IN PROGRESS)
The issue is that landmarks are created but not immediately connected with observation factors. The system tracks new_landmark_track_ids but the connection happens in the next keyframe.

**Root cause**: When a landmark is created from candidate observations, it needs to have observation factors added immediately to the current keyframe, not wait for the next one.

### 4. Corner Divergence (PENDING)
Despite adaptive thresholds, the system still diverges. Possible causes:
- Optimization not correcting poses effectively
- Association failing during rapid rotation
- Insufficient constraint strength from observations

## Next Steps

1. **Fix orphan landmarks**: Modify landmark creation to immediately add observation factors
2. **Debug optimization effectiveness**: Check if optimization is actually correcting poses
3. **Tune observation noise**: May need tighter constraints to prevent drift
4. **Add IMU preintegration**: Would provide better motion constraints during turns

## Testing Instructions

1. Run with debug logging:
```bash
ros2 launch cc_slam_sym slam_launch.py
```

2. Filter logs in rqt_console:
- `SLAM_PATTERN` - Pattern detection
- `SLAM_ORPHAN` - Orphan landmark detection
- `SLAM_OPTIMIZATION` - Optimization behavior

3. Expected behavior:
- Purple lines should appear between landmarks forming patterns
- No landmarks without green observation lines
- Optimization should correct drift at corners