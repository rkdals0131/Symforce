# SLAM Debug Logging Guide

## Overview
The SLAM system uses tagged logging that can be filtered in `rqt_console` for easy debugging. All debug messages are tagged with specific prefixes to make filtering straightforward.

## Using rqt_console

### Launch rqt_console
```bash
ros2 run rqt_console rqt_console
```

### Configure Filters
In the "Highlight Messages..." field, you can use regular expressions to filter specific messages:

1. **Loop Closure Events**: `SLAM_LOOP_CLOSURE`
2. **Optimization Events**: `SLAM_OPTIMIZATION`
3. **Duplicate Landmarks**: `SLAM_DUPLICATE`
4. **Association Issues**: `SLAM_ASSOCIATION`
5. **Errors Only**: `SLAM_.*ERROR`

## Log Tag Reference

### Backend Optimization Tags
- `[SLAM_OPTIMIZATION_START]` - Optimization beginning
- `[SLAM_OPTIMIZATION_COMPLETE]` - Successful optimization with error reduction
- `[SLAM_OPTIMIZATION_ERROR]` - Optimization failures (e.g., Jacobian errors)
- `[SLAM_POSE_CHANGES]` - Summary of pose corrections
- `[SLAM_NO_POSE_CHANGES]` - Warning when optimization doesn't correct poses
- `[SLAM_POSE_CORRECTION]` - Individual pose corrections

### Data Association Tags
- `[SLAM_ASSOCIATION_START]` - Association process beginning
- `[SLAM_ASSOCIATION_SUCCESS]` - Successful observation-landmark match
- `[SLAM_ASSOCIATION_FAILED]` - Failed association attempt
- `[SLAM_ASSOCIATION_CANDIDATE]` - Candidate evaluation details
- `[SLAM_ASSOCIATION_REJECTED]` - Rejection reasons
- `[SLAM_LOOP_CLOSURE]` - Loop closure detection events
- `[SLAM_LOOP_CLOSURE_DETECTED]` - Confirmed loop closure

### Landmark Management Tags
- `[SLAM_LANDMARK_CREATED]` - New landmark creation
- `[SLAM_LANDMARK_INIT]` - Landmark initialization details
- `[SLAM_DUPLICATE_WARNING]` - Potential duplicate landmarks
- `[SLAM_OLD_LANDMARK]` - Old landmarks in loop closure

### System State Tags
- `[SLAM_BACKEND_STATE]` - Backend graph state
- `[SLAM_KEYFRAME_CREATED]` - Keyframe creation
- `[SLAM_ODOMETRY_UPDATE]` - Odometry updates
- `[SLAM_PROCESS_CONES]` - Cone processing statistics

## Common Filter Examples

### Monitor Loop Closures
```
SLAM_LOOP_CLOSURE
```
Shows when the system detects it's revisiting a previous area.

### Track Optimization Performance
```
SLAM_OPTIMIZATION_(COMPLETE|ERROR)
```
Shows optimization results and any errors.

### Find Duplicate Landmarks
```
SLAM_DUPLICATE
```
Highlights when landmarks are created too close to existing ones.

### Debug Association Failures
```
SLAM_ASSOCIATION_(FAILED|REJECTED)
```
Shows why observations aren't matching landmarks.

### Monitor All Errors
```
\[SLAM_.*ERROR\]
```
Catches all SLAM-related errors.

## Log Severity Levels
- **ERROR**: Critical failures (optimization errors, missing data)
- **WARN**: Issues that may affect performance (duplicates, no pose changes)
- **INFO**: Important events (landmark creation, optimization complete)
- **DEBUG**: Detailed information (association candidates, pose updates)

## Example Workflow

1. **Start rqt_console before running SLAM**:
   ```bash
   ros2 run rqt_console rqt_console
   ```

2. **Set initial filter for errors**:
   - In "Highlight Messages..." enter: `SLAM_.*ERROR`
   - This shows only critical issues

3. **Monitor loop closure behavior**:
   - Change filter to: `SLAM_LOOP_CLOSURE`
   - Watch for activation when revisiting areas

4. **Debug duplicate landmarks**:
   - Filter: `SLAM_DUPLICATE|SLAM_LANDMARK_CREATED`
   - Shows new landmarks and duplicate warnings together

5. **Analyze optimization**:
   - Filter: `SLAM_OPTIMIZATION`
   - Shows all optimization-related messages

## Tips
- Use `|` to combine multiple filters (OR operation)
- Use `.*` for wildcards in regex
- Save useful filter configurations for quick access
- Adjust log levels in code if needed (DEBUG vs INFO)
- Export logs from rqt_console for offline analysis