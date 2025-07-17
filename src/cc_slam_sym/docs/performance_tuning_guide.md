# Performance Tuning Guide for CC-SLAM-SYM

## System Overview
- **Input Data Rates**: 100Hz odometry, 20Hz cone observations
- **Current Settings**: Keyframe every 2m/0.3rad/1s, optimization every 5 keyframes
- **Sliding Window**: 20 keyframes (may be too large)

## Performance Issues and Solutions

### 1. Reduce Sliding Window Size
Current: 20 keyframes
Recommended: 10-12 keyframes

```yaml
backend:
  max_keyframes: 12  # Reduced from 20
```

**Rationale**: With keyframes every 2m and 20 keyframes, the system maintains 40m of trajectory. For a Formula Student track, 20-24m (10-12 keyframes) should be sufficient.

### 2. Adjust Keyframe Creation Thresholds
For high-speed racing (>10 m/s), consider:

```yaml
frontend:
  keyframe_distance_threshold: 3.0    # Increased from 2.0m
  keyframe_rotation_threshold: 0.4    # Increased from 0.3 rad
  keyframe_time_threshold: 0.5        # Decreased from 1.0s
```

**Rationale**: Fewer keyframes = faster optimization, but must balance with tracking accuracy.

### 3. Optimize Processing Thread
Current sleep: 1ms
Recommended: Adaptive based on queue size

```python
# In _processing_loop():
if self.cone_queue.size() > 50:
    time.sleep(0.0001)  # 0.1ms when busy
else:
    time.sleep(0.001)   # 1ms when idle
```

### 4. Batch Observation Processing
Instead of processing each cone message individually:

```python
# Process multiple observations at once
cone_msgs = []
while len(cone_msgs) < 5 and not self.cone_queue.empty():
    msg = self.cone_queue.try_pop_front()
    if msg:
        cone_msgs.append(msg)
```

### 5. Profile Critical Sections
Add timing to identify bottlenecks:

```python
import cProfile
import pstats

# In main():
profiler = cProfile.Profile()
profiler.enable()
# ... run SLAM ...
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

## Expected Performance After Tuning
- Keyframe creation: <5ms
- Data association: <10ms  
- Optimization (12 keyframes): <50ms
- Total processing: <100ms per optimization cycle

## Testing Recommendations
1. Start with reduced sliding window (12 keyframes)
2. Monitor optimization timing with debug logs
3. Gradually increase keyframe thresholds if needed
4. Use rqt_console to filter for timing messages:
   - `[SLAM_OPTIMIZATION_COMPLETE]`
   - `[SLAM_KEYFRAME_CREATED]`
   - `[SLAM_ASSOCIATION]`