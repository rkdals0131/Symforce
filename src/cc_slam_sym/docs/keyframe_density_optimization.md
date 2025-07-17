# Keyframe Density Optimization Strategy

## Problem: Excessive Information Condensation
Current 2m keyframe spacing causes:
- 16:1 compression at 2.5 m/s (800ms of data → 1 keyframe)
- 8:1 compression at 5 m/s (400ms of data → 1 keyframe)
- Loss of temporal dynamics and intermediate poses

## Recommended Configuration

### Option 1: Dense Keyframes + Small Window
```yaml
frontend:
  keyframe_distance_threshold: 0.5    # Reduced from 2.0m
  keyframe_rotation_threshold: 0.15   # Reduced from 0.3 rad (~8.5°)
  keyframe_time_threshold: 0.2        # Reduced from 1.0s

backend:
  optimization_interval: 10           # Increased from 5
  max_keyframes: 15                   # Reduced from 20
```

**Benefits:**
- Max 4 observations per keyframe at 5 m/s
- Maintains 7.5m trajectory coverage (15 × 0.5m)
- Better captures motion dynamics
- More odometry constraints for trajectory smoothing

### Option 2: Adaptive Keyframe Spacing
```python
def should_create_keyframe(self, current_time: float) -> bool:
    # Speed-adaptive threshold
    speed = np.linalg.norm(self.current_velocity[:2])
    
    # Adaptive distance threshold: faster = larger spacing
    adaptive_distance = min(0.3 + speed * 0.1, 2.0)  # 0.3m to 2.0m
    
    # Adaptive time threshold: ensure minimum temporal resolution
    adaptive_time = min(0.1 + speed * 0.02, 0.5)  # 0.1s to 0.5s
    
    # Check all conditions
    relative_pose = self.last_keyframe_pose.between(self.current_pose)
    translation = np.sqrt(relative_pose.x()**2 + relative_pose.y()**2)
    rotation = abs(relative_pose.theta())
    time_diff = current_time - self.last_keyframe_time
    
    return (translation > adaptive_distance or 
            rotation > self.config.keyframe_rotation_threshold or
            time_diff > adaptive_time)
```

### Option 3: Observation-Count Based
```yaml
frontend:
  keyframe_distance_threshold: 1.0     # Moderate spacing
  keyframe_observation_threshold: 5    # New: max observations before keyframe
  keyframe_rotation_threshold: 0.2
  keyframe_time_threshold: 0.3
```

```python
# In frontend.py
if (self.observations_since_keyframe >= self.config.keyframe_observation_threshold):
    return True  # Force keyframe to prevent information loss
```

## Performance Impact Analysis

| Config | Keyframes/meter | Graph Size (15m) | Optimization Time |
|--------|----------------|------------------|-------------------|
| Current | 0.5 kf/m | 8 keyframes | ~50ms |
| Dense | 2.0 kf/m | 30 keyframes | ~150ms |
| Adaptive | 0.5-3.3 kf/m | 15-50 keyframes | 50-250ms |
| Obs-based | 1.0 kf/m | 15 keyframes | ~75ms |

## Recommendation: Adaptive Approach

Use **Option 2 (Adaptive)** with these benefits:
1. Preserves detail during slow maneuvers
2. Reduces computation during high-speed straights
3. Maintains consistent observation density
4. Adapts to driving conditions

## Implementation Priority
1. First: Reduce distance threshold to 1.0m
2. Test performance impact
3. Implement adaptive spacing if needed
4. Consider observation-count trigger as fallback