# Adaptive Keyframe Implementation Plan

## Summary of Analysis
- Current fixed 2m spacing causes 16:1 information compression at low speeds
- Changed to 1m spacing, but still not optimal for all speeds
- Need adaptive approach based on vehicle dynamics

## Implementation Strategy (Phase 1: Velocity-Based)

### 1. Add Velocity Tracking to Frontend
```python
# In frontend.py __init__:
self.current_velocity = np.zeros(3)  # [vx, vy, vtheta]
self.velocity_history = []  # For smoothing

# In process_odometry():
if hasattr(odom_data, 'linear_velocity'):
    self.current_velocity[:2] = odom_data.linear_velocity
    self.current_velocity[2] = odom_data.angular_velocity
```

### 2. Implement Adaptive Keyframe Creation
```python
def should_create_keyframe(self, current_time: float) -> bool:
    """Adaptive keyframe creation based on velocity and information density"""
    
    # Get current speed
    speed = np.linalg.norm(self.current_velocity[:2])
    
    # Adaptive distance threshold
    # At 2.5 m/s: 0.5m spacing (4 obs/kf)
    # At 10 m/s: 1.0m spacing (2 obs/kf)  
    # At 20 m/s: 1.5m spacing (1.5 obs/kf)
    adaptive_distance = 0.3 + speed * 0.06  # Clamp to [0.3, 2.0]
    adaptive_distance = np.clip(adaptive_distance, 0.3, 2.0)
    
    # Adaptive time threshold
    # Ensure we don't wait too long at any speed
    adaptive_time = 0.15 + speed * 0.015  # Clamp to [0.15, 0.5]
    adaptive_time = np.clip(adaptive_time, 0.15, 0.5)
    
    # Calculate motion since last keyframe
    relative_pose = self.last_keyframe_pose.between(self.current_pose)
    translation = np.sqrt(relative_pose.x()**2 + relative_pose.y()**2)
    rotation = abs(relative_pose.theta())
    time_diff = current_time - self.last_keyframe_time
    
    # Observation count check (prevent excessive condensation)
    obs_count = self.observations_since_keyframe  # Need to track this
    max_obs_per_kf = 8  # Maximum observations before forcing keyframe
    
    # Create keyframe if any condition met
    create_kf = (translation > adaptive_distance or 
                 rotation > self.config.keyframe_rotation_threshold or
                 time_diff > adaptive_time or
                 obs_count >= max_obs_per_kf)
    
    if create_kf and self.logger:
        self.logger.debug(f"[SLAM_KF_TRIGGER] speed={speed:.1f}m/s, "
                         f"dist={translation:.2f}m (thresh={adaptive_distance:.2f}), "
                         f"rot={rotation:.2f}rad, time={time_diff:.2f}s, "
                         f"obs={obs_count}")
    
    return create_kf
```

### 3. Track Observation Count
```python
# In frontend.py:
def process_cone_observations(self, observations, timestamp):
    # Existing association code...
    
    # Track observations for keyframe density control
    if hasattr(self, 'observations_since_keyframe'):
        self.observations_since_keyframe += len(observations)
    else:
        self.observations_since_keyframe = len(observations)
    
    # ... rest of method

def create_keyframe(self, timestamp, observations, association_result=None):
    # ... existing code ...
    
    # Reset observation counter
    self.observations_since_keyframe = 0
    
    return keyframe
```

## Configuration Updates
```yaml
frontend:
  # Adaptive keyframing parameters
  keyframe_min_distance: 0.3         # Minimum spacing (low speed)
  keyframe_max_distance: 2.0         # Maximum spacing (high speed) 
  keyframe_speed_factor: 0.06        # Distance = min + speed * factor
  keyframe_max_observations: 8       # Force keyframe after N observations
  keyframe_rotation_threshold: 0.2   # Keep rotation threshold fixed
```

## Expected Results
| Speed | Old Spacing | New Adaptive | Obs/Keyframe | Info Ratio |
|-------|------------|--------------|--------------|------------|
| 2.5 m/s | 2.0m | 0.45m | ~4 | 4:1 |
| 5 m/s | 2.0m | 0.60m | ~2-3 | 2.5:1 |
| 10 m/s | 2.0m | 0.90m | ~2 | 2:1 |
| 20 m/s | 2.0m | 1.50m | ~1-2 | 1.5:1 |

## Phase 2 Considerations (Future)
1. Information-gain based keyframing (as Gemini suggested)
2. Keyframe culling to remove redundant frames
3. Cone clustering to reduce observation count
4. Covariance-based adaptive thresholds