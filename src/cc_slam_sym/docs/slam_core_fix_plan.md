# SLAM Core Fix Plan

## Root Causes Identified

### 1. Association Result Not Used
- `process_cone_observations()` returns an `AssociationResult` with matched pairs
- But the code ignores this and tries to match by track_id later
- This means NO observation factors are added to the graph!

### 2. Keyframe Should Store Matched Landmarks
- Currently stores raw observations
- Should store the association result or matched landmark IDs
- This would allow proper factor creation

### 3. Optimization Never Runs Successfully
- Without observation factors, optimization has nothing to correct
- The graph only has odometry factors following the noisy path
- This is why poses are never corrected

## Fix Strategy

### Option 1: Store Association Result in Keyframe
```python
# In frontend.py - modify create_keyframe
keyframe = Keyframe(
    id=self.next_keyframe_id,
    timestamp=timestamp,
    pose_symbol=gtsam.symbol('x', self.next_keyframe_id),
    pose=self.current_pose,
    observations=observations,
    association_result=association_result  # NEW
)
```

### Option 2: Process Associations When Creating Keyframe
```python
# In slam_ros_node.py - use association result immediately
association_result = self.frontend.process_cone_observations(observations, timestamp)

if self.frontend.should_create_keyframe(timestamp):
    keyframe = self.frontend.create_keyframe(timestamp, observations)
    
    # Add observation factors based on association result
    for obs_idx, lm_idx in association_result.matched_pairs:
        landmark = self.frontend.landmarks[lm_idx]
        observation = observations[obs_idx]
        self.backend.add_landmark_observation(keyframe, landmark, observation)
```

### Option 3: Change Keyframe to Store Landmark References
```python
# Store landmark IDs and observations in keyframe
keyframe.landmark_observations = [
    (landmark_id, observation) 
    for obs_idx, lm_idx in association_result.matched_pairs
]
```

## Why SLAM Is Failing

1. **No Observation Factors**: The graph has no landmark observation constraints
2. **Only Odometry Factors**: The graph only follows noisy odometry
3. **No Correction Possible**: Without observations, there's nothing to correct the path
4. **Landmarks Without Edges**: Landmarks exist but aren't connected to keyframes

## Expected Behavior After Fix

1. Each keyframe should have edges to observed landmarks
2. When the same landmark is seen from multiple poses, optimization should:
   - Adjust keyframe poses to minimize observation errors
   - Keep relative geometry consistent
3. The optimized path should be smoother than noisy odometry
4. Loop closure should snap the trajectory when revisiting areas