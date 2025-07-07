# Bundle Adjustment Fix Summary

## Fixed Issues in chapter10_visual_slam_with_cameras.ipynb

### 1. Camera Coordinate Transformation

**Problem**: The original `create_bundle_adjustment_problem` function had incorrect camera orientation setup. The cameras were not properly looking at the origin, causing all landmarks to appear behind the cameras.

**Solution**: Fixed the rotation matrix construction to ensure cameras properly look at the origin:

```python
# Camera forward direction (-Z axis) should point at target
forward = target - cam_pos  # Camera to target direction
forward = forward / np.linalg.norm(forward)

# Right direction (X axis): cross product of forward and world up
world_up = np.array([0, 0, 1])
right = np.cross(forward, world_up)
right = right / np.linalg.norm(right)

# Up direction (Y axis): cross product of right and forward
up = np.cross(right, forward)

# Rotation matrix [X, Y, Z] = [right, up, -forward]
# Camera looks in -Z direction
R_matrix = np.column_stack([right, up, -forward])
```

### 2. Pose Convention Clarification

**Added clear documentation** to `visual_feature_residual` function:
- `camera_pose` represents world-to-camera transformation
- `camera_pose.t` is the camera position in world coordinates
- To transform world points to camera coordinates: `landmark_cam = pose.inverse() * landmark`

### 3. Key Points

1. **Camera Convention**: Cameras look in the -Z direction in their local coordinate frame
2. **Pose Definition**: The pose stores the world-to-camera transformation
3. **Coordinate Transform**: Use `pose.inverse()` to transform from world to camera coordinates

### Testing

Added a test cell to verify the fix:
- Camera at (3, 0, 1.5) looking at origin
- Landmark at origin projects near image center
- Z coordinate in camera frame is positive (in front of camera)

## Results

With these fixes:
- All landmarks are now visible to the cameras
- Bundle adjustment optimization can proceed with actual measurements
- The camera setup correctly implements a pinhole camera model looking at the scene