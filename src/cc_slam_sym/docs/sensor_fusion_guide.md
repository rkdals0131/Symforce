# Sensor Fusion Guide for CC-SLAM-SYM

## Overview

This guide explains the sensor simulation implementation and its compatibility with:
1. **First Plan**: robot_localization EKF (Extended Kalman Filter) for state estimation
2. **Second Plan**: IMU preintegration with GPS factors for tight coupling in factor graphs

The sensor fusion system combines data from three simulated sensors:
- **IMU** (`/ouster/imu_sim`): Provides orientation, angular velocity, and linear acceleration at 100Hz
- **GPS** (`/ublox_gps_node/fix_sim`): Provides global position at 8Hz with RTK support
- **Odometry** (`/odom_sim`): Provides position and velocity with cumulative drift at 100Hz

## Sensor Models

### IMU Simulation (Allan Variance Based)

The IMU simulator implements a realistic noise model based on Allan variance parameters, commonly used in MEMS IMU specifications.

#### Noise Parameters
- **Noise Density (N)**: White noise spectral density in units/√Hz
- **Bias Stability (B)**: Flicker noise representing the minimum achievable bias
- **Random Walk (K)**: Rate random walk in units/s/√Hz

#### Implementation
```python
# White noise: sigma = N * sqrt(sampling_rate)
sampling_rate = 1.0 / dt  # Hz
accel_white_noise_sigma = noise_density * sqrt(sampling_rate)

# Bias random walk: sigma = K * sqrt(dt)
bias_rw_sigma = random_walk * sqrt(dt)
bias += normal(0, bias_rw_sigma)

# Bias limited by stability
bias = clip(bias, -bias_stability, bias_stability)
```

#### Default Values (MPU9250-like MEMS IMU)
- Gyro noise density: 0.005 rad/s/√Hz
- Gyro bias stability: 0.1 rad/s
- Gyro random walk: 0.00001 rad/s²/√Hz
- Accel noise density: 0.01 m/s²/√Hz
- Accel bias stability: 0.01 m/s²
- Accel random walk: 0.0001 m/s³/√Hz

### RTK GPS Simulation

The GPS simulator supports multiple fix modes with realistic noise levels:

#### GPS Modes
1. **RTK Fix** (`gps_mode: "rtk"`): Highest accuracy
   - Horizontal: 2cm (0.02m)
   - Vertical: 4cm (0.04m)
   - Status: 2 (DGPS/RTK fix)

2. **RTK Float** (`gps_mode: "rtk_float"`): Medium accuracy
   - Horizontal: 30cm (0.3m)
   - Vertical: 50cm (0.5m)
   - Status: 2 (DGPS/RTK fix)

3. **DGPS** (`gps_mode: "dgps"`): Differential GPS
   - Horizontal: ~1m
   - Vertical: ~2.5m
   - Status: 2 (DGPS fix)

4. **Single** (`gps_mode: "single"`): Standard GPS
   - Horizontal: 2m
   - Vertical: 5m
   - Status: 1 (GPS fix)

### Odometry Simulation

The odometry simulator implements distance-based drift suitable for both wheel odometry and IMU-based odometry:

#### Drift Parameters (Per Axis)
- **X-axis (forward)**:
  - Systematic bias: % of distance traveled
  - Random noise: m per update
- **Y-axis (lateral)**:
  - Systematic bias: % of forward distance (simulates misalignment)
  - Random noise: m per update
- **Theta (heading)**:
  - Systematic bias: % of rotation
  - Random noise: rad/m (proportional to distance)

## Usage

### 1. Build the Package
```bash
cd /home/user1/ROS2_Workspace/Symforce_ws
colcon build --packages-select cc_slam_sym
source install/setup.bash
```

### 2. Launch Sensor Fusion
```bash
ros2 launch cc_slam_sym sensor_fusion_launch.py
```

Optional parameters:
- `scenario:=1` or `scenario:=2` (track selection)
- `vehicle_speed:=5.0` (speed in m/s)
- `use_sim_time:=false` (use simulation time)

### 3. Monitor the Fusion Output
```bash
# View fused odometry
ros2 topic echo /odom_sim_fusion

# View individual sensors
ros2 topic echo /ouster/imu_sim
ros2 topic echo /ublox_gps_node/fix_sim
ros2 topic echo /odom_sim
```

## robot_localization Configuration

### EKF Configuration Example

The EKF configuration is in `config/ekf_sensor_fusion.yaml`:

```yaml
ekf_filter_node:
  ros__parameters:
    frequency: 30.0
    sensor_timeout: 0.1
    two_d_mode: true  # For ground vehicles
    
    # Frames
    map_frame: map
    odom_frame: odom
    base_link_frame: base_link
    world_frame: odom
    
    # IMU configuration
    imu0: /ouster/imu_sim
    imu0_config: [false, false, false,  # position
                  true,  true,  true,   # orientation
                  false, false, false,  # velocity
                  true,  true,  true,   # angular velocity
                  true,  true,  true]   # acceleration
    imu0_differential: false
    imu0_relative: false
    imu0_remove_gravitational_acceleration: true
    
    # GPS configuration (through navsat_transform_node)
    odom0: /odometry/gps
    odom0_config: [true,  true,  false,  # position (x,y only)
                   false, false, false,  # orientation
                   false, false, false,  # velocity
                   false, false, false,  # angular velocity
                   false, false, false]  # acceleration
    odom0_differential: false
    
    # Odometry configuration
    odom1: /odom_sim
    odom1_config: [true,  true,  false,  # position
                   false, false, true,   # orientation (yaw only)
                   true,  true,  false,  # velocity
                   false, false, true,   # angular velocity (yaw rate)
                   false, false, false]  # acceleration
    odom1_differential: false
```

### Topics

#### Input Topics
- `/ouster/imu_sim` (sensor_msgs/Imu)
- `/ublox_gps_node/fix_sim` (sensor_msgs/NavSatFix)
- `/ublox_gps_node/fix_velocity_sim` (geometry_msgs/TwistWithCovarianceStamped)
- `/odom_sim` (nav_msgs/Odometry)

#### Output Topics
- `/odom_sim_fusion` (nav_msgs/Odometry) - Fused odometry output
- `/odometry/gps` (nav_msgs/Odometry) - GPS converted to local coordinates
- `/gps/filtered` (sensor_msgs/NavSatFix) - Filtered GPS output

## IMU Preintegration Compatibility

For future tight coupling with IMU preintegration:

### Required Data
1. **High-rate IMU data** (100Hz+) with accurate timestamps
2. **Bias estimates** from filter or optimization
3. **Noise parameters** for preintegration covariance

### Preintegration Factors
The simulator provides all necessary data for computing:
- Preintegrated measurements (position, velocity, rotation)
- Preintegration Jacobians
- Noise propagation

### GPS Factor Integration
RTK GPS measurements can be used as:
- **Position factors**: Direct position constraints with mode-appropriate covariances
- **Velocity factors**: From GPS Doppler measurements

## Testing and Validation

### 1. Sensor Data Validation
```bash
# Check IMU noise characteristics
ros2 topic echo /ouster/imu_sim | grep -A3 angular_velocity

# Check GPS mode and accuracy
ros2 topic echo /ublox_gps_node/fix_sim | grep -A2 status

# Monitor odometry drift
ros2 topic echo /odom_sim | grep -A2 position
```

### 2. Drift Testing
```python
# Test script to monitor drift accumulation
import rclpy
from nav_msgs.msg import Odometry

def odom_callback(msg):
    # Compare with ground truth
    print(f"Odom: x={msg.pose.pose.position.x:.3f}, "
          f"y={msg.pose.pose.position.y:.3f}, "
          f"theta={msg.pose.pose.orientation.z:.3f}")
```

### 3. Covariance Validation
- IMU covariances scale with sampling rate
- GPS covariances match selected mode
- Odometry covariances grow with distance

## Tuning Tips

1. **Process Noise**: 
   - Increase for aggressive maneuvers
   - Decrease for smooth motion

2. **Sensor Weights**:
   - Trust RTK GPS more (lower covariance)
   - Trust odometry less over time (growing covariance)

3. **Bias Estimation**:
   - Enable IMU bias estimation for long-term accuracy
   - Monitor bias convergence in filter diagnostics

## Troubleshooting

1. **Poor GPS/IMU Fusion**:
   - Verify time synchronization
   - Check frame transformations
   - Review covariance settings

2. **Excessive Drift**:
   - Reduce odometry drift parameters
   - Increase GPS fusion weight
   - Enable differential mode for odometry

3. **Jumpy Estimates**:
   - Increase process noise
   - Check for sensor dropouts
   - Verify covariance matrices are positive definite

## Future Enhancements

1. **Time Synchronization**: Add realistic time delays and clock drift
2. **Sensor Failures**: Simulate GPS outages, IMU saturation
3. **Environmental Effects**: Temperature-dependent bias, multipath GPS
4. **Wheel Slip Detection**: For wheel odometry validation

## References

- [robot_localization Documentation](http://docs.ros.org/en/melodic/api/robot_localization/html/)
- [Allan Variance for MEMS IMU](https://www.analog.com/en/technical-articles/allan-variance-noise-analysis-for-gyroscopes.html)
- [RTK GPS Accuracy Standards](https://www.novatel.com/tech-talk/an-introduction-to-gnss/chapter-4-gnss-error-sources/)