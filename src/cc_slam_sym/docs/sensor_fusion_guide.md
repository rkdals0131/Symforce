# Sensor Fusion Guide

This guide explains the sensor fusion setup using `robot_localization` package to fuse IMU, GPS, and wheel odometry data.

## Overview

The sensor fusion system combines data from three simulated sensors:
- **IMU** (`/ouster/imu_sim`): Provides orientation, angular velocity, and linear acceleration at 100Hz
- **GPS** (`/ublox_gps_node/fix_sim`): Provides global position at 8Hz
- **Wheel Odometry** (`/odom_sim`): Provides position and velocity with cumulative drift

The fusion output is published on `/odom_sim_fusion` using an Extended Kalman Filter (EKF).

## Sensor Characteristics

### IMU Simulation
- Bias drift: 0.1 m/s² (accel), 0.01 rad/s (gyro)
- White noise: 0.002 m/s² (accel), 0.0002 rad/s (gyro)
- Includes gravity compensation
- Publishes full covariance matrices

### GPS Simulation
- Position noise: 2.0m horizontal, 5.0m vertical
- Provides both position and velocity measurements
- Simulates typical consumer-grade GPS accuracy

### Odometry Simulation
- Cumulative drift: 0.5% per meter (linear), 0.2% per radian (angular)
- Position noise: 0.1m, angle noise: 0.05 rad
- Simulates wheel encoder drift and slippage

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
# In a new terminal
ros2 topic echo /odom_sim_fusion
```

### 4. Test the Fusion
```bash
# Run the test script to compare raw and fused odometry
python3 test_sensor_fusion.py
```

## Topics

### Input Topics
- `/ouster/imu_sim` (sensor_msgs/Imu)
- `/ublox_gps_node/fix_sim` (sensor_msgs/NavSatFix)
- `/ublox_gps_node/fix_velocity_sim` (geometry_msgs/TwistWithCovarianceStamped)
- `/odom_sim` (nav_msgs/Odometry)

### Output Topics
- `/odom_sim_fusion` (nav_msgs/Odometry) - Fused odometry output
- `/odometry/gps` (nav_msgs/Odometry) - GPS converted to local coordinates
- `/gps/filtered` (sensor_msgs/NavSatFix) - Filtered GPS output

## Configuration

The EKF configuration is in `config/ekf_sensor_fusion.yaml`. Key parameters:

- `frequency`: 30Hz update rate
- `two_d_mode`: true (constrains to x, y, yaw)
- `process_noise_covariance`: Tunable based on robot dynamics
- Sensor configurations specify which state variables each sensor provides

## Tuning Tips

1. **Process Noise**: Increase if the robot motion is unpredictable
2. **Initial Covariance**: Set based on initial uncertainty
3. **Sensor Timeouts**: Adjust based on expected sensor rates
4. **Covariance Values**: Should match actual sensor noise characteristics

## Troubleshooting

1. **No output on /odom_sim_fusion**:
   - Check that all sensor topics are publishing
   - Verify robot_localization is installed: `ros2 pkg list | grep robot_localization`
   - Check for errors in the EKF node output

2. **Poor fusion performance**:
   - Review sensor covariances in the config file
   - Check for timing issues between sensors
   - Ensure proper TF tree is published

3. **GPS not being fused**:
   - Verify GPS is publishing valid fixes (status > 0)
   - Check that navsat_transform_node is running
   - Ensure proper datum is set