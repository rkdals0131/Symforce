# Detection Error Simulation for CC-SLAM-SYM

## Overview

This document outlines the detection error simulation features implemented to make the dummy publisher more realistic for SLAM algorithm development.

## Implemented Features

### ✅ Detection Error Simulation
- **False Negative**: Configurable rate to miss real cones (simulates sensor occlusion, range limits)
- **False Positive**: Random fake cone generation within ROI (simulates noise, artifacts)
- **Wrong Color Classification**: Configurable rate for incorrect color detection
- **Unknown Classification**: Configurable rate for fusion failure (LiDAR detected, camera failed)

### ✅ Configuration Parameters
All parameters are configurable via `dummy_publisher_config.yaml`:
```yaml
detection_errors:
  enable: true                  # Master switch
  false_negative_rate: 0.07     # 7% miss rate (current default)
  false_positive_rate: 0.002    # 0.2% fake detection rate  
  wrong_color_rate: 0.002       # 0.2% wrong color rate
  unknown_color_rate: 0.08      # 8% unknown classification rate
```

### ✅ Formula Student Compliance
- **Orange cones completely removed** - Not used in Formula Student Driverless
- **Supported colors**: Yellow, Blue, Red only
- **Unknown classification**: Green visualization for LiDAR-only detections

### ✅ Visualization
- **Unknown cones**: Green markers in RViz
- **Wrong colors**: Display with detected (wrong) color
- **False positives**: Random colored cones
- **Track IDs**: Displayed above each cone

## Simulation Rationale

### LiDAR + Camera Fusion Modeling
The detection errors simulate real-world sensor fusion challenges:

1. **LiDAR detection** → Creates 3D point, but no color info
2. **Camera fusion** → Adds color classification
3. **Fusion failures** → Result in "Unknown" classification (green)
4. **Misclassification** → Wrong color assigned
5. **Miss detection** → Cone present but not detected
6. **False detection** → Sensor artifacts appear as cones

### Hardware Context
- **LiDAR**: Ouster OS1-32ch
- **IMU**: myAHRS+ (planned) / Ouster internal 6-axis (current)
- **GPS**: Ublox ZED-F9P
- **Fusion**: CALICO package output simulation

## Usage

### Enable/Disable Errors
```bash
# Disable all detection errors
ros2 param set /dummy_publisher sensors.detection_errors.enable false

# Adjust specific error rates
ros2 param set /dummy_publisher sensors.detection_errors.unknown_color_rate 0.2
```

### Launch with Custom Config
```bash
ros2 launch cc_slam_sym dummy_publisher_launch.py
```

## SLAM Algorithm Testing

These errors help test SLAM robustness:

### Data Association Challenges
- **Multiple track IDs** for same physical cone (re-entry after miss)
- **False landmarks** that should be rejected
- **Color inconsistency** requiring robust classification

### Expected SLAM Behavior
- **Outlier rejection** for false positives
- **Landmark merging** for re-observed cones
- **Robust optimization** despite noisy observations

## Configuration Guidelines

### Conservative Settings (Easier SLAM)
```yaml
false_negative_rate: 0.01    # 1% miss
false_positive_rate: 0.005   # 0.5% fake
wrong_color_rate: 0.02       # 2% wrong color
unknown_color_rate: 0.05     # 5% unknown
```

### Aggressive Settings (Challenging SLAM)
```yaml
false_negative_rate: 0.05    # 5% miss
false_positive_rate: 0.02    # 2% fake
wrong_color_rate: 0.1        # 10% wrong color
unknown_color_rate: 0.2      # 20% unknown
```

### Debugging (Perfect Sensors)
```yaml
enable: false                # All errors disabled
```

## Future Improvements

### Potential Enhancements
- Distance-dependent error rates
- Occlusion-based miss detection
- Systematic color bias simulation
- Temporal correlation in errors

### Not Planned (KISS Principle)
- Full physics simulation
- Weather effects
- Advanced sensor modeling
- Vehicle dynamics

---

*Last updated: 2025-01-12*
*Implemented for: SLAM algorithm development and testing*