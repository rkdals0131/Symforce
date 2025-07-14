#!/usr/bin/env python3
"""
Launch file for dummy publisher with robot_localization (RoLo) EKF
This runs the exact same dummy publisher configuration with EKF sensor fusion
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def launch_setup(context, *args, **kwargs):
    """Setup function to handle conditional parameter overrides"""
    # Get package directory
    pkg_dir = get_package_share_directory('cc_slam_sym')
    config_file = os.path.join(pkg_dir, 'config', 'dummy_publisher_config.yaml')
    ekf_config_file = os.path.join(pkg_dir, 'config', 'robot_localization_ekf.yaml')
    
    # Start with config file parameters for dummy publisher
    params = []
    if os.path.exists(config_file):
        params.append(config_file)
    
    # Build parameter overrides dict - only include if argument was provided
    param_overrides = {}
    
    # Check if scenario was provided
    scenario_id = 2  # Default to scenario 2
    try:
        scenario_value = LaunchConfiguration('scenario').perform(context)
        if scenario_value:
            scenario_id = int(scenario_value)
            param_overrides['scenario.id'] = scenario_id
    except:
        param_overrides['scenario.id'] = scenario_id
    
    # Check if vehicle_speed was provided
    try:
        speed_value = LaunchConfiguration('vehicle_speed').perform(context)
        if speed_value:
            param_overrides['vehicle.speed'] = float(speed_value)
    except:
        pass
    
    # Check if roi_type was provided
    try:
        roi_value = LaunchConfiguration('roi_type').perform(context)
        if roi_value:
            param_overrides['sensors.cone_detection.roi_type'] = roi_value
    except:
        pass
    
    # Disable odometry simulation - robot_localization will generate odometry from IMU+GPS
    param_overrides['odometry_simulation.enable'] = False
    
    # Add overrides if any were provided
    if param_overrides:
        params.append(param_overrides)
    
    # Create nodes list
    nodes = []
    
    # Dummy publisher node (same as original)
    dummy_publisher_node = Node(
        package='cc_slam_sym',
        executable='dummy_publisher',
        name='dummy_publisher',
        output='screen',
        parameters=params
    )
    nodes.append(dummy_publisher_node)
    
    # EKF localization node from robot_localization
    # Set initial state based on scenario
    initial_x = 30.0 if scenario_id == 2 else 0.0
    initial_y = 12.5 if scenario_id == 2 else 0.0
    
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            ekf_config_file,
            {
                'use_sim_time': False,
                'initial_state': [initial_x, initial_y, 0.0,
                                  0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0]
            }
        ],
        remappings=[
            ('odometry/filtered', '/odom'),  # Output to standard /odom topic
        ]
    )
    nodes.append(ekf_node)
    
    # Navsat transform not needed - we publish GPS as local odometry directly
    # This bypasses the complexity of lat/lon to UTM conversion
    
    return nodes

def generate_launch_description():
    # Declare launch arguments with empty defaults (same as dummy_publisher_launch.py)
    # Empty string means "use config file value"
    scenario_arg = DeclareLaunchArgument(
        'scenario',
        default_value='',
        description='Scenario to use (1: straight track, 2: FS track) - overrides config file'
    )
    
    vehicle_speed_arg = DeclareLaunchArgument(
        'vehicle_speed',
        default_value='',
        description='Vehicle speed in m/s - overrides config file'
    )
    
    roi_type_arg = DeclareLaunchArgument(
        'roi_type',
        default_value='',
        description='ROI visualization type: sector or rectangle - overrides config file'
    )
    
    return LaunchDescription([
        scenario_arg,
        vehicle_speed_arg,
        roi_type_arg,
        OpaqueFunction(function=launch_setup)
    ])