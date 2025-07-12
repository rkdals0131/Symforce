#!/usr/bin/env python3
"""
Launch file for dummy publisher
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
    
    # Start with config file parameters
    params = []
    if os.path.exists(config_file):
        params.append(config_file)
    
    # Build parameter overrides dict - only include if argument was provided
    param_overrides = {}
    
    # Check if scenario was provided
    try:
        scenario_value = LaunchConfiguration('scenario').perform(context)
        if scenario_value:
            param_overrides['scenario.id'] = int(scenario_value)
    except:
        pass
    
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
    
    # Add overrides if any were provided
    if param_overrides:
        params.append(param_overrides)
    
    # Create and return the node
    return [Node(
        package='cc_slam_sym',
        executable='dummy_publisher',
        name='dummy_publisher',
        output='screen',
        parameters=params
    )]

def generate_launch_description():
    # Declare launch arguments with empty defaults
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