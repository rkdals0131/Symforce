#!/usr/bin/env python3
"""
Launch file for dummy publisher
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    scenario_arg = DeclareLaunchArgument(
        'scenario',
        default_value='1',
        description='Scenario to use (1: straight track, 2: FS track)'
    )
    
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='20.0',
        description='Cone publishing rate in Hz'
    )
    
    vehicle_speed_arg = DeclareLaunchArgument(
        'vehicle_speed',
        default_value='5.0',
        description='Vehicle speed in m/s'
    )
    
    # Dummy publisher node
    dummy_publisher = Node(
        package='cc_slam_sym',
        executable='dummy_publisher',
        name='dummy_publisher',
        output='screen',
        parameters=[{
            'scenario': LaunchConfiguration('scenario'),
            'publish_rate': LaunchConfiguration('publish_rate'),
            'imu_rate': 100.0,
            'gps_rate': 8.0,
            'cone_detection_range': 15.0,
            'cone_fov_deg': 120.0,
            'vehicle_speed': LaunchConfiguration('vehicle_speed'),
        }]
    )
    
    return LaunchDescription([
        scenario_arg,
        publish_rate_arg,
        vehicle_speed_arg,
        dummy_publisher,
    ])