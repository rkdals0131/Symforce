#!/usr/bin/env python3
"""
Launch file for CC-SLAM-SYM
Simple launch file that only starts the SLAM node
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate launch description for SLAM system"""
    
    # Get package share directory
    pkg_share = get_package_share_directory('cc_slam_sym')
    
    # Config file path
    config_file = os.path.join(pkg_share, 'config', 'slam_config.yaml')
    
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=config_file,
        description='Path to SLAM configuration file'
    )
    
    # SLAM node
    slam_node = Node(
        package='cc_slam_sym',
        executable='slam_node',
        name='cc_slam_node',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ]
    )
    
    return LaunchDescription([
        # Arguments
        use_sim_time_arg,
        config_file_arg,
        
        # Just the SLAM node
        slam_node,
    ])