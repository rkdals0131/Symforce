#!/usr/bin/env python3
"""
Unified launch file for CC-SLAM-SYM with mode selection
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('cc_slam_sym')
    
    # Declare launch arguments
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='simulation_basic',
        description='System operation mode',
        choices=['simulation_basic', 'simulation_gps', 'external_ekf', 'direct_fusion']
    )
    
    scenario_arg = DeclareLaunchArgument(
        'scenario',
        default_value='2',
        description='Simulation scenario (1: straight track, 2: formula student)',
        choices=['1', '2']
    )
    
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz for visualization'
    )
    
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Enable debug logging'
    )
    
    # Load configuration files
    slam_config = os.path.join(pkg_dir, 'config', 'slam_config.yaml')
    system_modes_config = os.path.join(pkg_dir, 'config', 'system_modes.yaml')
    dummy_publisher_config = os.path.join(pkg_dir, 'config', 'dummy_publisher_config.yaml')
    ekf_config = os.path.join(pkg_dir, 'config', 'robot_localization_ekf.yaml')
    
    # SLAM Node (always launched)
    slam_node = Node(
        package='cc_slam_sym',
        executable='slam_node',
        name='cc_slam_node',
        parameters=[
            slam_config,
            system_modes_config,
            {'system_mode': LaunchConfiguration('mode')},
            {'debug.verbose': LaunchConfiguration('debug')}
        ],
        output='screen'
    )
    
    # Dummy Publisher (for simulation modes)
    dummy_publisher = Node(
        package='cc_slam_sym',
        executable='dummy_publisher_node',
        name='dummy_publisher',
        parameters=[
            dummy_publisher_config,
            system_modes_config,
            {'system_mode': LaunchConfiguration('mode')},
            {'scenario.id': LaunchConfiguration('scenario')}
        ],
        condition=IfCondition(
            PythonExpression([
                "'", LaunchConfiguration('mode'), "' in ['simulation_basic', 'simulation_gps']"
            ])
        ),
        output='screen'
    )
    
    # Robot Localization EKF (for external_ekf mode)
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_localization',
        parameters=[ekf_config],
        condition=LaunchConfigurationEquals('mode', 'external_ekf'),
        output='screen'
    )
    
    # Navsat Transform (for real GPS)
    navsat_node = Node(
        package='robot_localization',
        executable='navsat_transform_node',
        name='navsat_transform',
        parameters=[ekf_config],
        condition=LaunchConfigurationEquals('mode', 'external_ekf'),
        output='screen'
    )
    
    # RViz
    rviz_config = os.path.join(pkg_dir, 'rviz', 'slam_config.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        condition=IfCondition(LaunchConfiguration('use_rviz'))
    )
    
    # RQt Console for debugging
    rqt_console = ExecuteProcess(
        cmd=['ros2', 'run', 'rqt_console', 'rqt_console'],
        condition=IfCondition(LaunchConfiguration('debug'))
    )
    
    # Mode information printer
    mode_info = ExecuteProcess(
        cmd=['echo', '\\n====================================\\n',
             'Starting CC-SLAM-SYM in mode:', LaunchConfiguration('mode'),
             '\\n====================================\\n'],
        output='screen'
    )
    
    return LaunchDescription([
        # Arguments
        mode_arg,
        scenario_arg,
        use_rviz_arg,
        debug_arg,
        
        # Mode info
        mode_info,
        
        # Nodes
        slam_node,
        dummy_publisher,
        ekf_node,
        navsat_node,
        rviz_node,
        rqt_console
    ])