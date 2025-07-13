#!/usr/bin/env python3
"""
Launch file for sensor fusion using robot_localization EKF
Fuses IMU, GPS, and Odometry data from the dummy publisher
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # Get package directories
    pkg_share = get_package_share_directory('cc_slam_sym')
    
    # Paths to configuration files
    ekf_config = os.path.join(pkg_share, 'config', 'ekf_sensor_fusion.yaml')
    dummy_config = os.path.join(pkg_share, 'config', 'dummy_publisher_config.yaml')
    
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    scenario = LaunchConfiguration('scenario', default='1')
    vehicle_speed = LaunchConfiguration('vehicle_speed', default='5.0')
    
    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    declare_scenario = DeclareLaunchArgument(
        'scenario',
        default_value='1',
        description='Scenario ID (1: straight track, 2: Formula Student track)'
    )
    
    declare_vehicle_speed = DeclareLaunchArgument(
        'vehicle_speed',
        default_value='5.0',
        description='Vehicle speed in m/s'
    )
    
    # Include dummy publisher launch file with parameters
    dummy_publisher_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_share, 'launch', 'dummy_publisher_launch.py')
        ]),
        launch_arguments={
            'scenario': scenario,
            'vehicle_speed': vehicle_speed,
            'use_sim_time': use_sim_time
        }.items()
    )
    
    # EKF localization node from robot_localization package
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            ekf_config,
            {'use_sim_time': use_sim_time}
        ],
        remappings=[
            ('odometry/filtered', '/odom_sim_fusion'),  # Output fused odometry
        ]
    )
    
    # Optional: navsat_transform_node for converting GPS to local coordinates
    # This is useful if you want to transform GPS coordinates to map frame
    navsat_transform_node = Node(
        package='robot_localization',
        executable='navsat_transform_node',
        name='navsat_transform_node',
        output='screen',
        parameters=[
            {'frequency': 30.0},
            {'delay': 3.0},
            {'magnetic_declination_radians': 0.0},
            {'yaw_offset': 0.0},
            {'zero_altitude': True},
            {'broadcast_utm_transform': True},
            {'publish_filtered_gps': True},
            {'use_odometry_yaw': True},
            {'wait_for_datum': False},
            {'use_sim_time': use_sim_time}
        ],
        remappings=[
            ('imu/data', '/ouster/imu_sim'),
            ('gps/fix', '/ublox_gps_node/fix_sim'),
            ('odometry/filtered', '/odom_sim_fusion'),
            ('odometry/gps', '/odometry/gps'),
            ('gps/filtered', '/gps/filtered')
        ]
    )
    
    # Visualization node for sensor fusion output
    fusion_viz_node = Node(
        package='cc_slam_sym',
        executable='sensor_fusion_visualizer',
        name='sensor_fusion_visualizer',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'publish_rate': 10.0
        }],
        remappings=[
            ('odom_raw', '/odom_sim'),
            ('odom_fused', '/odom_sim_fusion'),
            ('gps_odom', '/odometry/gps')
        ]
    )
    
    # Static transform publishers for sensor frames (if not defined in URDF)
    # Base link to IMU
    static_tf_base_to_imu = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_base_to_imu',
        arguments=['0', '0', '0.1', '0', '0', '0', 'base_link', 'imu_link']
    )
    
    # Base link to GPS
    static_tf_base_to_gps = Node(
        package='tf2_ros',
        executable='static_transform_publisher', 
        name='static_tf_base_to_gps',
        arguments=['0', '0', '0.2', '0', '0', '0', 'base_link', 'gps_link']
    )
    
    return LaunchDescription([
        # Launch arguments
        declare_use_sim_time,
        declare_scenario,
        declare_vehicle_speed,
        
        # Static transforms
        static_tf_base_to_imu,
        static_tf_base_to_gps,
        
        # Launch dummy publisher
        dummy_publisher_launch,
        
        # Launch EKF node
        ekf_node,
        
        # Launch navsat transform node
        navsat_transform_node,
        
        # Launch visualization (optional - create this node if needed)
        # fusion_viz_node
    ])