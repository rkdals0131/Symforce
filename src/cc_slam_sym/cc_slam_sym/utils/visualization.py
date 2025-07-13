#!/usr/bin/env python3
"""
Visualization Utilities for CC-SLAM-SYM

Common visualization functions for both simulation and real SLAM systems.
Provides functions for visualizing cones, paths, ROI, and other SLAM-related data.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion
from std_msgs.msg import ColorRGBA
import rclpy.duration


class VisualizationHelper:
    """Helper class for creating ROS visualization markers"""
    
    # Color presets
    COLORS = {
        'yellow': (1.0, 1.0, 0.0, 0.8),
        'blue': (0.0, 0.0, 1.0, 0.8),
        'red': (1.0, 0.0, 0.0, 0.8),
        'green': (0.0, 1.0, 0.0, 0.8),  # For unknown cones
        'white': (1.0, 1.0, 1.0, 0.8),
        'purple': (0.5, 0.0, 0.5, 0.8),
        'orange': (1.0, 0.5, 0.0, 0.8),
    }
    
    @staticmethod
    def create_cone_marker(position: np.ndarray, color: str, marker_id: int, 
                          namespace: str = "cones", frame_id: str = "map",
                          timestamp: Optional[object] = None) -> Marker:
        """Create a cone marker (cylinder)"""
        marker = Marker()
        
        # Header
        if timestamp:
            marker.header.stamp = timestamp.to_msg()
        marker.header.frame_id = frame_id
        
        # Namespace and ID
        marker.ns = namespace
        marker.id = marker_id
        
        # Type and action
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # Pose
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = -0.15  # Cone center height
        marker.pose.orientation.w = 1.0
        
        # Scale (cone dimensions)
        marker.scale.x = 0.3  # Diameter
        marker.scale.y = 0.3
        marker.scale.z = 0.3  # Height
        
        # Color
        color_tuple = VisualizationHelper.COLORS.get(color.lower(), (0.5, 0.5, 0.5, 0.8))
        marker.color.r = color_tuple[0]
        marker.color.g = color_tuple[1]
        marker.color.b = color_tuple[2]
        marker.color.a = color_tuple[3]
        
        # Lifetime (0 = forever)
        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
        
        return marker
    
    @staticmethod
    def create_text_marker(position: np.ndarray, text: str, marker_id: int,
                          namespace: str = "text", frame_id: str = "map",
                          timestamp: Optional[object] = None) -> Marker:
        """Create a text marker above a position"""
        marker = Marker()
        
        # Header
        if timestamp:
            marker.header.stamp = timestamp.to_msg()
        marker.header.frame_id = frame_id
        
        # Namespace and ID
        marker.ns = namespace
        marker.id = marker_id
        
        # Type and action
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Pose (slightly above the position)
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = 0.5  # Above cone
        marker.pose.orientation.w = 1.0
        
        # Text
        marker.text = str(text)
        
        # Scale (text size)
        marker.scale.z = 0.3
        
        # Color (white)
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        # Lifetime
        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
        
        return marker
    
    @staticmethod
    def create_delete_all_marker(namespace: str, frame_id: str = "map",
                               timestamp: Optional[object] = None) -> Marker:
        """Create a marker to delete all markers in a namespace"""
        marker = Marker()
        
        if timestamp:
            marker.header.stamp = timestamp.to_msg()
        marker.header.frame_id = frame_id
        marker.ns = namespace
        marker.action = Marker.DELETEALL
        
        return marker
    
    @staticmethod
    def create_roi_marker(roi_type: str, max_range: float, fov_rad: float,
                         frame_id: str = "base_link", timestamp: Optional[object] = None) -> Marker:
        """Create ROI visualization marker"""
        marker = Marker()
        
        if timestamp:
            marker.header.stamp = timestamp.to_msg()
        marker.header.frame_id = frame_id
        marker.ns = "roi"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Red color for ROI
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        marker.scale.x = 0.1  # Line width
        marker.pose.orientation.w = 1.0
        
        # Generate ROI boundary points
        points = []
        
        if roi_type == 'sector':
            # Origin
            p = Point()
            p.x = 0.0
            p.y = 0.0
            p.z = 0.0
            points.append(p)
            
            # Arc points
            num_arc_points = 20
            for i in range(num_arc_points + 1):
                angle = -fov_rad/2 + i * fov_rad / num_arc_points
                p = Point()
                p.x = max_range * np.cos(angle)
                p.y = max_range * np.sin(angle)
                p.z = 0.0
                points.append(p)
            
            # Back to origin
            p = Point()
            p.x = 0.0
            p.y = 0.0
            p.z = 0.0
            points.append(p)
        
        else:  # rectangle
            # Four corners
            corners = [
                (max_range, max_range/2),
                (max_range, -max_range/2),
                (0, -max_range/2),
                (0, max_range/2),
                (max_range, max_range/2)  # Close the rectangle
            ]
            
            for x, y in corners:
                p = Point()
                p.x = x
                p.y = y
                p.z = 0.0
                points.append(p)
        
        marker.points = points
        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
        
        return marker
    
    @staticmethod
    def create_path_marker(path_points: List[Tuple[float, float]], 
                          color: str = "green", namespace: str = "path",
                          frame_id: str = "map", timestamp: Optional[object] = None) -> Marker:
        """Create a path visualization marker"""
        marker = Marker()
        
        if timestamp:
            marker.header.stamp = timestamp.to_msg()
        marker.header.frame_id = frame_id
        marker.ns = namespace
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Color
        color_tuple = VisualizationHelper.COLORS.get(color.lower(), (0.0, 1.0, 0.0, 0.8))
        marker.color.r = color_tuple[0]
        marker.color.g = color_tuple[1]
        marker.color.b = color_tuple[2]
        marker.color.a = color_tuple[3]
        
        marker.scale.x = 0.05  # Line width
        marker.pose.orientation.w = 1.0
        
        # Add points
        for x, y in path_points:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0
            marker.points.append(p)
        
        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
        
        return marker


def publish_cone_array(publisher, cones: List[Dict], namespace: str = "cones",
                      frame_id: str = "map", with_text: bool = False,
                      timestamp: Optional[object] = None) -> None:
    """
    Publish an array of cones with optional text labels
    
    Args:
        publisher: ROS publisher for MarkerArray
        cones: List of cone dictionaries with 'pos', 'type', and optionally 'id'
        namespace: Marker namespace
        frame_id: TF frame ID
        with_text: Whether to include text labels
        timestamp: ROS timestamp
    """
    marker_array = MarkerArray()
    
    # Clear previous markers
    delete_marker = VisualizationHelper.create_delete_all_marker(namespace, frame_id, timestamp)
    marker_array.markers.append(delete_marker)
    
    if with_text:
        delete_text = VisualizationHelper.create_delete_all_marker(f"{namespace}_text", frame_id, timestamp)
        marker_array.markers.append(delete_text)
    
    # Add cone markers
    for idx, cone in enumerate(cones):
        # Cone marker
        cone_marker = VisualizationHelper.create_cone_marker(
            position=cone['pos'],
            color=cone['type'],
            marker_id=idx,
            namespace=namespace,
            frame_id=frame_id,
            timestamp=timestamp
        )
        marker_array.markers.append(cone_marker)
        
        # Text marker (if requested)
        if with_text and 'id' in cone:
            text_marker = VisualizationHelper.create_text_marker(
                position=cone['pos'],
                text=str(cone['id']),
                marker_id=idx,
                namespace=f"{namespace}_text",
                frame_id=frame_id,
                timestamp=timestamp
            )
            marker_array.markers.append(text_marker)
    
    publisher.publish(marker_array)


def publish_detected_cones(publisher, detected_cones: List[Tuple[int, np.ndarray, str]],
                          namespace: str = "detected_cones", frame_id: str = "base_link",
                          timestamp: Optional[object] = None) -> None:
    """
    Publish detected cones with track IDs
    
    Args:
        publisher: ROS publisher for MarkerArray
        detected_cones: List of (track_id, position, cone_type) tuples
        namespace: Marker namespace
        frame_id: TF frame ID
        timestamp: ROS timestamp
    """
    marker_array = MarkerArray()
    
    # Clear previous markers
    delete_marker = VisualizationHelper.create_delete_all_marker(namespace, frame_id, timestamp)
    marker_array.markers.append(delete_marker)
    
    delete_text = VisualizationHelper.create_delete_all_marker(f"{namespace}_text", frame_id, timestamp)
    marker_array.markers.append(delete_text)
    
    # Add detected cone markers
    for idx, (track_id, position, cone_type) in enumerate(detected_cones):
        # Cone marker
        cone_marker = VisualizationHelper.create_cone_marker(
            position=position,
            color=cone_type,
            marker_id=idx * 2,
            namespace=namespace,
            frame_id=frame_id,
            timestamp=timestamp
        )
        marker_array.markers.append(cone_marker)
        
        # Track ID text
        text_marker = VisualizationHelper.create_text_marker(
            position=position,
            text=f"ID: {track_id}",
            marker_id=idx * 2 + 1,
            namespace=f"{namespace}_text",
            frame_id=frame_id,
            timestamp=timestamp
        )
        marker_array.markers.append(text_marker)
    
    publisher.publish(marker_array)