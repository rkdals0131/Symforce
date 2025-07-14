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

# Matplotlib visualization support (optional)
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Wedge
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


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
    }
    
    @staticmethod
    def create_cone_marker(position: np.ndarray, color: str, marker_id: int, 
                          namespace: str = "cones", frame_id: str = "map",
                          timestamp: Optional[object] = None, ground_truth: bool = False) -> Marker:
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
        if ground_truth:
            # Semi-transparent gray for ground truth cones
            marker.color.r = 0.5
            marker.color.g = 0.5
            marker.color.b = 0.5
            marker.color.a = 0.3
        else:
            # Normal colors for detected cones
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
                      timestamp: Optional[object] = None, ground_truth: bool = False) -> None:
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
            timestamp=timestamp,
            ground_truth=ground_truth
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


# Matplotlib-based visualization (optional, for debugging and analysis)
if MATPLOTLIB_AVAILABLE:
    class MatplotlibVisualizer:
        """Real-time SLAM visualization using matplotlib"""
        
        def __init__(self, xlim=(-10, 30), ylim=(-15, 15), figure_size=(12, 8)):
            """Initialize visualizer
            
            Args:
                xlim: X-axis limits (min, max)
                ylim: Y-axis limits (min, max)
                figure_size: Figure size in inches
            """
            self.xlim = xlim
            self.ylim = ylim
            
            # Create figure and axis
            self.fig, self.ax = plt.subplots(figsize=figure_size)
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self.ax.set_aspect('equal')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_title('CC-SLAM-SYM Visualization')
            
            # Plot elements
            self.robot_marker = None
            self.robot_fov = None
            self.trajectory_line = None
            self.keyframe_markers = None
            self.landmark_markers = {}
            self.observation_lines = []
            
            # Data storage
            self.trajectory_x = []
            self.trajectory_y = []
            self.keyframe_x = []
            self.keyframe_y = []
            
            # Colors for different cone types
            self.cone_colors = {
                "yellow": "gold",
                "blue": "blue",
                "red": "red",
                "unknown": "gray"
            }
            
            # Statistics text
            self.stats_text = None
            
            # Interactive mode
            plt.ion()
            plt.show()
            
        def update(self, 
                  robot_pose: Tuple[float, float, float],
                  landmarks: Dict[int, object],
                  observations: Optional[List[object]] = None,
                  is_keyframe: bool = False,
                  stats: Optional[Dict] = None):
            """Update visualization with new SLAM data
            
            Args:
                robot_pose: Current robot pose (x, y, theta)
                landmarks: Dictionary of current landmarks
                observations: Current cone observations (optional)
                is_keyframe: Whether current pose is a keyframe
                stats: Optional statistics to display
            """
            # Clear previous dynamic elements
            if self.robot_marker:
                self.robot_marker.remove()
            if self.robot_fov:
                self.robot_fov.remove()
            for line in self.observation_lines:
                line.remove()
            self.observation_lines.clear()
            
            # Update trajectory
            self.trajectory_x.append(robot_pose[0])
            self.trajectory_y.append(robot_pose[1])
            
            # Update trajectory line
            if self.trajectory_line:
                self.trajectory_line.remove()
            self.trajectory_line, = self.ax.plot(
                self.trajectory_x, self.trajectory_y,
                'g-', linewidth=2, alpha=0.7, label='Trajectory'
            )
            
            # Update keyframes
            if is_keyframe:
                self.keyframe_x.append(robot_pose[0])
                self.keyframe_y.append(robot_pose[1])
                if self.keyframe_markers:
                    self.keyframe_markers.remove()
                self.keyframe_markers = self.ax.scatter(
                    self.keyframe_x, self.keyframe_y,
                    c='green', s=50, marker='s', alpha=0.7, label='Keyframes'
                )
            
            # Draw robot
            self._draw_robot(robot_pose)
            
            # Update landmarks
            self._update_landmarks(landmarks)
            
            # Draw observations if provided
            if observations:
                self._draw_observations(robot_pose, observations)
                
            # Update statistics
            if stats:
                self._update_stats(stats)
                
            # Update display
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        def _draw_robot(self, pose: Tuple[float, float, float]):
            """Draw robot at current pose
            
            Args:
                pose: Robot pose (x, y, theta)
            """
            x, y, theta = pose
            
            # Robot body (circle)
            self.robot_marker = Circle(
                (x, y), 0.3, color='red', fill=True, alpha=0.8
            )
            self.ax.add_patch(self.robot_marker)
            
            # Field of view (wedge)
            fov_angle = 60  # degrees
            fov_range = 5   # meters
            self.robot_fov = Wedge(
                (x, y), fov_range,
                np.degrees(theta) - fov_angle/2,
                np.degrees(theta) + fov_angle/2,
                color='red', alpha=0.2
            )
            self.ax.add_patch(self.robot_fov)
            
            # Direction arrow
            arrow_len = 0.5
            dx = arrow_len * np.cos(theta)
            dy = arrow_len * np.sin(theta)
            self.ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.1,
                         fc='red', ec='red')
                         
        def _update_landmarks(self, landmarks: Dict[int, object]):
            """Update landmark visualization
            
            Args:
                landmarks: Dictionary of landmarks
            """
            # Remove old landmarks not in current set
            old_ids = set(self.landmark_markers.keys())
            current_ids = set(landmarks.keys())
            for old_id in old_ids - current_ids:
                if old_id in self.landmark_markers:
                    self.landmark_markers[old_id].remove()
                    del self.landmark_markers[old_id]
                    
            # Update or create landmarks
            for lm_id, landmark in landmarks.items():
                # Access landmark properties
                color = getattr(landmark, 'color', 'unknown')
                color = self.cone_colors.get(color, "gray")
                position = getattr(landmark, 'position', [0, 0])
                
                if lm_id in self.landmark_markers:
                    # Update existing marker
                    self.landmark_markers[lm_id].remove()
                    
                # Create new marker
                marker = Circle(
                    (position[0], position[1]),
                    0.15, color=color, fill=True, alpha=0.8,
                    edgecolor='black', linewidth=1
                )
                self.ax.add_patch(marker)
                self.landmark_markers[lm_id] = marker
                
                # Add landmark ID text
                self.ax.text(
                    position[0] + 0.2,
                    position[1] + 0.2,
                    f"{lm_id}",
                    fontsize=8, alpha=0.7
                )
                
        def _draw_observations(self, 
                             robot_pose: Tuple[float, float, float],
                             observations: List[object]):
            """Draw observation rays from robot to observed cones
            
            Args:
                robot_pose: Current robot pose
                observations: List of cone observations
            """
            robot_x, robot_y, _ = robot_pose
            
            for obs in observations:
                # Get observation position
                obs_pos = getattr(obs, 'position', [0, 0])
                
                # Transform observation to world frame
                obs_world_x = robot_x + obs_pos[0] * np.cos(robot_pose[2]) - obs_pos[1] * np.sin(robot_pose[2])
                obs_world_y = robot_y + obs_pos[0] * np.sin(robot_pose[2]) + obs_pos[1] * np.cos(robot_pose[2])
                
                # Draw observation line
                line, = self.ax.plot(
                    [robot_x, obs_world_x],
                    [robot_y, obs_world_y],
                    'b--', alpha=0.3, linewidth=1
                )
                self.observation_lines.append(line)
                
        def _update_stats(self, stats: Dict):
            """Update statistics display
            
            Args:
                stats: Statistics dictionary
            """
            if self.stats_text:
                self.stats_text.remove()
                
            text = f"Frame: {stats.get('frame_id', 0)}\n"
            text += f"Landmarks: {stats.get('num_landmarks', 0)}\n"
            
            if 'association' in stats:
                assoc = stats['association']
                text += f"Matched: {assoc.get('matched', 0)}\n"
                text += f"New obs: {assoc.get('unmatched_obs', 0)}\n"
                
            if 'processing_time' in stats:
                timing = stats['processing_time']
                text += f"Time: {timing.get('total', 0)*1000:.1f}ms"
                
            self.stats_text = self.ax.text(
                0.02, 0.98, text,
                transform=self.ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
            
        def save_figure(self, filename: str):
            """Save current figure to file
            
            Args:
                filename: Output filename
            """
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            
        def close(self):
            """Close the visualization window"""
            plt.close(self.fig)
