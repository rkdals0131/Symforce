#!/usr/bin/env python3
"""
ROS2 SLAM Node for CC-SLAM-SYM
Subscribes to cone detections and odometry, performs SLAM, publishes visualization
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time

# ROS messages
from custom_interface.msg import TrackedConeArray, TrackedCone
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped, Point
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header
import tf2_ros
from tf_transformations import quaternion_from_euler

# Standard imports
import numpy as np
from typing import List, Dict, Optional, Tuple
import time
import threading

# SLAM imports
from cc_slam_sym.slam_core.frontend import SlamFrontend, FrontendConfig
from cc_slam_sym.slam_core.backend import SlamBackend, BackendConfig
from cc_slam_sym.slam_core.data_association import AssociationConfig
from cc_slam_sym.utils.data_structures import ConeCluster, OdometryData, Keyframe, Landmark
from cc_slam_sym.utils.visualization import VisualizationHelper, publish_cone_array
from cc_slam_sym.utils.concurrent_containers import AsyncWorker, ConcurrentVector
from cc_slam_sym.ros_bridge.data_converter import RosSlamConverter, TfPublisher


class SlamNode(Node):
    """ROS2 node for CC-SLAM-SYM with integrated async processing"""
    
    def __init__(self):
        super().__init__('cc_slam_node')
        
        # Declare parameters
        self._declare_parameters()
        
        # Initialize components
        self._init_slam_components()
        self._init_ros_interfaces()
        
        # Create processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.running = True
        
        # Input queues
        max_queue_size = self.get_parameter('performance.max_queue_size').value
        self.cone_queue = ConcurrentVector(max_size=max_queue_size)
        self.odom_queue = ConcurrentVector(max_size=max_queue_size)
        
        # State tracking
        self.last_optimization_time = time.time()
        self.frame_count = 0
        
        # Start processing
        self.processing_thread.start()
        
        # Visualization timer
        viz_period = 1.0 / self.get_parameter('visualization.publish_rate').value
        self.viz_timer = self.create_timer(viz_period, self._visualization_callback)
        
        self.get_logger().info("CC-SLAM-SYM node initialized")
        
    def _declare_parameters(self):
        """Declare ROS2 parameters"""
        # Topics
        self.declare_parameter('topics.cone_detection', '/fused_sorted_cones_ukf_sim')
        self.declare_parameter('topics.odometry', '/odom_sim')
        self.declare_parameter('topics.imu', '/ouster/imu_sim')
        self.declare_parameter('topics.gps', '/ublox_gps_node/fix_sim')
        
        # Frame IDs
        self.declare_parameter('frames.map', 'map')
        self.declare_parameter('frames.odom', 'odom')
        self.declare_parameter('frames.base_link', 'base_link')
        
        # Frontend parameters
        self.declare_parameter('frontend.keyframe_distance_threshold', 2.0)
        self.declare_parameter('frontend.keyframe_rotation_threshold', 0.3)
        self.declare_parameter('frontend.keyframe_time_threshold', 1.0)
        self.declare_parameter('frontend.max_association_distance', 1.5)
        self.declare_parameter('frontend.require_color_match', True)
        self.declare_parameter('frontend.min_observations_for_landmark', 2)
        self.declare_parameter('frontend.max_landmark_init_distance', 20.0)
        self.declare_parameter('frontend.odometry_position_noise', 0.1)
        self.declare_parameter('frontend.odometry_rotation_noise', 0.05)
        self.declare_parameter('frontend.landmark_position_noise', 0.2)
        
        # Backend parameters
        self.declare_parameter('backend.optimization_interval', 5)
        self.declare_parameter('backend.max_iterations', 100)
        self.declare_parameter('backend.relinearize_threshold', 0.1)
        self.declare_parameter('backend.relinearize_skip', 10)
        self.declare_parameter('backend.prior_position_noise', 0.1)
        self.declare_parameter('backend.prior_rotation_noise', 0.05)
        self.declare_parameter('backend.odometry_position_noise', 0.2)
        self.declare_parameter('backend.odometry_rotation_noise', 0.1)
        self.declare_parameter('backend.landmark_observation_noise', 0.3)
        self.declare_parameter('backend.use_robust_kernels', True)
        self.declare_parameter('backend.huber_parameter', 1.0)
        self.declare_parameter('backend.use_sliding_window', True)
        self.declare_parameter('backend.max_keyframes', 50)
        self.declare_parameter('backend.enable_loop_closure', False)
        
        # Visualization parameters
        self.declare_parameter('visualization.publish_rate', 10.0)
        self.declare_parameter('visualization.publish_tf', True)
        self.declare_parameter('visualization.publish_path', True)
        self.declare_parameter('visualization.publish_graph_markers', True)
        
        # Performance parameters
        self.declare_parameter('performance.max_queue_size', 100)
        self.declare_parameter('performance.processing_thread_sleep', 0.001)
        
    def _init_slam_components(self):
        """Initialize SLAM components"""
        # Frontend configuration
        frontend_config = FrontendConfig(
            keyframe_distance_threshold=self.get_parameter('frontend.keyframe_distance_threshold').value,
            keyframe_rotation_threshold=self.get_parameter('frontend.keyframe_rotation_threshold').value,
            keyframe_time_threshold=self.get_parameter('frontend.keyframe_time_threshold').value,
            association_config=AssociationConfig(
                max_distance_threshold=self.get_parameter('frontend.max_association_distance').value,
                color_match_required=self.get_parameter('frontend.require_color_match').value,
                min_observations_for_landmark=self.get_parameter('frontend.min_observations_for_landmark').value
            ),
            min_observations_for_landmark=self.get_parameter('frontend.min_observations_for_landmark').value,
            max_landmark_init_distance=self.get_parameter('frontend.max_landmark_init_distance').value,
            odometry_position_noise=self.get_parameter('frontend.odometry_position_noise').value,
            odometry_rotation_noise=self.get_parameter('frontend.odometry_rotation_noise').value,
            landmark_position_noise=self.get_parameter('frontend.landmark_position_noise').value
        )
        
        # Backend configuration
        backend_config = BackendConfig(
            optimization_interval=self.get_parameter('backend.optimization_interval').value,
            max_iterations=self.get_parameter('backend.max_iterations').value,
            relinearize_threshold=self.get_parameter('backend.relinearize_threshold').value,
            relinearize_skip=self.get_parameter('backend.relinearize_skip').value,
            prior_position_noise=self.get_parameter('backend.prior_position_noise').value,
            prior_rotation_noise=self.get_parameter('backend.prior_rotation_noise').value,
            odometry_position_noise=self.get_parameter('backend.odometry_position_noise').value,
            odometry_rotation_noise=self.get_parameter('backend.odometry_rotation_noise').value,
            landmark_observation_noise=self.get_parameter('backend.landmark_observation_noise').value,
            use_robust_kernels=self.get_parameter('backend.use_robust_kernels').value,
            huber_parameter=self.get_parameter('backend.huber_parameter').value,
            use_sliding_window=self.get_parameter('backend.use_sliding_window').value,
            max_keyframes=self.get_parameter('backend.max_keyframes').value
        )
        
        # Create SLAM components
        self.frontend = SlamFrontend(frontend_config)
        self.backend = SlamBackend(backend_config)
        
        # Data converter
        self.converter = RosSlamConverter()
        
        # Thread safety
        self.slam_lock = threading.RLock()
        
    def _init_ros_interfaces(self):
        """Initialize ROS2 publishers and subscribers"""
        # QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers
        self.cone_sub = self.create_subscription(
            TrackedConeArray,
            self.get_parameter('topics.cone_detection').value,
            self._cone_callback,
            sensor_qos
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            self.get_parameter('topics.odometry').value,
            self._odometry_callback,
            sensor_qos
        )
        
        # Publishers
        self.landmark_pub = self.create_publisher(
            MarkerArray, '/slam/landmarks', reliable_qos
        )
        
        self.keyframe_pub = self.create_publisher(
            MarkerArray, '/slam/keyframes', reliable_qos
        )
        
        self.path_pub = self.create_publisher(
            Path, '/slam/path', reliable_qos
        )
        
        self.graph_pub = self.create_publisher(
            MarkerArray, '/slam/factor_graph', reliable_qos
        )
        
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_publisher = TfPublisher(self.tf_broadcaster)
        
    def _cone_callback(self, msg: TrackedConeArray):
        """Handle cone detection messages"""
        # Queue for processing
        self.cone_queue.push_back(msg, block=False)
        
    def _odometry_callback(self, msg: Odometry):
        """Handle odometry messages"""
        # Queue for processing
        self.odom_queue.push_back(msg, block=False)
        
    def _processing_loop(self):
        """Main processing loop (runs in separate thread)"""
        while self.running:
            # Process odometry
            odom_msg = self.odom_queue.try_pop_front()
            if odom_msg:
                self._process_odometry(odom_msg)
            
            # Process cone observations
            cone_msg = self.cone_queue.try_pop_front()
            if cone_msg:
                self._process_cones(cone_msg)
            
            # Sleep briefly to avoid busy waiting
            time.sleep(self.get_parameter('performance.processing_thread_sleep').value)
            
    def _process_odometry(self, msg: Odometry):
        """Process odometry data"""
        try:
            with self.slam_lock:
                # Convert to internal format
                odom_data = self.converter.odometry_to_data(msg)
                
                # Update frontend
                pose = self.frontend.process_odometry(odom_data)
                self.get_logger().debug(f"Updated pose: x={pose.x():.2f}, y={pose.y():.2f}, theta={pose.theta():.2f}")
                
        except Exception as e:
            self.get_logger().error(f"Error processing odometry: {e}")
            
    def _process_cones(self, msg: TrackedConeArray):
        """Process cone observations"""
        try:
            with self.slam_lock:
                # Convert to internal format
                observations = self.converter.cone_array_to_clusters(msg)
                
                if not observations:
                    return
                
                # Get timestamp
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                
                # Process observations
                association_result = self.frontend.process_cone_observations(
                    observations, timestamp
                )
                
                # Check if new keyframe should be created
                if self.frontend.should_create_keyframe(timestamp):
                    keyframe = self.frontend.create_keyframe(timestamp, observations)
                    
                    if keyframe:
                        # Store keyframe
                        if not hasattr(self.backend, 'keyframes'):
                            self.backend.keyframes = {}
                        if not hasattr(self.backend, 'landmarks'):
                            self.backend.landmarks = {}
                            
                        # Ensure keyframe has a valid pose
                        if keyframe.pose is None:
                            self.get_logger().error(f"Keyframe {keyframe.id} has None pose!")
                            return
                        
                        self.get_logger().info(f"Created keyframe {keyframe.id} at pose: x={keyframe.pose.x():.2f}, y={keyframe.pose.y():.2f}, theta={keyframe.pose.theta():.2f}")
                            
                        self.backend.keyframes[keyframe.id] = keyframe
                        
                        # Add prior for first keyframe
                        if keyframe.id == 0:
                            self.backend.add_prior(keyframe)
                        else:
                            # Add odometry factor between consecutive keyframes
                            prev_keyframe = self.backend.keyframes.get(keyframe.id - 1)
                            if prev_keyframe and prev_keyframe.pose is not None:
                                self.backend.add_odometry_factor(prev_keyframe, keyframe)
                        
                        # Add landmark observations
                        for obs in keyframe.observations:
                            # Find corresponding landmark
                            for lm_id, landmark in self.frontend.landmarks.items():
                                if obs.track_id == lm_id:
                                    # Update backend's landmark list
                                    self.backend.landmarks[lm_id] = landmark
                                    
                                    # Add observation factor
                                    self.backend.add_landmark_observation(keyframe, landmark, obs)
                                    break
                        
                        # Check if optimization should run
                        if len(self.backend.keyframes) % self.backend.config.optimization_interval == 0:
                            self._perform_optimization()
                        
                        self.get_logger().debug(f"Backend now has {len(self.backend.keyframes)} keyframes")
                            
        except Exception as e:
            self.get_logger().error(f"Error processing cones: {e}")
            
    def _perform_optimization(self):
        """Run backend optimization"""
        try:
            start_time = time.time()
            success = self.backend.optimize()
            
            if success:
                opt_time = time.time() - start_time
                stats = self.backend.get_factor_graph_stats()
                self.get_logger().info(
                    f"Optimization complete in {opt_time*1000:.1f}ms, "
                    f"factors: {stats['num_factors']}, values: {stats['num_values']}"
                )
                self.last_optimization_time = time.time()
                
        except Exception as e:
            self.get_logger().error(f"Optimization error: {e}")
            
    def _visualization_callback(self):
        """Publish visualization data"""
        try:
            current_time = self.get_clock().now()
            
            with self.slam_lock:
                # Publish landmarks
                if self.frontend.landmarks:
                    try:
                        self._publish_landmarks(self.frontend.landmarks, current_time)
                    except Exception as e:
                        self.get_logger().error(f"Error publishing landmarks: {e}")
                
                # Publish keyframes and path
                if self.backend.keyframes:
                    try:
                        self._publish_keyframes(current_time)
                    except Exception as e:
                        self.get_logger().error(f"Error publishing keyframes: {e}")
                        import traceback
                        self.get_logger().error(f"Traceback: {traceback.format_exc()}")
                        
                    try:
                        self._publish_path(current_time)
                    except Exception as e:
                        self.get_logger().error(f"Error publishing path: {e}")
                
                # Publish factor graph
                if self.get_parameter('visualization.publish_graph_markers').value:
                    try:
                        self._publish_factor_graph(current_time)
                    except Exception as e:
                        self.get_logger().error(f"Error publishing factor graph: {e}")
                    
        except Exception as e:
            self.get_logger().error(f"Visualization error: {e}")
            
    def _publish_landmarks(self, landmarks: Dict[int, Landmark], timestamp: Time):
        """Publish landmark visualization"""
        # Convert to format for visualization helper
        cone_list = []
        for lm_id, landmark in landmarks.items():
            cone_list.append({
                'pos': landmark.position,
                'type': landmark.color,
                'id': lm_id
            })
            
        publish_cone_array(
            self.landmark_pub,
            cone_list,
            namespace="landmarks",
            frame_id=self.get_parameter('frames.map').value,
            with_text=True,
            timestamp=timestamp
        )
        
    def _publish_keyframes(self, timestamp: Time):
        """Publish keyframe visualization"""
        marker_array = MarkerArray()
        
        # Delete all previous markers
        delete_marker = VisualizationHelper.create_delete_all_marker(
            "keyframes", self.get_parameter('frames.map').value, timestamp
        )
        marker_array.markers.append(delete_marker)
        
        # Create keyframe markers
        for i, (kf_id, kf) in enumerate(self.backend.keyframes.items()):
            # Debug keyframe structure
            if kf is None:
                self.get_logger().error(f"Keyframe {kf_id} is None!")
                continue
                
            if not hasattr(kf, 'pose'):
                self.get_logger().error(f"Keyframe {kf_id} has no 'pose' attribute")
                continue
                
            # Get optimized or initial pose
            pose = None
            
            # First try to get optimized pose from backend estimate
            if self.backend.current_estimate is not None:
                try:
                    if self.backend.current_estimate.exists(kf.pose_symbol):
                        pose = self.backend.current_estimate.atPose2(kf.pose_symbol)
                        self.get_logger().debug(f"Using optimized pose for keyframe {kf_id}")
                except Exception as e:
                    self.get_logger().warn(f"Failed to get optimized pose for keyframe {kf_id}: {e}")
                    
            # Fall back to initial pose from keyframe
            if pose is None and kf.pose is not None:
                pose = kf.pose
                self.get_logger().debug(f"Using initial pose for keyframe {kf_id}")
                
            if pose is None:
                self.get_logger().error(f"Keyframe {kf_id} has no pose (optimized or initial), kf.pose={kf.pose if hasattr(kf, 'pose') else 'no pose attr'}")
                continue
                
            # Keyframe position marker
            marker = Marker()
            marker.header.stamp = timestamp.to_msg()
            marker.header.frame_id = self.get_parameter('frames.map').value
            marker.ns = "keyframes"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            try:
                marker.pose.position.x = pose.x()
                marker.pose.position.y = pose.y()
                marker.pose.position.z = 0.0
                marker.pose.orientation.w = 1.0
            except AttributeError as e:
                self.get_logger().error(f"Keyframe {kf_id} pose is not a valid gtsam.Pose2 object: {type(pose)}, error: {e}")
                continue
            
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            
            marker_array.markers.append(marker)
            
        self.keyframe_pub.publish(marker_array)
        
    def _publish_path(self, timestamp: Time):
        """Publish optimized trajectory"""
        trajectory = self.backend.get_optimized_trajectory()
        
        if trajectory:
            path_msg = self.converter.create_path_message(
                trajectory, timestamp, self.get_parameter('frames.map').value
            )
            self.path_pub.publish(path_msg)
            
    def _publish_factor_graph(self, timestamp: Time):
        """Publish factor graph visualization"""
        marker_array = MarkerArray()
        
        # Delete all previous markers
        delete_marker = VisualizationHelper.create_delete_all_marker(
            "graph", self.get_parameter('frames.map').value, timestamp
        )
        marker_array.markers.append(delete_marker)
        
        # Check if we have data to visualize
        if not hasattr(self.backend, 'keyframes') or not self.backend.keyframes:
            self.graph_pub.publish(marker_array)
            return
            
        if not self.backend.graph:
            self.graph_pub.publish(marker_array)
            return
            
        marker_id = 0
        
        # Visualize pose-to-pose factors (odometry)
        for i in range(len(self.backend.keyframes) - 1):
            if i in self.backend.keyframes and i+1 in self.backend.keyframes:
                kf1 = self.backend.keyframes[i]
                kf2 = self.backend.keyframes[i+1]
                
                # Get poses (optimized or initial)
                pose1 = None
                pose2 = None
                
                # Try to get optimized poses if available
                if self.backend.current_estimate is not None:
                    try:
                        if self.backend.current_estimate.exists(kf1.pose_symbol) and \
                           self.backend.current_estimate.exists(kf2.pose_symbol):
                            pose1 = self.backend.current_estimate.atPose2(kf1.pose_symbol)
                            pose2 = self.backend.current_estimate.atPose2(kf2.pose_symbol)
                    except Exception:
                        pass
                        
                # Fall back to initial poses
                if pose1 is None and hasattr(kf1, 'pose') and kf1.pose is not None:
                    pose1 = kf1.pose
                if pose2 is None and hasattr(kf2, 'pose') and kf2.pose is not None:
                    pose2 = kf2.pose
                    
                if pose1 is not None and pose2 is not None:
                    
                    # Create line marker
                    marker = Marker()
                    marker.header.stamp = timestamp.to_msg()
                    marker.header.frame_id = self.get_parameter('frames.map').value
                    marker.ns = "graph"
                    marker.id = marker_id
                    marker.type = Marker.LINE_STRIP
                    marker.action = Marker.ADD
                    
                    # Points
                    p1 = Point()
                    p1.x = pose1.x()
                    p1.y = pose1.y()
                    p1.z = 0.0
                    marker.points.append(p1)
                    
                    p2 = Point()
                    p2.x = pose2.x()
                    p2.y = pose2.y()
                    p2.z = 0.0
                    marker.points.append(p2)
                    
                    marker.scale.x = 0.05  # Line width
                    
                    # Blue color for odometry factors
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 1.0
                    marker.color.a = 0.5
                    
                    marker_array.markers.append(marker)
                    marker_id += 1
        
        # Visualize pose-to-landmark factors
        for kf_id, kf in self.backend.keyframes.items():
            pose = None
            
            # Try to get optimized pose if available
            if self.backend.current_estimate is not None:
                try:
                    if self.backend.current_estimate.exists(kf.pose_symbol):
                        pose = self.backend.current_estimate.atPose2(kf.pose_symbol)
                except Exception:
                    pass
                    
            # Fall back to initial pose
            if pose is None and hasattr(kf, 'pose') and kf.pose is not None:
                pose = kf.pose
                
            if pose is not None:
                
                # For each observed landmark
                for obs in kf.observations:
                    # Find corresponding landmark
                    for lm_id, landmark in self.backend.landmarks.items():
                        if obs.track_id == lm_id:
                            lm_pos = None
                            
                            # Try to get optimized landmark position
                            if self.backend.current_estimate is not None:
                                try:
                                    if self.backend.current_estimate.exists(landmark.symbol):
                                        lm_pos = self.backend.current_estimate.atPoint2(landmark.symbol)
                                except Exception:
                                    pass
                                    
                            # Fall back to initial position
                            if lm_pos is None and hasattr(landmark, 'position') and landmark.position is not None:
                                lm_pos = landmark.position
                                
                            if lm_pos is not None:
                                # Create line marker
                                marker = Marker()
                                marker.header.stamp = timestamp.to_msg()
                                marker.header.frame_id = self.get_parameter('frames.map').value
                                marker.ns = "graph"
                                marker.id = marker_id
                                marker.type = Marker.LINE_STRIP
                                marker.action = Marker.ADD
                                
                                # Points
                                p1 = Point()
                                p1.x = pose.x()
                                p1.y = pose.y()
                                p1.z = 0.0
                                marker.points.append(p1)
                                
                                p2 = Point()
                                if isinstance(lm_pos, np.ndarray):
                                    p2.x = lm_pos[0]
                                    p2.y = lm_pos[1]
                                else:
                                    p2.x = lm_pos.x()
                                    p2.y = lm_pos.y()
                                p2.z = 0.0
                                marker.points.append(p2)
                                
                                marker.scale.x = 0.02  # Thinner line
                                
                                # Green color for landmark factors
                                marker.color.r = 0.0
                                marker.color.g = 1.0
                                marker.color.b = 0.0
                                marker.color.a = 0.3
                                
                                marker_array.markers.append(marker)
                                marker_id += 1
                                break
        
        self.graph_pub.publish(marker_array)
        
    def destroy_node(self):
        """Clean shutdown"""
        self.get_logger().info("Shutting down SLAM node...")
        self.running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join()
        super().destroy_node()


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    node = SlamNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()