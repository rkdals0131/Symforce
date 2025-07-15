#!/usr/bin/env python3
"""
SLAM Debug Monitor
Real-time debugging and health monitoring utility for CC-SLAM-SYM
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import MultiThreadedExecutor

from diagnostic_msgs.msg import DiagnosticArray
from custom_interface.msg import TrackedConeArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, NavSatFix

import time
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import os


@dataclass
class TopicStatus:
    """Status of a single topic"""
    name: str
    msg_type: str
    required: bool = True
    timeout: float = 2.0  # seconds
    min_rate: float = 1.0  # Hz
    
    # Runtime statistics
    received_count: int = 0
    last_received: Optional[float] = None
    timestamps: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update(self, timestamp: float):
        """Update topic statistics"""
        self.received_count += 1
        self.last_received = timestamp
        self.timestamps.append(timestamp)
        
    def get_rate(self) -> float:
        """Calculate message rate (Hz)"""
        if len(self.timestamps) < 2:
            return 0.0
        time_span = self.timestamps[-1] - self.timestamps[0]
        if time_span > 0:
            return (len(self.timestamps) - 1) / time_span
        return 0.0
        
    def is_healthy(self, current_time: float) -> Tuple[bool, str]:
        """Check if topic is healthy"""
        # Check if ever received
        if self.received_count == 0:
            return False, "Never received"
            
        # Check minimum messages
        if self.received_count < 2:
            return False, f"Only {self.received_count} message received"
            
        # Check timeout
        time_since_last = current_time - self.last_received
        if time_since_last > self.timeout:
            return False, f"Timeout ({time_since_last:.1f}s)"
            
        # Check rate
        rate = self.get_rate()
        if rate < self.min_rate:
            return False, f"Low rate ({rate:.1f} Hz)"
            
        return True, "OK"


class SlamDebugMonitor(Node):
    """Debug monitor for SLAM system health and diagnostics"""
    
    def __init__(self):
        super().__init__('slam_debug_monitor')
        
        # Terminal display settings
        self.terminal_width = 80
        self.update_rate = 2.0  # Hz
        
        # Topic monitoring
        self.topics = {
            'cone_detection': TopicStatus(
                name='/fused_sorted_cones_ukf_sim',
                msg_type='TrackedConeArray',
                required=True,
                timeout=1.0,
                min_rate=10.0
            ),
            'odometry': TopicStatus(
                name='/odom_sim',
                msg_type='Odometry',
                required=True,
                timeout=0.5,
                min_rate=20.0
            ),
            'imu': TopicStatus(
                name='/ouster/imu_sim',
                msg_type='Imu',
                required=False,
                timeout=0.5,
                min_rate=50.0
            ),
            'gps': TopicStatus(
                name='/ublox_gps_node/fix_sim',
                msg_type='NavSatFix',
                required=False,
                timeout=2.0,
                min_rate=1.0
            ),
        }
        
        # SLAM internal status
        self.slam_status = {
            'node_alive': False,
            'last_keyframe_time': None,
            'keyframe_count': 0,
            'landmark_count': 0,
            'optimization_count': 0,
            'last_optimization_time': None,
            'processing_queue_size': 0,
        }
        
        # System status
        self.system_go = False
        self.errors = []
        self.warnings = []
        
        # Set up subscriptions with best effort QoS for monitoring
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Topic subscriptions
        self.cone_sub = self.create_subscription(
            TrackedConeArray,
            self.topics['cone_detection'].name,
            lambda msg: self._on_topic_received('cone_detection', msg),
            qos
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            self.topics['odometry'].name,
            lambda msg: self._on_topic_received('odometry', msg),
            qos
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            self.topics['imu'].name,
            lambda msg: self._on_topic_received('imu', msg),
            qos
        )
        
        self.gps_sub = self.create_subscription(
            NavSatFix,
            self.topics['gps'].name,
            lambda msg: self._on_topic_received('gps', msg),
            qos
        )
        
        # SLAM diagnostics subscription
        self.diag_sub = self.create_subscription(
            DiagnosticArray,
            '/diagnostics',
            self._on_diagnostics,
            10
        )
        
        # Display timer
        self.display_timer = self.create_timer(1.0 / self.update_rate, self._update_display)
        
        # Start time
        self.start_time = time.time()
        
        self.get_logger().info("SLAM Monitor Node started")
        
    def _on_topic_received(self, topic_key: str, msg):
        """Handle received topic message"""
        current_time = time.time()
        self.topics[topic_key].update(current_time)
        
    def _on_diagnostics(self, msg: DiagnosticArray):
        """Handle diagnostics from SLAM node"""
        for status in msg.status:
            if status.name == "cc_slam_node":
                self.slam_status['node_alive'] = True
                
                # Parse key-value pairs
                for kv in status.values:
                    if kv.key == "keyframe_count":
                        self.slam_status['keyframe_count'] = int(kv.value)
                    elif kv.key == "landmark_count":
                        self.slam_status['landmark_count'] = int(kv.value)
                    elif kv.key == "optimization_count":
                        self.slam_status['optimization_count'] = int(kv.value)
                    elif kv.key == "processing_queue_size":
                        self.slam_status['processing_queue_size'] = int(kv.value)
                        
    def _check_system_health(self) -> Tuple[bool, List[str], List[str]]:
        """Check overall system health"""
        errors = []
        warnings = []
        current_time = time.time()
        
        # Check required topics
        for key, topic in self.topics.items():
            if topic.required:
                healthy, reason = topic.is_healthy(current_time)
                if not healthy:
                    errors.append(f"{key}: {reason}")
            else:
                # Optional topics
                healthy, reason = topic.is_healthy(current_time)
                if not healthy and topic.received_count > 0:
                    warnings.append(f"{key}: {reason}")
                    
        # Check SLAM node
        if not self.slam_status['node_alive']:
            errors.append("SLAM node not responding")
            
        # Check processing queue
        if self.slam_status['processing_queue_size'] > 50:
            warnings.append(f"High queue size: {self.slam_status['processing_queue_size']}")
            
        go = len(errors) == 0
        return go, errors, warnings
        
    def _update_display(self):
        """Update terminal display"""
        # Clear screen (Unix/Linux)
        os.system('clear')
        
        # Check system health
        self.system_go, self.errors, self.warnings = self._check_system_health()
        
        # Header
        print("=" * self.terminal_width)
        print("CC-SLAM-SYM SYSTEM MONITOR".center(self.terminal_width))
        print("=" * self.terminal_width)
        
        # System status
        status_color = "\033[92m" if self.system_go else "\033[91m"  # Green if GO, Red if NOGO
        status_text = "GO" if self.system_go else "NOGO"
        reset_color = "\033[0m"
        
        print(f"\nSYSTEM STATUS: {status_color}{status_text}{reset_color}")
        print(f"Uptime: {time.time() - self.start_time:.1f}s")
        print()
        
        # Topic status
        print("INPUT TOPICS:")
        print("-" * self.terminal_width)
        print(f"{'Topic':<25} {'Rate (Hz)':<12} {'Count':<10} {'Status':<20}")
        print("-" * self.terminal_width)
        
        current_time = time.time()
        for key, topic in self.topics.items():
            rate = topic.get_rate()
            healthy, reason = topic.is_healthy(current_time)
            status_color = "\033[92m" if healthy else "\033[91m"
            
            print(f"{key:<25} {rate:>8.1f} Hz  {topic.received_count:>8}  "
                  f"{status_color}{reason:<20}{reset_color}")
                  
        # SLAM internal status
        print("\nSLAM STATUS:")
        print("-" * self.terminal_width)
        print(f"Keyframes: {self.slam_status['keyframe_count']}")
        print(f"Landmarks: {self.slam_status['landmark_count']}")
        print(f"Optimizations: {self.slam_status['optimization_count']}")
        print(f"Queue size: {self.slam_status['processing_queue_size']}")
        
        # Errors and warnings
        if self.errors:
            print(f"\n\033[91mERRORS ({len(self.errors)}):\033[0m")
            for error in self.errors[:5]:  # Show max 5 errors
                print(f"  • {error}")
                
        if self.warnings:
            print(f"\n\033[93mWARNINGS ({len(self.warnings)}):\033[0m")
            for warning in self.warnings[:5]:  # Show max 5 warnings
                print(f"  • {warning}")
                
        # Footer
        print("\n" + "=" * self.terminal_width)
        print("Press Ctrl+C to exit")


def main(args=None):
    rclpy.init(args=args)
    
    monitor = SlamDebugMonitor()
    
    # Use MultiThreadedExecutor for better performance
    executor = MultiThreadedExecutor()
    executor.add_node(monitor)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        print("\nDebug monitor shutdown requested")
    finally:
        monitor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()