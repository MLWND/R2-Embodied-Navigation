#!/usr/bin/env python3
"""
Master controller module
Responsible for coordinating action manager and user interface, handling system-level functions
"""

import os
import time
import json
import threading
import signal
import sys
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from sensor_msgs.msg import JointState

from .action_manager import ActionManager, SystemState


class ControllerState(Enum):
    """Controller state enumeration"""
    INIT = 0
    READY = 1
    CONNECTED = 2
    ERROR = 3
    STOPPED = 4

@dataclass
class SystemConfig:
    """System configuration parameters"""
    # Heartbeat configuration
    heartbeat_interval: float = 1.0
    connection_timeout: float = 10.0
    
    # Control configuration
    command_timeout: float = 5.0
    
    # Safety configuration
    enable_safety_checks: bool = True
    emergency_stop_timeout: float = 2.0
    
    # Performance monitoring
    monitor_interval: float = 0.5
    log_level: str = "INFO"

class MasterController(Node):
    """Master controller node"""
    
    def __init__(self):
        super().__init__('master_controller')
        
        # System configuration
        self.config = SystemConfig()
        
        # System state
        self.controller_state = ControllerState.INIT
        self.start_time = time.time()
        self.last_joint_state_time = 0
        self.slave_connected = False
        
        # Statistics information
        self.stats = {
            'commands_sent': 0,
            'actions_completed': 0,
            'errors': 0,
            'last_command_time': 0
        }
        
        # Thread safety locks
        self.state_lock = threading.RLock()
        self.stats_lock = threading.RLock()
        
        # Action manager
        self.action_manager = None
        
        # Create publishers and subscribers
        self._create_publishers()
        self._create_subscribers()
        
        # Create timers
        self._create_timers()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        # Start action manager
        self._start_action_manager()
        
        # Initialization complete
        self.controller_state = ControllerState.READY
        self.get_logger().info("Master controller initialization complete")
    
    def _create_publishers(self):
        """Create publishers"""
        # System control command publisher
        control_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        
        self.control_publisher = self.create_publisher(
            String,
            '/system_control_commands',
            control_qos)
        
        # System status publisher
        self.status_publisher = self.create_publisher(
            String,
            '/master_status',
            10)
    
    def _create_subscribers(self):
        """Create subscribers"""
        # Joint state subscriber (for monitoring slave status)
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10)
        
        # System log subscriber
        self.log_subscriber = self.create_subscription(
            String,
            '/slave_system_log',
            self._log_callback,
            10)
    
    def _create_timers(self):
        """Create timers"""
        # Monitoring timer
        self.monitor_timer = self.create_timer(
            self.config.monitor_interval,
            self._system_monitor)
        
        # Connection check timer
        self.connection_timer = self.create_timer(
            2.0,
            self._check_connection)
        
        # Status publishing timer
        self.status_timer = self.create_timer(
            1.0,
            self._publish_status)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers"""
        def signal_handler(signum, frame):
            self.get_logger().info(f"Received signal {signum}, starting safe shutdown")
            self.shutdown_system()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _start_action_manager(self):
        """Start action manager"""
        self.get_logger().info("Waiting for action manager node to start...")
        # Do not create instance directly, but call through ROS2 service
    
    def _joint_state_callback(self, msg: JointState):
        """Joint state callback"""
        with self.state_lock:
            self.last_joint_state_time = time.time()
            if not self.slave_connected:
                self.slave_connected = True
                self.controller_state = ControllerState.CONNECTED
                self.get_logger().info("Slave connected")
    
    def _log_callback(self, msg: String):
        """Log callback"""
        # Slave log information can be processed here
        self.get_logger().debug(f"Slave log: {msg.data}")
    
    def _system_monitor(self):
        """System monitoring"""
        current_time = time.time()
        
        # Check system runtime
        uptime = current_time - self.start_time
        if uptime > 0 and int(uptime) % 300 == 0:  # Log every 5 minutes
            self.get_logger().info(f"System runtime: {uptime/60:.1f} minutes")
        
        # Check command timeout
        with self.stats_lock:
            if (self.stats['last_command_time'] > 0 and 
                current_time - self.stats['last_command_time'] > self.config.command_timeout):
                self.get_logger().warn("Command execution timeout")
                self.stats['errors'] += 1
    
    def _check_connection(self):
        """Check connection status"""
        current_time = time.time()
        
        with self.state_lock:
            if (self.slave_connected and 
                current_time - self.last_joint_state_time > self.config.connection_timeout):
                self.slave_connected = False
                self.controller_state = ControllerState.READY
                self.get_logger().warn("Slave connection timeout")
    
    def _publish_status(self):
        """Publish system status"""
        status_data = {
            'controller_state': self.controller_state.name,
            'slave_connected': self.slave_connected,
            'uptime': time.time() - self.start_time,
            'stats': self.stats.copy(),
            'last_joint_state_time': self.last_joint_state_time
        }
        
        status_msg = String()
        status_msg.data = json.dumps(status_data)
        self.status_publisher.publish(status_msg)
    
    def send_control_command(self, command: str):
        """Send control command"""
        command_msg = String()
        command_msg.data = json.dumps({'command': command})
        self.control_publisher.publish(command_msg)
        self.get_logger().info(f"Sent control command: {command}")
        
        with self.stats_lock:
            self.stats['commands_sent'] += 1
            self.stats['last_command_time'] = time.time()
    
    def get_system_status(self) -> Dict:
        """Get system status"""
        with self.state_lock, self.stats_lock:
            return {
                'controller_state': self.controller_state.name,
                'slave_connected': self.slave_connected,
                'uptime': time.time() - self.start_time,
                'stats': self.stats.copy(),
                'last_joint_state_time': self.last_joint_state_time
            }
    
    def shutdown_system(self):
        """Safe system shutdown"""
        self.get_logger().info("Starting master controller shutdown")
        
        # Send stop command
        self.send_control_command("shutdown")
        
        # Stop action manager
        if self.action_manager:
            self.action_manager.stop_execution()
        
        # Update state
        with self.state_lock:
            self.controller_state = ControllerState.STOPPED
        
        # Wait for a while to ensure command is sent
        time.sleep(1.0)
        
        self.get_logger().info("Master controller shutdown complete")
 
def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    try:
        # Set environment variables
        os.environ['ROS_DOMAIN_ID'] = '42'
        
        controller = MasterController()
        
        # Start node
        rclpy.spin(controller)
        
    except KeyboardInterrupt:
        controller.get_logger().info("Master controller interrupted by user")
    except Exception as e:
        controller.get_logger().error(f"Master controller exception: {e}")
    finally:
        if 'controller' in locals():
            controller.shutdown_system()
            controller.destroy_node()
        rclpy.shutdown()
 
if __name__ == '__main__':
    main()
