#!/usr/bin/env python3
"""
Action manager module

"""

import os
import yaml
import math
import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import JointState

from .robot_actions import RobotActions, ActionType, JointTrajectory

class SystemState(Enum):
    """System state enumeration"""
    INIT = 0
    READY = 1
    EXECUTING = 2
    WAITING = 3
    ERROR = 4
    STOPPED = 5

class ActionManager(Node):
    """Action manager"""
    
    def __init__(self):
        super().__init__('action_manager')
        
        # System state
        self.system_state = SystemState.INIT
        self.execution_active = False
        self.execution_thread = None
        self.current_action = None
        
        # Robot actions
        self.robot_actions = RobotActions()
        
        # Action configuration
        self.action_config = {}
        
        # Create publishers
        self._create_publishers()
        
        # Create subscribers
        self._create_subscribers()
        
        # Load action configuration
        self._load_action_config()
        
        # Create timers
        self._create_timers()
        
        self.get_logger().info("Action manager initialization complete")
        self.system_state = SystemState.READY
    
    def _create_publishers(self):
        """Create publishers"""
        # Joint position command publisher
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        self.joint_command_publisher = self.create_publisher(
            Float64MultiArray,
            '/joint_position_commands',
            qos_profile)
        
        # System status publisher
        self.status_publisher = self.create_publisher(
            String,
            '/master_status',
            10)
    
    def _create_subscribers(self):
        """Create subscribers"""
        # Joint state subscriber (for monitoring)
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10)
    
    def _create_timers(self):
        """Create timers"""
        # Status publishing timer
        self.status_timer = self.create_timer(
            1.0,  # 1Hz
            self._publish_status)
    
    def _load_action_config(self):
        """Load action configuration"""
        try:
            # Get configuration file path
            import ament_index_python
            pkg_path = ament_index_python.packages.get_package_share_directory('master')
            config_path = os.path.join(
                pkg_path, 
                'config', 
                'robot_actions.yaml'
            )
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            self.action_config = config.get('actions', {})
            self.safety_config = config.get('safety_config', {})
            
            self.get_logger().info(f"Successfully loaded {len(self.action_config)} action configurations")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load action configuration: {e}")
            self.system_state = SystemState.ERROR
    
    def _joint_state_callback(self, msg: JointState):
        """Joint state callback"""
        # Joint state feedback can be processed here
        pass
    
    def _publish_status(self):
        """Publish system status"""
        status_data = {
            'system_state': self.system_state.name,
            'execution_active': self.execution_active,
            'current_action': self.current_action,
            'timestamp': time.time()
        }
        
        status_msg = String()
        status_msg.data = str(status_data)
        self.status_publisher.publish(status_msg)
    
    def get_available_actions(self) -> List[str]:
        """Get list of available actions"""
        return list(self.action_config.keys())
    
    def execute_action(self, action_name: str, repeat_count: int = 1) -> bool:
        """Execute action"""
        if action_name not in self.action_config:
            self.get_logger().error(f"Action not found: {action_name}")
            return False
        
        if self.execution_active:
            self.get_logger().warning("Action already in execution")
            return False
        
        # Get action type
        action_type = None
        for action in ActionType:
            if action.value == action_name:
                action_type = action
                break
        
        if not action_type:
            self.get_logger().error(f"Invalid action type: {action_name}")
            return False
        
        # Get action configuration
        config = self.action_config[action_name]
        
        # Create execution thread
        if self.execution_thread is None or not self.execution_thread.is_alive():
            self.execution_thread = threading.Thread(
                target=self._execute_action_thread,
                args=(action_type, config, repeat_count),
                daemon=True
            )
            self.execution_thread.start()
            return True
        else:
            self.get_logger().error("Execution thread still running")
            return False
    
    def stop_execution(self):
        """Stop current execution"""
        self.execution_active = False
        self.current_action = None
        self.get_logger().info("Stop action execution")
    
    def _execute_action_thread(self, action_type: ActionType, config: Dict, repeat_count: int):
        """Execute action thread"""
        self.execution_active = True
        self.system_state = SystemState.EXECUTING
        self.current_action = action_type.value
        
        self.get_logger().info(f"Start executing action: {config['name']}")
        
        try:
            for repeat in range(repeat_count):
                if not self.execution_active:
                    break
                
                self.get_logger().info(f"Executing {repeat + 1}/{repeat_count} times")
                
                # Start delay
                if config.get('start_delay', 0) > 0:
                    time.sleep(config['start_delay'])
                
                # Get action trajectories
                trajectories = self.robot_actions.get_action_trajectories(action_type)
                
                # Execute trajectories
                self._execute_trajectories(trajectories, config)
                
                # Inter-sequence buffer time
                if repeat < repeat_count - 1 and config.get('transition_time', 0) > 0:
                    self.get_logger().info(f"Inter-sequence buffer: {config['transition_time']}s")
                    time.sleep(config['transition_time'])
        
        except Exception as e:
            self.get_logger().error(f"Action execution exception: {e}")
        
        finally:
            self.execution_active = False
            self.current_action = None
            self.system_state = SystemState.READY
            self.get_logger().info(f"Action execution completed: {config['name']}")
    
    def _execute_trajectories(self, trajectories: List[JointTrajectory], config: Dict):
        """Execute trajectory sequence"""
        if not trajectories:
            return
        
        # Get interpolation parameters
        smooth_interpolation = config.get('smooth_interpolation', True)
        max_joint_velocity = config.get('max_joint_velocity', 2.0)
        
        for i, trajectory in enumerate(trajectories):
            if not self.execution_active:
                break
            
            if i == 0:
                # First trajectory point, directly set position
                self._publish_joint_positions(trajectory.positions)
                continue
            
            # Calculate time interval with previous trajectory point
            prev_trajectory = trajectories[i - 1]
            time_diff = trajectory.time_from_start - prev_trajectory.time_from_start
            
            if time_diff <= 0:
                continue
            
            # Interpolation execution
            start_time = time.time()
            while time.time() - start_time < time_diff:
                if not self.execution_active:
                    break
                
                # Calculate interpolation progress
                elapsed = time.time() - start_time
                progress = min(elapsed / time_diff, 1.0)
                
                # Calculate interpolated position
                if smooth_interpolation:
                    interpolated_positions = self._smooth_interpolate(
                        prev_trajectory.positions,
                        trajectory.positions,
                        progress
                    )
                else:
                    interpolated_positions = self._linear_interpolate(
                        prev_trajectory.positions,
                        trajectory.positions,
                        progress
                    )
                
                # Publish joint positions
                self._publish_joint_positions(interpolated_positions)
                
                # Control publishing frequency
                time.sleep(0.05)  # 20Hz
    
    def _linear_interpolate(self, start: List[float], end: List[float], progress: float) -> List[float]:
        """Linear interpolation"""
        return [s + (e - s) * progress for s, e in zip(start, end)]
    
    def _smooth_interpolate(self, start: List[float], end: List[float], progress: float) -> List[float]:
        """Smooth interpolation"""
        # Use smooth step function
        t = progress
        if t < 0.5:
            t2 = 4 * t * t * t
        else:
            t2 = 1 - math.pow(-2 * t + 2, 3) / 2
        return [s + (e - s) * t2 for s, e in zip(start, end)]
    
    def _publish_joint_positions(self, positions: List[float]):
        """Publish joint position commands"""
        msg = Float64MultiArray()
        msg.data = positions
        self.joint_command_publisher.publish(msg)
 
def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    try:
        action_manager = ActionManager()
        rclpy.spin(action_manager)
    except KeyboardInterrupt:
        action_manager.get_logger().info("Action manager interrupted by user")
    except Exception as e:
        action_manager.get_logger().error(f"Action manager exception: {e}")
    finally:
        action_manager.destroy_node()
        rclpy.shutdown()
 
if __name__ == '__main__':
    main()
