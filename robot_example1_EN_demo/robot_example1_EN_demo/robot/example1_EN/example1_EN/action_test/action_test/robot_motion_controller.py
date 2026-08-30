#!/usr/bin/env python3
"""
Robot motion controller
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Float64MultiArray
import math
import time
import json
import os

class RobotMotionController(Node):
    def __init__(self):
        super().__init__('robot_motion_controller')
        
        # Subscribe to control commands
        self.control_subscription = self.create_subscription(
            String,
            '/robot_control_commands',
            self.control_callback,
            10)
        
        # Publish joint states
        self.joint_publisher = self.create_publisher(
            JointState,
            '/joint_states',
            10)
        
        # Publish status feedback
        self.status_publisher = self.create_publisher(
            String,
            '/robot_status',
            10)
        
        # Define joint names
        self.joint_names = [
            'wheel_left_joint', 'wheel_right_joint', 'lift_joint', 'waist_joint', 'head_joint',
            'J1_left_joint', 'J2_left_joint', 'J3_left_joint', 'J4_left_joint', 'J5_left_joint', 
            'J6_left_joint', 'J7_left_joint',
            'J1_right_joint', 'J2_right_joint', 'J3_right_joint', 'J4_right_joint', 'J5_right_joint', 
            'J6_right_joint', 'J7_right_joint'
        ]
        
        # Define motion range for each joint [min, max]
        self.joint_limits = {
            'wheel_left_joint': [-3.14, 3.14],
            'wheel_right_joint': [-3.14, 3.14],
            'lift_joint': [0.0, 0.6],
            'waist_joint': [0.0, 1.57],
            'head_joint': [0.0, 0.87],
            'J1_left_joint': [-3.05, 3.05],
            'J2_left_joint': [0.0, 3.14],
            'J3_left_joint': [-2.7, 2.7],
            'J4_left_joint': [-1.65, 1.65],
            'J5_left_joint': [-2.6, 2.6],
            'J6_left_joint': [-1.66, 1.66],
            'J7_left_joint': [-3.05, 3.05],
            'J1_right_joint': [-3.05, 3.05],
            'J2_right_joint': [-3.14, 0.0],
            'J3_right_joint': [-2.7, 2.7],
            'J4_right_joint': [-1.65, 1.65],
            'J5_right_joint': [-2.6, 2.6],
            'J6_right_joint': [-1.66, 1.66],
            'J7_right_joint': [-3.05, 3.05]
        }
        
        # Initial joint positions
        self.initial_positions = [0.0] * len(self.joint_names)
        self.current_positions = self.initial_positions.copy()
        self.target_positions = self.initial_positions.copy()
        
        # Action control state
        self.is_active = False
        self.current_action = None
        self.action_start_time = 0
        self.action_duration = 0
        self.last_heartbeat_time = time.time()
        self.connection_timeout = 5.0  # 5 seconds without heartbeat then stop
        
        # Action library
        self.action_library = ActionLibrary()
        
        # Smooth motion parameters
        self.max_velocity = 0.5  # Maximum angular velocity (rad/s)
        self.max_acceleration = 0.5  # Maximum angular acceleration (rad/s²)
        self.previous_positions = self.initial_positions.copy()
        self.previous_time = time.time()
        
        # Timer for publishing joint states and checking connection status
        self.timer = self.create_timer(0.05, self.timer_callback)  # 20Hz
        
        self.get_logger().info('Robot motion controller launched, waiting for commands...')
        self.get_logger().info(f'Control {len(self.joint_names)} joints')
    
    def control_callback(self, msg):
        """Process control commands from host"""
        try:
            command = json.loads(msg.data)
            cmd_type = command.get('type', '')
            
            # Update heartbeat time
            self.last_heartbeat_time = time.time()
            
            if cmd_type == 'start':
                action_name = command.get('action', 'wave')
                self.start_action(action_name)
            elif cmd_type == 'stop':
                self.stop_action()
            elif cmd_type == 'heartbeat':
                # Only update heartbeat time
                pass
            else:
                self.get_logger().warn(f'Unknown command type: {cmd_type}')
                
        except json.JSONDecodeError:
            self.get_logger().error(f'Cannot parse command: {msg.data}')
        except Exception as e:
            self.get_logger().error(f'Error processing command: {str(e)}')
    
    def start_action(self, action_name):
        """Start executing specified action"""
        if action_name in self.action_library.get_actions():
            self.is_active = True
            self.current_action = action_name
            self.action_start_time = time.time()
            self.action_duration = self.action_library.get_action_duration(action_name)
            
            # Get action sequence
            self.action_sequence = self.action_library.get_action_sequence(action_name)
            self.sequence_step = 0
            self.sequence_start_time = time.time()
            
            self.get_logger().info(f'Start executing action: {action_name}')
            self.publish_status(f'Executing action: {action_name}')
        else:
            self.get_logger().warn(f'Unknown action: {action_name}')
            self.publish_status(f'Error: Unknown action {action_name}')
    
    def stop_action(self):
        """Stop current action"""
        self.is_active = False
        self.current_action = None
        self.target_positions = self.initial_positions.copy()
        
        self.get_logger().info('Stop action, return to initial pose')
        self.publish_status('Stop action, return to initial pose')
    
    def check_connection(self):
        """Check connection status, stop action if timeout"""
        current_time = time.time()
        if self.is_active and (current_time - self.last_heartbeat_time > self.connection_timeout):
            self.get_logger().warn('Connection timeout, stop action')
            self.stop_action()
            self.publish_status('Connection timeout, stop action')
    
    def smooth_interpolate(self, current_time):
        """Use velocity and acceleration limits for smooth interpolation"""
        dt = current_time - self.previous_time
        if dt <= 0:
            return
        
        # Calculate distance from current position to target position
        delta_pos = [t - c for t, c in zip(self.target_positions, self.current_positions)]
        
        # Calculate maximum allowed velocity change
        max_delta_vel = self.max_acceleration * dt
        
        # Calculate current velocity (based on previous frame position change)
        current_vel = [(c - p) / dt if dt > 0 else 0.0 
                       for c, p in zip(self.current_positions, self.previous_positions)]
        
        # Calculate new target velocity, considering acceleration limits
        target_vel = []
        for i, delta in enumerate(delta_pos):
            # Ideal velocity is to directly reach target position
            ideal_vel = delta / dt if dt > 0 else 0.0
            
            # Limit velocity change rate (acceleration)
            vel_change = ideal_vel - current_vel[i]
            if abs(vel_change) > max_delta_vel:
                vel_change = max_delta_vel if vel_change > 0 else -max_delta_vel
            
            new_vel = current_vel[i] + vel_change
            
            # Limit maximum velocity
            new_vel = max(-self.max_velocity, min(self.max_velocity, new_vel))
            target_vel.append(new_vel)
        
        # Calculate new positions
        new_positions = []
        for i, vel in enumerate(target_vel):
            new_pos = self.current_positions[i] + vel * dt
            
            # Ensure not exceeding target position
            if delta_pos[i] > 0:  # Need to move forward
                new_pos = min(new_pos, self.target_positions[i])
            else:  # Need to move backward
                new_pos = max(new_pos, self.target_positions[i])
            
            new_positions.append(new_pos)
        
        # Update state
        self.previous_positions = self.current_positions.copy()
        self.current_positions = new_positions
        self.previous_time = current_time
    
    def execute_action_sequence(self):
        """Execute action sequence"""
        if not self.is_active or not self.action_sequence:
            return
        
        current_time = time.time()
        elapsed_time = current_time - self.sequence_start_time
        
        # Check if current step is completed
        if self.sequence_step < len(self.action_sequence):
            current_step = self.action_sequence[self.sequence_step]
            step_duration = current_step['duration']
            
            if elapsed_time >= step_duration:
                # Enter next step
                self.sequence_step += 1
                self.sequence_start_time = current_time
                
                if self.sequence_step < len(self.action_sequence):
                    # Set new target position
                    next_step = self.action_sequence[self.sequence_step]
                    self.target_positions = next_step['positions'].copy()
                    self.get_logger().info(f'Action step: {next_step["name"]}')
                else:
                    # Action sequence completed
                    self.get_logger().info(f'Action {self.current_action} completed')
                    self.publish_status(f'Action {self.current_action} completed')
                    
                    # If it's a loop action, restart
                    if self.action_library.is_loop_action(self.current_action):
                        self.sequence_step = 0
                        self.sequence_start_time = current_time
                        first_step = self.action_sequence[0]
                        self.target_positions = first_step['positions'].copy()
                        self.get_logger().info(f'Restart loop action: {self.current_action}')
                    else:
                        # Non-loop action, stop
                        self.stop_action()
        else:
            # Sequence completed, decide whether to loop based on action type
            if self.action_library.is_loop_action(self.current_action):
                self.sequence_step = 0
                self.sequence_start_time = current_time
                first_step = self.action_sequence[0]
                self.target_positions = first_step['positions'].copy()
            else:
                self.stop_action()
    
    def publish_status(self, status):
        """Publish status information"""
        status_msg = String()
        status_msg.data = status
        self.status_publisher.publish(status_msg)
    
    def timer_callback(self):
        """Timer callback, update action and publish joint states"""
        current_time = time.time()
        
        # Check connection status
        self.check_connection()
        
        # Execute action sequence
        if self.is_active:
            self.execute_action_sequence()
        
        # Smooth interpolate to target position
        self.smooth_interpolate(current_time)
        
        # Publish joint states
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = self.joint_names
        joint_msg.position = self.current_positions
        self.joint_publisher.publish(joint_msg)

class ActionLibrary:
    """Action library, contains all predefined action sequences"""
    
    def __init__(self):
        self.actions = {}
        self.define_actions()
    
    def define_actions(self):
        """Define all action sequences"""
        # Wave action
        self.actions['wave'] = {
            'duration': 10.0,
            'loop': True,
            'sequence': [
                {
                    'name': 'Initial pose',
                    'duration': 1.0,
                    'positions': [0.0] * 19  # All joints zero
                },
                {
                    'name': 'Prepare to raise arm',
                    'duration': 1.0,
                    'positions': self._create_positions({
                        'J1_left_joint': 0.3,
                        'J1_right_joint': 0.3
                    })
                },
                {
                    'name': 'Raise large arm',
                    'duration': 1.5,
                    'positions': self._create_positions({
                        'J1_left_joint': 1.2,
                        'J1_right_joint': 1.2,
                        'J2_left_joint': 1.0,
                        'J2_right_joint': -1.0
                    })
                },
                {
                    'name': 'Fully raise both arms',
                    'duration': 1.0,
                    'positions': self._create_positions({
                        'J1_left_joint': 1.2,
                        'J1_right_joint': 1.2,
                        'J2_left_joint': 1.2,
                        'J2_right_joint': -1.2,
                        'J3_left_joint': -0.8,
                        'J3_right_joint': -0.8
                    })
                },
                {
                    'name': 'Left arm wave',
                    'duration': 1.0,
                    'positions': self._create_positions({
                        'J1_left_joint': 1.2,
                        'J1_right_joint': 1.2,
                        'J2_left_joint': 1.2,
                        'J2_right_joint': -1.2,
                        'J3_left_joint': -0.8,
                        'J3_right_joint': -0.8,
                        'J4_left_joint': 0.8
                    })
                },
                {
                    'name': 'Right arm wave',
                    'duration': 1.0,
                    'positions': self._create_positions({
                        'J1_left_joint': 1.2,
                        'J1_right_joint': 1.2,
                        'J2_left_joint': 1.2,
                        'J2_right_joint': -1.2,
                        'J3_left_joint': -0.8,
                        'J3_right_joint': -0.8,
                        'J4_right_joint': 0.8
                    })
                },
                {
                    'name': 'Lower arms',
                    'duration': 2.0,
                    'positions': [0.0] * 19  # All joints zero
                }
            ]
        }
        
        # Nod action
        self.actions['nod'] = {
            'duration': 5.0,
            'loop': True,
            'sequence': [
                {
                    'name': 'Initial pose',
                    'duration': 1.0,
                    'positions': [0.0] * 19
                },
                {
                    'name': 'Nod',
                    'duration': 1.0,
                    'positions': self._create_positions({
                        'head_joint': 0.5
                    })
                },
                {
                    'name': 'Return to middle',
                    'duration': 1.0,
                    'positions': [0.0] * 19
                },
                {
                    'name': 'Nod again',
                    'duration': 1.0,
                    'positions': self._create_positions({
                        'head_joint': 0.5
                    })
                },
                {
                    'name': 'Return to initial',
                    'duration': 1.0,
                    'positions': [0.0] * 19
                }
            ]
        }
        
        # Turn action
        self.actions['turn'] = {
            'duration': 8.0,
            'loop': True,
            'sequence': [
                {
                    'name': 'Initial pose',
                    'duration': 1.0,
                    'positions': [0.0] * 19
                },
                {
                    'name': 'Turn left',
                    'duration': 2.0,
                    'positions': self._create_positions({
                        'waist_joint': 1.0
                    })
                },
                {
                    'name': 'Return to middle',
                    'duration': 2.0,
                    'positions': [0.0] * 19
                },
                {
                    'name': 'Turn right',
                    'duration': 2.0,
                    'positions': self._create_positions({
                        'waist_joint': -1.0
                    })
                },
                {
                    'name': 'Return to initial',
                    'duration': 1.0,
                    'positions': [0.0] * 19
                }
            ]
        }
    
    def _create_positions(self, joint_values):
        """Create complete position array based on joint value dictionary"""
        # Joint name mapping to index
        joint_to_index = {
            'wheel_left_joint': 0,
            'wheel_right_joint': 1,
            'lift_joint': 2,
            'waist_joint': 3,
            'head_joint': 4,
            'J1_left_joint': 5,
            'J2_left_joint': 6,
            'J3_left_joint': 7,
            'J4_left_joint': 8,
            'J5_left_joint': 9,
            'J6_left_joint': 10,
            'J7_left_joint': 11,
            'J1_right_joint': 12,
            'J2_right_joint': 13,
            'J3_right_joint': 14,
            'J4_right_joint': 15,
            'J5_right_joint': 16,
            'J6_right_joint': 17,
            'J7_right_joint': 18
        }
        
        positions = [0.0] * 19
        for joint_name, value in joint_values.items():
            if joint_name in joint_to_index:
                positions[joint_to_index[joint_name]] = value
        
        return positions
    
    def get_actions(self):
        """Get names of all available actions"""
        return list(self.actions.keys())
    
    def get_action_sequence(self, action_name):
        """Get sequence of specified action"""
        if action_name in self.actions:
            return self.actions[action_name]['sequence']
        return []
    
    def get_action_duration(self, action_name):
        """Get total duration of specified action"""
        if action_name in self.actions:
            return self.actions[action_name]['duration']
        return 0.0
    
    def is_loop_action(self, action_name):
        """Check if action loops"""
        if action_name in self.actions:
            return self.actions[action_name]['loop']
        return False

def main(args=None):
    rclpy.init(args=args)
    node = RobotMotionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()