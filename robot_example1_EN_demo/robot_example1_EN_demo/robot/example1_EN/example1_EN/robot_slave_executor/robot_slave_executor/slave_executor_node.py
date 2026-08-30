#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
import time
import json

class SlaveExecutorNode(Node):
    def __init__(self):
        super().__init__('slave_executor_node')
        
        # Subscribe to joint angle commands
        self.joint_command_subscription = self.create_subscription(
            Float64MultiArray,
            '/joint_position_commands',
            self.joint_command_callback,
            10)
        
        # Publish status information
        self.status_publisher = self.create_publisher(
            String,
            '/slave_status',
            10)
        
        # Publish joint states - This is necessary for RViz to display robot pose
        self.joint_publisher = self.create_publisher(
            JointState,
            '/joint_states',
            10)
        
        # Define all joint names - Must completely match joint names in URDF file
        self.joint_names = [
            'wheel_left_joint', 'wheel_right_joint', 'lift_joint', 'waist_joint', 'head_joint',
            'J1_left_joint', 'J2_left_joint', 'J3_left_joint', 'J4_left_joint', 'J5_left_joint', 
            'J6_left_joint', 'J7_left_joint',
            'J1_right_joint', 'J2_right_joint', 'J3_right_joint', 'J4_right_joint', 'J5_right_joint', 
            'J6_right_joint', 'J7_right_joint'
        ]
        
        # Current joint positions
        self.current_positions = [0.0] * len(self.joint_names)
        
        # Last received command time
        self.last_command_time = time.time()
        
        # Periodically publish joint states (20Hz) - This is critical to ensure RViz can continuously receive joint states
        self.timer = self.create_timer(0.05, self.timer_callback)  # 20Hz
        
        self.get_logger().info('Slave executor node launched - Zero configuration execution mode')
        self.get_logger().info(f'Monitor {len(self.joint_names)} joints')
        self.publish_status("Ready")
    
    def joint_command_callback(self, msg):
        """Receive joint angle commands and update current positions"""
        if len(msg.data) == len(self.joint_names):
            self.current_positions = list(msg.data)  # Convert to list to ensure correct type
            self.last_command_time = time.time()
            self.publish_status("Executing")
            self.get_logger().debug(f'Received joint command: {self.current_positions[:3]}...')  # Only print first 3 values
        else:
            self.get_logger().warn(f'Received joint position count does not match: expected{len(self.joint_names)}, actual{len(msg.data)}')
            self.publish_status("Error", "Joint position count does not match")
    
    def publish_status(self, status="Ready", error=None):
        """Publish status information"""
        status_data = {
            'status': status,
            'timestamp': time.time(),
            'last_command_time': self.last_command_time,
            'error': error,
            'joint_count': len(self.joint_names)
        }
        
        status_msg = String()
        status_msg.data = json.dumps(status_data)
        self.status_publisher.publish(status_msg)
    
    def timer_callback(self):
        """Periodically publish joint states - This is necessary for RViz to display robot pose"""
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.header.frame_id = "base_link"  # Set reference coordinate system
        joint_msg.name = self.joint_names
        joint_msg.position = self.current_positions
        
        # Publish joint states
        self.joint_publisher.publish(joint_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SlaveExecutorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.publish_status("Manually stopped")
        node.get_logger().info('Slave executor node shutdown')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()