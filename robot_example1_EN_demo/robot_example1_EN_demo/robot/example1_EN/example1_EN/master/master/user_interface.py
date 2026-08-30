#!/usr/bin/env python3
"""
User interface module
Provides CLI interface for robot control
"""

import os
import sys
import time
import threading
import signal
from typing import Dict, List, Optional
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from sensor_msgs.msg import JointState

from .action_manager import ActionManager, SystemState

class UIState(Enum):
    """Interface state enumeration"""
    MAIN_MENU = 0
    ACTION_SELECTION = 1
    EXECUTION_CONTROL = 2
    SYSTEM_STATUS = 3
    SETTINGS = 4

class UserInterface(Node):
    """User interaction interface"""
    
    def __init__(self):
        super().__init__('user_interface')
        
        # Interface state
        self.ui_state = UIState.MAIN_MENU
        self.running = True
        self.current_action = None
        self.execution_active = False
        
        # Action manager
        self.action_manager = None
        
        # System state
        self.last_joint_state = None
        self.connection_status = "Not connected"
        self.last_heartbeat_time = 0
        
        # Create subscribers
        self._create_subscribers()
        
        # Create publishers
        self._create_publishers()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        # Start action manager
        self._start_action_manager()
        
        self.get_logger().info("User interface initialization complete")
    
    def _create_subscribers(self):
        """Create subscribers"""
        # Joint state subscription
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10)
        
        # Master status subscription
        self.status_subscriber = self.create_subscription(
            String,
            '/master_status',
            self._status_callback,
            10)
    
    def _create_publishers(self):
        """Create publishers"""
        # System control command publisher
        self.control_publisher = self.create_publisher(
            String,
            '/system_control_commands',
            10)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers"""
        def signal_handler(signum, frame):
            self.get_logger().info(f"Received signal {signum}, starting safe shutdown")
            self.running = False
            if self.action_manager:
                self.action_manager.stop_execution()
            self._send_control_command("shutdown")
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _start_action_manager(self):
        """Start action manager"""
        try:
            self.action_manager = ActionManager()
            self.get_logger().info("Action manager started successfully")
        except Exception as e:
            self.get_logger().error(f"Action manager startup failed: {e}")
    
    def _joint_state_callback(self, msg: JointState):
        """Joint state callback"""
        self.last_joint_state = msg
        self.last_heartbeat_time = time.time()
        if self.connection_status == "Not connected":
            self.connection_status = "Connected"
    
    def _status_callback(self, msg: String):
        """Status callback"""
        # Status information can be processed here
        pass
    
    def _send_control_command(self, command: str):
        """Send control command"""
        msg = String()
        msg.data = f'{{"command": "{command}"}}'
        self.control_publisher.publish(msg)
        self.get_logger().info(f"Sent control command: {command}")
    
    def _check_connection(self):
        """Check connection status"""
        if self.last_heartbeat_time == 0:
            return False
        
        # If no joint state received for more than 5 seconds, consider connection lost
        if time.time() - self.last_heartbeat_time > 5.0:
            self.connection_status = "Connection lost"
            return False
        
        return True
    
    def _clear_screen(self):
        """Clear screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def _print_header(self):
        """Print header"""
        self._clear_screen()
        print("=" * 60)
        print("        Distributed Robot Control System - Master")
        print("=" * 60)
        print(f"Connection status: {self.connection_status}")
        print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)
    
    def _print_main_menu(self):
        """Print main menu"""
        self._print_header()
        print("Main Menu:")
        print("1. Action Control")
        print("2. System Status")
        print("3. System Settings")
        print("4. Help Information")
        print("0. Exit System")
        print("-" * 60)
        print("Please select operation (0-4): ", end='')
    
    def _print_action_menu(self):
        """Print action menu"""
        self._print_header()
        print("Action Control:")
        
        if self.action_manager:
            actions = self.action_manager.get_available_actions()
            for i, action in enumerate(actions, 1):
                print(f"{i}. {action}")
            
            if self.execution_active:
                print("s. Stop Execution")
            else:
                print("s. Start Execution")
        
        print("b. Return to Main Menu")
        print("-" * 60)
        
        if self.current_action:
            print(f"Current selection: {self.current_action}")
        
        print("Please select operation: ", end='')
    
    def _print_status_menu(self):
        """Print status menu"""
        self._print_header()
        print("System Status:")
        print("-" * 60)
        
        if self.last_joint_state:
            print(f"Joint count: {len(self.last_joint_state.name)}")
            print("\nJoint Status:")
            for i, (name, position) in enumerate(zip(
                self.last_joint_state.name,
                self.last_joint_state.position)):
                if i < 10:  # Only display first 10 joints
                    print(f"  {name}: {position:.3f} rad")
                elif i == 10:
                    print(f"  ... {len(self.last_joint_state.name) - 10} more joints")
        else:
            print("No joint state information received")
        
        if self.action_manager:
            print(f"\nAction manager status: {self.action_manager.system_state.name}")
            print(f"Execution status: {'Executing' if self.action_manager.execution_active else 'Idle'}")
            if self.action_manager.current_action:
                print(f"Current action: {self.action_manager.current_action}")
        
        print("-" * 60)
        print("Press Enter to return to main menu: ", end='')
    
    def _print_settings_menu(self):
        """Print settings menu"""
        self._print_header()
        print("System Settings:")
        print("1. Send Stop Command")
        print("2. Test Connection")
        print("b. Return to Main Menu")
        print("-" * 60)
        print("Please select operation: ", end='')
    
    def _print_help_menu(self):
        """Print help menu"""
        self._print_header()
        print("Help Information:")
        print("-" * 60)
        print("Action Description:")
        print("1. wave - Wave action: Robot raises right arm and waves left and right for 3 seconds")
        print("2. expand - Expand action: Robot extends both arms outward to shoulder level, repeats in loop")
        print("3. agree - Agree action: Robot body slightly rises, then nods 3 times")
        print()
        print("Usage Instructions:")
        print("- Select action, then press 's' to start execution")
        print("- Press 's' during execution to stop")
        print("- System automatically handles smooth transitions and safety checks")
        print()
        print("Precautions:")
        print("- Ensure slave node is running normally")
        print("- Check robot surroundings before execution")
        print("- Use stop command in emergency situations")
        print("-" * 60)
        print("Press Enter to return to main menu: ", end='')
    
    def _handle_main_menu(self, choice: str):
        """Handle main menu selection"""
        if choice == '0':
            self.running = False
        elif choice == '1':
            self.ui_state = UIState.ACTION_SELECTION
        elif choice == '2':
            self.ui_state = UIState.SYSTEM_STATUS
        elif choice == '3':
            self.ui_state = UIState.SETTINGS
        elif choice == '4':
            self._print_help_menu()
            input()
            self.ui_state = UIState.MAIN_MENU
        else:
            print("Invalid selection, please try again")
            time.sleep(1)
    
    def _handle_action_menu(self, choice: str):
        """Handle action menu selection"""
        if choice == 'b':
            self.ui_state = UIState.MAIN_MENU
            self.current_action = None
        elif choice == 's':
            if self.action_manager:
                if self.execution_active:
                    self.action_manager.stop_execution()
                    self.execution_active = False
                    print("Stop execution")
                else:
                    if self.current_action:
                        success = self.action_manager.execute_action(self.current_action)
                        if success:
                            self.execution_active = True
                            print("Start execution")
                        else:
                            print("Execution failed")
                    else:
                        print("Please select action first")
                time.sleep(1)
        else:
            # Handle numeric selection
            try:
                action_index = int(choice) - 1
                if self.action_manager:
                    actions = self.action_manager.get_available_actions()
                    if 0 <= action_index < len(actions):
                        self.current_action = actions[action_index]
                        print(f"Selected action: {self.current_action}")
                    else:
                        print("Invalid selection")
                    time.sleep(1)
            except ValueError:
                print("Invalid selection")
                time.sleep(1)
    
    def _handle_status_menu(self, choice: str):
        """Handle status menu selection"""
        if choice == '':
            self.ui_state = UIState.MAIN_MENU
    
    def _handle_settings_menu(self, choice: str):
        """Handle settings menu selection"""
        if choice == 'b':
            self.ui_state = UIState.MAIN_MENU
        elif choice == '1':
            self._send_control_command("stop")
            print("Stop command sent")
            time.sleep(1)
        elif choice == '2':
            if self._check_connection():
                print("Connection normal")
            else:
                print("Connection abnormal")
            time.sleep(1)
    
    def _update_execution_status(self):
        """Update execution status"""
        if self.action_manager and self.execution_active:
            if not self.action_manager.execution_active:
                self.execution_active = False
                print("Action execution completed")
    
    def run(self):
        """Run user interface"""
        # Create multi-threaded executor
        executor = MultiThreadedExecutor()
        executor.add_node(self)
        
        if self.action_manager:
            executor.add_node(self.action_manager)
        
        # Start ROS thread
        ros_thread = threading.Thread(target=executor.spin, daemon=True)
        ros_thread.start()
        
        try:
            while self.running and rclpy.ok():
                # Check connection status
                self._check_connection()
                
                # Update execution status
                self._update_execution_status()
                
                # Display corresponding menu
                if self.ui_state == UIState.MAIN_MENU:
                    self._print_main_menu()
                    choice = input().strip()
                    self._handle_main_menu(choice)
                
                elif self.ui_state == UIState.ACTION_SELECTION:
                    self._print_action_menu()
                    choice = input().strip()
                    self._handle_action_menu(choice)
                
                elif self.ui_state == UIState.SYSTEM_STATUS:
                    self._print_status_menu()
                    choice = input().strip()
                    self._handle_status_menu(choice)
                
                elif self.ui_state == UIState.SETTINGS:
                    self._print_settings_menu()
                    choice = input().strip()
                    self._handle_settings_menu(choice)
        
        except KeyboardInterrupt:
            self.get_logger().info("User interrupt")
        except Exception as e:
            self.get_logger().error(f"Interface exception: {e}")
        finally:
            # Clean up resources
            if self.action_manager:
                self.action_manager.stop_execution()
                self.action_manager.destroy_node()
            self.destroy_node()
    
    def shutdown(self):
        """Shutdown interface"""
        self.running = False
        if self.action_manager:
            self.action_manager.stop_execution()

def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    try:
        ui = UserInterface()
        ui.run()
    except Exception as e:
        print(f"User interface exception: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
