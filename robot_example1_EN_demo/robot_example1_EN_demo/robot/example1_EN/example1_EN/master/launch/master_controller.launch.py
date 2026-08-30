import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 获取包路径
    pkg_path = get_package_share_directory('master')
    
    # 配置文件路径
    config_file = os.path.join(pkg_path, 'config', 'master_config.yaml')
    
    # 创建启动描述
    ld = LaunchDescription()
    
    # 添加主控制器节点
    master_controller_node = Node(
        package='master',
        executable='master_controller',
        name='master_controller',
        output='screen',
        parameters=[config_file] if os.path.exists(config_file) else []
    )
    ld.add_action(master_controller_node)
    
    # 添加动作管理器节点
    action_manager_node = Node(
        package='master',
        executable='action_manager',
        name='action_manager',
        output='screen',
        parameters=[config_file] if os.path.exists(config_file) else []
    )
    ld.add_action(action_manager_node)
    
    # 添加用户界面节点
    user_interface_node = Node(
        package='master',
        executable='user_interface',
        name='user_interface',
        output='screen',
        parameters=[config_file] if os.path.exists(config_file) else [],
        emulate_tty=True  # 支持终端输入
    )
    ld.add_action(user_interface_node)
    
    return ld
