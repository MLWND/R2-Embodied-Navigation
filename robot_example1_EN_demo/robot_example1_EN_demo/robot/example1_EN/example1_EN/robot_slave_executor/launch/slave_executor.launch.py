import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # 获取包路径
    pkg_path = get_package_share_directory('robot_slave_executor')
    
    # 定义URDF文件路径
    urdf_file = os.path.join(pkg_path, 'urdf', 'r2_urdf_v01.urdf')
    
    # 读取URDF文件
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()
    
    # RViz配置路径
    rviz_config = os.path.join(pkg_path, 'rviz', 'robot_remote_view.rviz')
    
    # 检查RViz配置是否存在，如果不存在则创建一个基本配置
    if not os.path.exists(rviz_config):
        print(f"Warning: RViz configuration not found at {rviz_config}")
        # 创建一个简单的配置作为备用
        basic_config = """Panels:
  - Class: rviz_common/Displays
    Name: Displays
  - Class: rviz_common/Selection
    Name: Selection
Visualization Manager:
  Displays:
    - Class: rviz_default_plugins/Grid
      Name: Grid
    - Class: rviz_default_plugins/RobotModel
      Name: RobotModel
      Description Topic:
        Value: /robot_description
  Enabled: true
  Global Options:
    Fixed Frame: base_link
  Name: root
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Name: Current View
      Target Frame: base_link
"""
        os.makedirs(os.path.dirname(rviz_config), exist_ok=True)
        with open(rviz_config, 'w') as f:
            f.write(basic_config)
        print(f"Created basic RViz configuration at: {rviz_config}")
    
    # 创建启动描述
    ld = LaunchDescription()
    
    # 添加robot_state_publisher节点
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'robot_description': robot_desc},
            {'use_sim_time': False}
        ]
    )
    ld.add_action(robot_state_publisher_node)
    
    # 添加从站执行器节点
    slave_executor_node = Node(
        package='robot_slave_executor',
        executable='slave_executor_node',
        name='slave_executor_node',
        output='screen'
    )
    ld.add_action(slave_executor_node)
    
    # 添加RViz节点 - 简化配置确保兼容性
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config, '-f', 'base_link']
    )
    ld.add_action(rviz_node)
    
    return ld
