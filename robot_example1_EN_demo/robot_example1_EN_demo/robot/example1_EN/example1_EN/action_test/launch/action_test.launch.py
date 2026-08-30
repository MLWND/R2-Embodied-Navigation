import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    
    # 获取包路径
    pkg_path = get_package_share_directory('action_test')
    
    # 定义URDF文件路径
    urdf_file = os.path.join(pkg_path, 'urdf', 'r2_urdf_v01.urdf')
    
    # 读取URDF文件
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()
    
    # RViz配置路径
    rviz_config = os.path.join(pkg_path, 'rviz', 'robot_remote_view.rviz')
    
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
    
    # 添加joint_state_publisher_gui节点
    joint_state_publisher_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen'
    )
    ld.add_action(joint_state_publisher_node)
    
    # 添加机器人运动控制器节点
    motion_controller = Node(
        package='action_test',
        executable='robot_motion_controller',
        name='robot_motion_controller',
        output='screen'
    )
    ld.add_action(motion_controller)
    
    # 添加RViz节点
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config]
    )
    ld.add_action(rviz_node)
    
    return ld