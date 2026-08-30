#!/bin/bash

# 分布式机器人控制系统 - 主控端启动脚本

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查ROS2环境
check_ros2() {
    if [ -z "$ROS_DISTRO" ]; then
        print_error "ROS2环境未设置,请先运行: source /opt/ros/humble/setup.bash"
        exit 1
    fi
    print_info "ROS2环境检查通过: $ROS_DISTRO"
}

# 检查工作空间
check_workspace() {
    if [ ! -f "install/setup.bash" ]; then
        print_error "工作空间未构建，请先运行: colcon build"
        exit 1
    fi
    print_info "工作空间检查通过"
}

# 设置网络配置
setup_network() {
    export ROS_DOMAIN_ID=42
    export ROS_LOCALHOST_ONLY=0
    print_info "网络配置设置完成: ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
}

# 启动主控系统
start_master() {
    print_info "启动分布式机器人控制系统主控端..."
    
    # 设置工作空间环境
    source install/setup.bash
    
    # 启动主控制器
    # 启动用户界面节点（包含动作管理器）
    ros2 run master user_interface

}

# 主函数
main() {
    print_info "分布式机器人控制系统 - 主控端启动脚本"
    print_info "======================================"
    
    # 检查环境
    check_ros2
    check_workspace
    setup_network
    
    # 启动系统
    start_master
}

# 信号处理
trap 'print_warning "接收到中断信号，正在关闭..."; exit 0' INT TERM

# 执行主函数
main "$@"
