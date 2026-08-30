#!/bin/bash
WORKSPACE_PATH="$HOME/test_demo_script"
init_path(){
    # 声明工作空间
    if [ -z "$WORKSPACE_PATH" ]; then
        read -p "无法找到工作空间，请手动输入绝对路径: " WORKSPACE_PATH
        if [ ! -d "$WORKSPACE_PATH/install" ]; then
            echo "错误：工作空间路径无效！ $WORKSPACE_PATH/install 不存在"
            exit 1
        fi
    else
        echo "已检测到工作空间: $WORKSPACE_PATH"
    fi
    # 声明环境变量
    for version in humble foxy galactic rolling; do
        if [ -f "/opt/ros/$version/setup.bash" ]; then
            source "/opt/ros/$version/setup.bash"
            echo "Sourced ROS2 $version 环境"
            ros2_ver=$version
            break
        fi
    done
    if [ -z "$ros2_ver" ]; then
        echo "错误：未找到任何ROS2安装！"
        exit 1
    fi
    # source 工作空间环境
    if [ -f "$WORKSPACE_PATH/install/setup.bash" ]; then
        source "$WORKSPACE_PATH/install/setup.bash"
        echo "Sourced 工作空间环境"
    else
        echo "错误：未找到工作空间环境文件 $WORKSPACE_PATH/install/setup.bash"
        echo "请确认工作空间是否已构建 (colcon build)"
        exit 1
    fi
}

init_path
# 启动程序
python3 test_demo1.py
# python3 test_demo2.py
