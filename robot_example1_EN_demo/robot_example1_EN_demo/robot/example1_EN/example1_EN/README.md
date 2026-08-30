# 分布式机器人控制系统部署与使用指南 

## 系统概述

基于ROS2的分布式机器人控制系统，由两个主要功能包组成：
- `master`: 主控端功能包，部署在主机1
- `robot_slave_executor`: 从站执行器功能包，部署在主机2

系统通过网络通信实现主控端对从站机器人的远程控制，支持多种预设动作的执行，并通过RViz可视化机器人状态。

**注意**：本系统既支持分布式部署（两台主机），也支持本地部署（同一主机），具体配置请参考"本地控制配置"章节。


## 部署模式

### 1. 分布式部署（两台主机）

#### 网络配置

系统使用ROS2 DDS通信，需要配置以下环境变量：

**主机1和主机2均需配置：**
```bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
#export RMW_IMPLEMENTATION=rmw_fastrtps_cpp  该环境变量可提高跨版本兼容性，酌情配置
```

确保两台机器处于同一局域网
测试：ping <对方主机ip>

#### 环境准备

**主机1（控制端） && 主机2（执行端）**

安装ROS Humble 或 ROS Jazzy

以 sudo apt install ros-[ROS_DISTRO]-desktop 格式安装 ROS2 时，RViz已经默认被安装

#### 功能包部署与构建

在两台主机上分别执行以下步骤：

1. **创建工作空间**
若主机已建立工作空间可跳过，以下是新建工作空间例子
```bash
# 创建工作空间
mkdir -p ~/[workspace_path]/src
cd ~/robot_control_ws/src

# 将对应的功能包复制到src目录
# 主机1复制master功能包
# 主机2复制robot_slave_executor功能包
```

2. **构建工作空间**
```bash
cd [工作空间目录]
source /opt/ros/$ROS_DISTRO/setup.bash  
#rosdep install --from-paths src --ignore-src -r -y  可选，见下方说明
colcon build --packages-select [功能包名称]  # master 或 robot_slave_executor
source install/setup.bash
```

**关于rosdep install命令：**
- **可选执行**：如果已安装完整ROS2桌面版，大部分依赖已包含在内
- **建议执行**：确保所有依赖项完整安装，避免潜在问题
- **依赖检查**：系统主要依赖rclpy、std_msgs、sensor_msgs、geometry_msgs等ROS2核心包


### 2. 本地控制配置（同一主机）

系统也支持在同一主机上运行两个功能包，实现本地控制。这种配置更适合开发和测试环境。



## ROS2版本兼容性

### 支持的版本组合

系统支持以下ROS2版本组合：

| 主控端版本 | 从站版本 | 兼容性 |
|-----------|---------|--------|
| Humble | Humble | 完全兼容 |
| Humble | Jazzy | 兼容 |
| Jazzy | Humble | 兼容 | 
| Jazzy | Jazzy | 完全兼容 | 


**注意事项：**
- 确保使用相同版本的Python（建议Python 3.10+）
- 某些特定于发行版的功能可能不兼容
- 建议使用标准ROS2消息类型，避免使用发行版特有消息

## 系统启动

### 重要：启动顺序

**必须先启动从站执行器，再启动主控端**，否则主控端无法找到从站节点。

### 1. 分布式部署启动

#### 从站执行器 (主机2)
```bash
cd [工作空间目录]
source install/setup.bash
ros2 launch robot_slave_executor slave_executor.launch.py
```
**RViz报错**
- 从站启动后rviz报错：“Robot Model”-“Status Error”-“URDF”
- 由于该机器人模型文件没有hand_left_link.STL和hand_right_Link.STL,属于正常现象，不影响除手部以外的动作执行
- 若需除去报错，请自行删除或添加手部模型文件

#### 主控端 (主机1)
```bash
cd [工作空间目录]
source install/setup.bash
./install/master/share/master/scripts/start_master.sh
```

### 2. 本地控制启动

#### 从站执行器 (终端1)
```bash
cd [工作空间目录]
source install/setup.bash
ros2 launch robot_slave_executor slave_executor.launch.py
```

#### 主控端 (终端2)
```bash
cd [工作空间目录]
source install/setup.bash
./install/master/share/master/scripts/start_master.sh
```

## 使用说明

### 主控端界面操作

启动成功后，主控端将显示CLI交互界面，提供以下选项：

```
============================================================
        分布式机器人控制系统 - 主控端
============================================================
连接状态: 已连接
当前时间: 2024-01-01 12:00:00
------------------------------------------------------------
主菜单:
1. 动作控制
2. 系统状态
3. 系统设置
4. 帮助信息
0. 退出系统
------------------------------------------------------------
请选择操作 (0-4): 
```

#### 动作控制

选择"1"进入动作控制菜单：

```
============================================================
        分布式机器人控制系统 - 主控端
============================================================
连接状态: 已连接
当前时间: 2024-01-01 12:00:00
------------------------------------------------------------
动作控制:
1. wave - 招手
2. expand - 扩张
3. agree - 同意
s. 开始执行
b. 返回主菜单
------------------------------------------------------------
请选择操作: 
```

- 选择数字键选择要执行的动作
- 按's'键开始执行选定的动作
- 执行过程中可按's'键停止动作


## 仿真系统注意事项

由于当前系统是RViz仿真执行，而非物理机器人，应注意以下事项：

### 1. 启动顺序
- **必须先启动从站执行器**，再启动主控端
- 确保从站完全启动后再启动主控端（建议等待3-5秒）

### 2. 话题和服务冲突
- 确保网络中没有其他ROS2系统使用相同的话题名称
- 使用唯一的ROS_DOMAIN_ID避免冲突
- 在开发环境中，建议使用命名空间隔离不同系统

### 3. 资源管理
- RViz可能占用较多GPU资源，注意系统性能
- 监控系统资源使用情况，避免过载
- 在性能较低的机器上，可以降低RViz的显示质量

### 4. 进程管理
- 正确关闭所有节点，避免僵尸进程
- 使用Ctrl+C优雅关闭节点
- 如需强制关闭，使用`killall -9 ros2`清理进程

## 配置文件说明

## 机器人模型

系统使用的人形机器人模型包含以下关节：

- **移动关节**: wheel_left_joint, wheel_right_joint
- **升降关节**: lift_joint
- **腰部关节**: waist_joint
- **头部关节**: head_joint
- **左臂关节**: J1_left_joint ~ J7_left_joint
- **右臂关节**: J1_right_joint ~ J7_right_joint

## 调试命令

```bash
# 查看话题
ros2 topic list

# 查看关节状态
ros2 topic echo /joint_states

# 查看系统状态
ros2 topic echo /master_status

# 查看从站状态
ros2 topic echo /slave_status

# 查看节点图
ros2 run rqt_graph rqt_graph

# 监控话题频率
ros2 topic hz /joint_states

# 查看节点信息
ros2 node info /slave_executor_node
```

## 扩展开发

### 添加新动作

1. 在`robot_actions.py`中定义新动作轨迹
2. 在`robot_actions.yaml`中添加动作配置
3. 在`ActionType`枚举中添加新动作类型

### 修改机器人模型

1. 修改URDF文件
2. 更新mesh文件
3. 调整关节名称和限制

### 自定义界面

1. 修改`user_interface.py`中的菜单逻辑
2. 添加新的交互功能
3. 扩展状态显示

