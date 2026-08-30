# 机器人动作设计与测试支持工具

## 概述

action_test功能包是一个辅助工具，用于机器人动作的设计、测试和可视化。
功能包支持rviz机器人可视化和关节gui调试，通过gui控制机器人关节动作，从而辅助设计主系统中的动作序列。

## 功能特点

1. **RViz可视化**：显示完整的3D机器人模型
2. **关节调试界面**：提供GUI滑块控制每个关节
3. **动作序列测试**：支持预定义动作的执行和测试
4. **实时状态反馈**：显示当前关节位置和系统状态


## 部署与启动

### 1. 环境准备

```bash
# 创建工作空间
mkdir -p [workspace_path]/src #[workspace_path]为工作空间目录
cd [workspace_path]/src

# 复制功能包
# - action_test功能包
```

### 2. 构建工作空间

```bash
cd [workspace_path]
source /opt/ros/humble/setup.bash  # 或 /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y  # 可选指令，见README.md步骤：构建工作空间 说明
colcon build --packages-select action_test
source install/setup.bash
```

### 3. 启动系统

```bash
cd [workspace_path]
source install/setup.bash
ros2 launch action_test action_test.launch.py
```

## RViz配置说明

### RViz报错

- 完成配置后rviz报错：“Robot Model”-“Status Error”-“URDF”
- 由于该机器人模型文件没有hand_left_link.STL和hand_right_Link.STL,属于正常现象，不影响除手部以外的动作执行
- 若需除去报错，请自行删除或添加手部模型文件


### 关节状态发布

系统会自动发布关节状态到`/joint_states`话题，包含以下19个关节：

- wheel_left_joint, wheel_right_joint（轮子关节）
- lift_joint（升降关节）
- waist_joint（腰部关节）
- head_joint（头部关节）
- J1_left_joint ~ J7_left_joint（左臂关节）
- J1_right_joint ~ J7_right_joint（右臂关节）

## 关节调试界面

### Joint State Publisher

系统会自动启动Joint State Publisher GUI，提供以下功能：

1. **关节滑块控制**：每个关节对应一个滑块
2. **实时位置显示**：显示每个关节的当前位置
3. **零位重置**：一键将所有关节归零
4. **位置保存/加载**：保存特定姿态供后续使用

### 使用方法

1. **手动控制关节**
   - 拖动滑块调整对应关节位置
   - 观察RViz中机器人模型的实时变化
   - 记录关键姿态的关节角度值

2. **姿态记录**
   - 调整到目标姿态
   - 记录各关节的角度值
   - 这些值可用于设计主系统的动作序列

3. **运动范围测试**
   - 测试每个关节的运动范围
   - 验证关节限制是否合理
   - 检查是否存在干涉或奇异点

## 动作设计与测试

### 1. 动作序列设计

使用关节调试界面设计新动作的步骤：

1. **确定关键姿态**
   - 使用滑块调整到动作的每个关键姿态
   - 记录每个姿态的关节角度值
   - 确定姿态间的过渡时间

2. **创建动作序列**
   - 将关键姿态按时间顺序排列
   - 设置每个姿态的持续时间
   - 确定动作是否需要循环

3. **验证动作合理性**
   - 检查关节角度是否在限制范围内
   - 确保姿态变化平滑
   - 验证动作安全性



## 自定义动作开发

要添加新动作，需要修改`robot_motion_controller.py`中的`ActionLibrary`类：

```python
# 在define_actions方法中添加新动作
self.actions['new_action'] = {
    'duration': 5.0,
    'loop': False,
    'sequence': [
        {
            'name': '初始姿态',
            'duration': 1.0,
            'positions': [0.0] * 19
        },
        {
            'name': '目标姿态',
            'duration': 2.0,
            'positions': self._create_positions({
                'joint_name': angle_value,
                # ... 其他关节
            })
        },
        # ... 更多姿态
    ]
}
```

## 与主系统的集成

### 1. 动作数据转换

将在test_rvizbot中设计的动作转换为主系统格式：

1. **提取关节角度序列**
   - 从test_rvizbot的动作序列中提取关键姿态
   - 记录每个姿态的关节角度值

2. **转换为主系统格式**
   - 将关节角度值转换为master/robot_actions.py中的JointTrajectory格式
   - 设置合适的时间戳和描述

3. **更新主系统**
   - 在master/robot_actions.py中添加新动作方法
   - 在master/config/robot_actions.yaml中添加动作配置

### 2. 关节名称映射

test_rvizbot与主系统使用相同的关节名称，确保数据一致性：

```
关节名称映射（完全一致）：
wheel_left_joint -> wheel_left_joint
wheel_right_joint -> wheel_right_joint
lift_joint -> lift_joint
waist_joint -> waist_joint
head_joint -> head_joint
J1_left_joint -> J1_left_joint
...
J7_right_joint -> J7_right_joint
```


## 调试与故障排除

### 1. 常见问题

#### RViz无法显示机器人模型
- 检查"Fixed Frame"是否设置为"base_link"
- 确认"Robot Model"的"Description Topic"是否为"/robot_description"
- 验证URDF文件路径是否正确

#### 关节控制无响应
- 检查Joint State Publisher是否正常运行
- 确认/joint_states话题是否正常发布
- 验证关节名称是否匹配

#### 动作执行异常
- 检查控制命令格式是否正确
- 确认动作名称是否存在于动作库中
- 验证关节角度是否在限制范围内

### 2. 调试命令

```bash
# 查看关节状态
ros2 topic echo /joint_states

# 查看机器人状态
ros2 topic echo /robot_status

# 查看节点图
ros2 run rqt_graph rqt_graph

# 检查话题信息
ros2 topic info /joint_states
ros2 topic info /robot_control_commands
```



