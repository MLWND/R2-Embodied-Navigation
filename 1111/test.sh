#!/bin/bash
# 启动节点函数
start_node() {
    local pkg=$1
    local node=$2
    echo "启动节点: $pkg/$node"
    ros2 run $pkg $node 2>&1 | sed "s/^/[${node}] /" &
    PID=$!
    sleep 1
    if ps -p $PID > /dev/null; then
        echo "$node 启动成功 (PID $PID)"
        PIDS+=($PID)
    else
        echo "$node 启动失败！"
        exit 1
    fi
}
# 节点进程列表
declare -a PIDS=()
# 清理函数
cleanup() {
    echo "清理中..."
    for pid in "${PIDS[@]}"; do
        if ps -p $pid > /dev/null; then
            echo "终止进程 $pid"
            kill -SIGTERM $pid
        fi
    done
    exit
}

WORKSPACE_PATH="$HOME/ros_ws"
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


# 获取运动状态
# 手臂运动状态
get_robot_status(){
    start_time=$(date +%s)
    while true; do
        echo "提取手臂到位状态(joint_move_complete)和升降到位状态(ascend_ready)值:"
        msg=$(ros2 topic echo /robot_status --once 2>/dev/null | tr '\n' ' ' | sed -e 's/---//g' -e 's/  */ /g')
        # echo  "$msg"
        result=$(echo "$msg" | awk '{
            for (i=1; i<=NF; i++) {
                # 查找 is_alarming 的值
                if ($i == "is_alarming:") {
                    if ($(i+1) ~ /^(true|false|True|False)/) {
                        value = $(i+1);
                        i++;
                        is_alarming = (value == "true" || value == "True") ? 1 : 0;
                    }
                }
                
                # 查找 joint_move_complete 的值
                if ($i == "joint_move_complete:") {
                    if ($(i+1) ~ /^(true|false|True|False)/) {
                        value = $(i+1);
                        i++;
                        joint_move_complete = (value == "true" || value == "True") ? 1 : 0;
                    }
                }
            }
        }
        END {
            # 设置默认值
            if (is_alarming == "") is_alarming = -1;
            if (joint_move_complete == "") joint_move_complete = -1;
            printf "%s %s", is_alarming, joint_move_complete;
        }')
        # 将结果解析到两个变量
        read is_alarming joint_move_complete <<< "$result"
        # 输出变量值
        echo "is_alarming: $is_alarming"
        echo "joint_move_complete: $joint_move_complete"
        # 比对值是否为1
        if [ "$is_alarming" = "0" ] && [ "$joint_move_complete" = "1" ]; then
            echo "手臂状态都正常"
            end_time=$(date +%s)
            usetime=$(($end_time-$start_time))
            echo "运行时间：${usetime}秒"
            return 0

        fi
    done

}
# 腰部电机运动状态
get_motor_status(){
    start_time=$(date +%s)
    echo "仅提取腰部到位状态(waist_ready)和升降到位状态(ascend_ready)值:"
    while true; do 
    
        msg=$(ros2 topic echo /motor_feedback --once 2>/dev/null | tr '\n' ' ' | sed -e 's/---//g' -e 's/  */ /g')

        # 使用awk提取waist_ready和ascend_ready两个值并分配给变量
        result=$(echo "$msg" | awk 'BEGIN {
        waist = "-1";
        ascend = "-1";
        head_flag = "-1";}
        {for (i=1; i<=NF; i++) {
                if ($i ~ /:/) {
                    split($i, arr, ":");
                    key = arr[1];
                    if (length(arr) > 1 && arr[2] != "") {
                        value = arr[2];
                    } else if (i < NF && $(i+1) ~ /^[0-9-]/) {
                        value = $(i+1);
                        i++;
                    } else {value = "-1";}
                    if (key == "waist_ready") waist = value;
                    else if (key == "ascend_ready") ascend = value;
                    # else if (key == "head_flag") head_flag = value;
                    }}}
        END {
            printf "%s %s", waist, ascend;}')

        # 将结果解析到两个变量
        read waist_ready ascend_ready <<< "$result"
        # 输出变量值
        echo "waist_ready: $waist_ready"
        echo "ascend_ready: $ascend_ready"

        if [ "$waist_ready" = "1" ] && [ "$ascend_ready" = "1" ]; then
            echo "升降电机及腰部状态都正常"
            end_time=$(date +%s)
            usetime=$(($end_time-$start_time))
            echo "运行时间：${usetime}秒"
            return 0
        fi
    done
}
# agv 运动
get_agv_status(){
    echo "获取底盘运动状态："
    while true;do
        msg=$(ros2 topic echo /nav_result --once 2>/dev/null | tr '\n' ' ' | sed -e 's/---//g' -e 's/  */ /g')

        result=$(echo "$msg" | awk 'BEGIN {
        state = "-1";}
        {for (i=1; i<=NF; i++) {
                if ($i ~ /:/) {
                    split($i, arr, ":");
                    key = arr[1];
                    if (length(arr) > 1 && arr[2] != "") {
                        value = arr[2];
                    } else if (i < NF && $(i+1) ~ /^[0-9-]/) {
                        value = $(i+1);
                        i++;
                    } else {value = "-1";}
                    if (key == "state") state = value;
                    }}}
        END {
            printf "%s", state;}')
        # 将结果解析到两个变量
        read state <<< "$result"
        echo "state: $state"
        if [[ "$state" = "3"  ||  "$state" = "0" ]]; then
            echo "AGV状态正常"
            return 0
        fi
    done
    
}


# 手臂回原点
robot_move_origin(){
    echo "---------------------------------发送手臂控制运动--------------------------------------"
    ros2 service call /robot_clear_error interface_pkg/srv/ClearError "{clear_error: true}"
    # 给手臂使能
    while true ;do
        result=$(ros2 service call /robot_enable_control interface_pkg/srv/RobotEnableControl\
         "{enable: true}")
        if echo "$response" | grep -q "success: False"; then
            echo "命令返回 FALSE...重试..."
            continue
        else
            break
        fi
    done
    # 设速度
    ros2 service call /global_speed_set interface_pkg/srv/GlobalSpeedSet "{speed: 80.0}"
    echo "速度设置完成"
    # 动手臂
    # robot_move_fun "{left_joints: [-3.94,-89.56,-94.94,80,-1.68,-72.76,-77.78] ,
    #   right_joints: [-3.94,-89.56,-94.94,80,-1.68,-72.76,-77.78]}"
    arm_move_fun "{left_joints: [0,0,0,0,0,0,0] ,right_joints:  [0,0,0,0,0,0,0]}"
}

# 动手臂
arm_move_fun(){
    command=$1
    # 动手臂
    if ! ros2 service call /move_joint_positions interface_pkg/srv/MoveToJointPositions\
     "$command" > /dev/null; then
        echo "警告：手臂控制指令发送失败！"
    fi
    sleep 0.5
    echo "检测手臂运动后手臂电机状态..."
    get_robot_status
    status_1=$?
    if [ $status_1 -eq 0 ]; then # 为0就正常，进行下一步
        echo "检测到手臂电机状态正常，进行下一步..."
    fi

}
# 升降腰部及头部运动
line_move_fun(){
    command=$1
    # 动手臂
    if ! ros2 service call /motor_control interface_pkg/srv/MotControl\
     "$command" > /dev/null; then
        echo "警告：升降腰部头部控制指令发送失败！"
    fi
    sleep 0.5
    echo "检测升降腰部头部运动后手臂电机状态..."
    get_motor_status
    status_1=$?
    if [ $status_1 -eq 0 ]; then # 为0就正常，进行下一步
        echo "检测到升降腰部头部电机状态正常，进行下一步..."
    fi

}

# agv移动
agv_move_fun(){
    command=$1
    # 动手臂
    if ! ros2 service call /nav_control interface_pkg/srv/NavControl\
     "$command" > /dev/null; then
        echo "警告：agv底盘控制指令发送失败！"
    fi
    sleep 1
    echo "检测agv底盘运动后agv底盘电机状态..."
    get_agv_status
    status_1=$?
    if [ $status_1 -eq 0 ]; then # 为0就正常，进行下一步
        echo "检测到agv底盘电机状态正常，进行下一步..."    
    fi

}
# -----------------------------------------------------主进程---------------------------------------
# 初始化程序环境
init_path
# 注册信号处理
trap cleanup SIGINT SIGTERM
# # 启动所有节点
# start_node motor_can_pkg ros_serial_bridge
# start_node topic_pkg topic_interface_pub
# # start_node rohand_pkg RoHandInterface
# start_node arm_pkg robot_control

echo "所有节点已启动 | 等待系统初始化..."
# 腰回原
line_move_fun "{angle: 0, speed_waist: 1, reserve_1: 0, pos: 0, speed_ascend: 1, reserve_2: 0, head_angle: 0, speed_head: 1,reserve_3: 0}"
robot_move_origin   # 手臂原点

while true;do
    agv_move_fun "{command: 1, x: -0.31, y: 0.57, radian: -1.32, speed: 0.6}"

    arm_move_fun "{left_joints: [-57,22.07,83,-30,-84,-23,22] ,right_joints: [-57,22.07,83,-30,-84,-23,22]}"

    line_move_fun "{angle: 35, speed_waist: 1, reserve_1: 0, pos: 600, speed_ascend: 1, reserve_2: 0, head_angle: 35, speed_head: 1,reserve_3: 0}"

    agv_move_fun "{command: 1, x: -0.18, y: -0.99, radian: 1.85, speed: 0.6}"
    # 回原
    line_move_fun "{angle: 0, speed_waist: 1, reserve_1: 0, pos: 0, speed_ascend: 1, reserve_2: 0, head_angle: 0, speed_head: 1,reserve_3: 0}"
    arm_move_fun "{left_joints: [0,0,0,0,0,0,0] ,right_joints:  [0,0,0,0,0,0,0]}"
    
done




