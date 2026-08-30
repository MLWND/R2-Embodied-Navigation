#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import  MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import threading
import time
import signal
import sys

# 导入自定义消息和服务
from interface_pkg.srv import (
    SetFingerPositions, 
    ClearError, 
    RobotEnableControl,
    GlobalSpeedSet,
    MoveToJointPositions,
    MotControl,
    NavControl
)
from interface_pkg.msg import Robotstatus, MotFeedback, NavStatus

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.logger = self.get_logger()
        
        # 创建服务客户端
        self.callback_group = ReentrantCallbackGroup()
        
        self.finger_client = self.create_client(
            SetFingerPositions, '/set_finger_positions', callback_group=self.callback_group
        )
        self.clear_error_client = self.create_client(
            ClearError, '/robot_clear_error', callback_group=self.callback_group
        )
        self.enable_control_client = self.create_client(
            RobotEnableControl, '/robot_enable_control', callback_group=self.callback_group
        )
        self.global_speed_client = self.create_client(
            GlobalSpeedSet, '/global_speed_set', callback_group=self.callback_group
        )
        self.move_joints_client = self.create_client(
            MoveToJointPositions, '/move_joint_positions', callback_group=self.callback_group
        )
        self.motor_control_client = self.create_client(
            MotControl, '/motor_control', callback_group=self.callback_group
        )
        self.nav_control_client = self.create_client(
            NavControl, '/nav_control', callback_group=self.callback_group
        )
        
        # 创建话题订阅
        self.robot_status_sub = self.create_subscription(
            Robotstatus, 'robot_status', self.robot_status_callback, 10
        )
        self.motor_feedback_sub = self.create_subscription(
            MotFeedback, 'motor_feedback', self.motor_feedback_callback, 10
        )
        self.nav_result_sub = self.create_subscription(
            NavStatus, 'nav_result', self.nav_result_callback, 10
        )
        
        # 状态变量
        self.robot_status = None
        self.motor_feedback = None
        self.nav_result = None
        
    def wait_for_services(self, timeout_sec=30):
        """等待所有服务可用，带超时处理"""
        services = [
            (self.finger_client, '/set_finger_positions'),
            (self.clear_error_client, '/robot_clear_error'),
            (self.enable_control_client, '/robot_enable_control'),
            (self.global_speed_client, '/global_speed_set'),
            (self.move_joints_client, '/move_joint_positions'),
            (self.motor_control_client, '/motor_control'),
            (self.nav_control_client, '/nav_control')
        ]
        
        start_time = time.time()
        all_services_available = False
        
        while not all_services_available and (time.time() - start_time) < timeout_sec:
            all_services_available = True
            
            for service, name in services:
                if not service.service_is_ready():
                    self.logger.info(f'等待服务 {name}...')
                    all_services_available = False
                    time.sleep(0.5)
                    break  # 跳出内层循环，重新检查所有服务
            
            if all_services_available:
                self.logger.info("所有服务都已就绪!")
                return True
        
        # 检查哪些服务没有就绪
        for service, name in services:
            if not service.service_is_ready():
                self.logger.error(f"服务 {name} 超时未就绪!")
        
        return False
    

    # 回调函数
    def robot_status_callback(self, msg):
        self.robot_status = msg
        
    def motor_feedback_callback(self, msg):
        self.motor_feedback = msg
        
    def nav_result_callback(self, msg):
        self.nav_result = msg
    
    # 机器人状态检查
    def get_robot_status(self, timeout=30):
        """等待机器人状态正常"""
        start_time = time.time()
        self.logger.info("等待手臂状态正常...")
        
        while time.time() - start_time < timeout:
            if self.robot_status is not None:
                is_alarming = self.robot_status.is_alarming
                joint_move_complete = self.robot_status.joint_move_complete
                
                print(f"手臂状态: is_alarming={is_alarming}, joint_move_complete={joint_move_complete}")
                
                if not is_alarming and joint_move_complete:
                    elapsed = time.time() - start_time
                    self.logger.info(f"手臂状态正常! 耗时: {elapsed:.2f}秒")
                    return True
        
        time.sleep(0.5)
        
        # self.logger.error("等待手臂状态超时!")
        return False
    
    def get_motor_status(self, timeout=30):
        """等待电机状态正常"""
        start_time = time.time()
        self.logger.info("等待电机状态正常...")
        
        while time.time() - start_time < timeout:
            if self.motor_feedback is not None:
                waist_ready = self.motor_feedback.waist_ready
                ascend_ready = self.motor_feedback.ascend_ready
                
                # self.logger.info(f"电机状态: waist_ready={waist_ready}, ascend_ready={ascend_ready}")
                
                if waist_ready ==1 and ascend_ready ==1:
                    elapsed = time.time() - start_time
                    self.logger.info(f"电机状态正常! 耗时: {elapsed:.2f}秒")
                    return True
            
            #time.sleep(0.5)
        
        # self.logger.error("等待电机状态超时!")
        return False
    
    def get_agv_status(self, timeout=60):
        """等待AGV状态正常"""
        start_time = time.time()
        self.logger.info("等待AGV状态正常...")
        
        while time.time() - start_time < timeout:
            if self.nav_result is not None:
                state = self.nav_result.state
                # self.logger.info(f"AGV状态: state={state}")
                
                if state in [0, 3]:
                    elapsed = time.time() - start_time
                    self.logger.info(f"AGV状态正常! 耗时: {elapsed:.2f}秒")
                    return True
        
            #time.sleep(0.5)
        
        self.logger.error("等待AGV状态超时!")
        return False
    
    # 机器人控制方法
    def rhand_use(self,positions_left,positions_right):
        """发送放箱子命令"""
        if not self.finger_client.wait_for_service(timeout_sec=2.0):
            self.logger.error("抓服务未就绪!")
            return
        self.logger.info("发送放箱子命令...")
        request = SetFingerPositions.Request()
        request.positions_left = positions_left
        request.positions_right = positions_right
        
        future = self.finger_client.call_async(request)
        future.add_done_callback(
                self._rhand_response_callback
            )
        return True
    def _rhand_response_callback(self,future):
        try:
            response = future.result()
            print(f'手爪结果：{response.success}')
        except Exception as e:
            print(f'手爪结果：false')
    
    def _clear_response_callback(self,future):
        try:
            response = future.result()
            print(f'清错结果：{response.success}')
        except Exception as e:
            print(f'清错结果：false')
    
    def _enable_response_callback(self,future):
        try:
            response = future.result()
            print(f'使能结果：{response.success}')
        except Exception as e:
            print(f'使能结果：false')
    
    def _speed_response_callback(self,future):
        try:
            response = future.result()
            print(f'速度结果：{response.success}')
        except Exception as e:
            print(f'速度结果：false')
    
    def _robot_response_callback(self,future):
        try:
            response = future.result()
            print(f'运动结果：{response.success}')
        except Exception as e:
            print(f'运动结果：false')

    def _line_response_callback(self,future):
        try:
            response = future.result()
            print(f'升降结果：{response.success}')
        except Exception as e:
            print(f'升降结果：false')
    def _agv_response_callback(self,future):
        try:
            response = future.result()
            print(f'agv结果：{response.success}')
        except Exception as e:
            print(f'agv结果：false')
    def robot_move_origin(self):
        """手臂回原点"""
        self.logger.info("手臂回原点...")
        success = True

        if not self.clear_error_client.wait_for_service(timeout_sec=2.0):
            self.logger.error("清除错误服务未就绪!")
            return
        # 清除错误
        clear_request = ClearError.Request(clear_error=True)
        clear_future = self.clear_error_client.call_async(clear_request)
        clear_future.add_done_callback(
                self._clear_response_callback
            )

        if not self.enable_control_client.wait_for_service(timeout_sec=2.0):
            self.logger.error("使能控制服务未就绪!")
            return
        # 使能控制
        enable_request = RobotEnableControl.Request(enable=True)
        enable_future = self.enable_control_client.call_async(enable_request)
        enable_future.add_done_callback(
                self._enable_response_callback
            )

        if not self.global_speed_client.wait_for_service(timeout_sec=2.0):
            self.logger.error("设置速度服务未就绪!")
            return
        # 设置速度
        speed_request = GlobalSpeedSet.Request(speed=40.0)
        speed_future = self.global_speed_client.call_async(speed_request)
        speed_future.add_done_callback(
                self._speed_response_callback
            )

        return self.arm_move([0.0]*7, [0.0]*7)
        
    
    def arm_move(self, left_joints, right_joints):
        """控制手臂运动"""
        if not self.move_joints_client.wait_for_service(timeout_sec=2.0):
            self.logger.error("手臂运动服务未就绪!")
            return
        self.logger.info(f"控制手臂运动: left={left_joints}, right={right_joints}")
        request = MoveToJointPositions.Request()
        request.left_joints = left_joints
        request.right_joints = right_joints
        
        future = self.move_joints_client.call_async(request)
        future.add_done_callback(
                self._robot_response_callback
            )
        time.sleep(0.5)

        # 等待状态正常
        self.get_robot_status()

        
        return True
    
    def line_move(self, angle=0.0, pos=0.0, head_angle=0.0, 
                 speed_waist=1, speed_ascend=1, speed_head=1):
        """控制腰部、升降和头部运动"""
        if not self.motor_control_client.wait_for_service(timeout_sec=2.0):
            self.logger.error("线运动运动服务未就绪!")
            return
        self.logger.info(f"控制线运动: angle={angle}, pos={pos}, head_angle={head_angle}")
        request = MotControl.Request()
        request.angle = angle
        request.speed_waist = int(speed_waist)
        request.pos = pos
        request.speed_ascend = int(speed_ascend)
        request.head_angle = head_angle
        request.speed_head = int(speed_head)
        
        future = self.motor_control_client.call_async(request)
        future.add_done_callback(
                self._line_response_callback
            )

        time.sleep(0.5)
        # 等待状态正常
        self.get_motor_status()

        return True
    
    def agv_move(self, command, x, y, radian, speed):
        """控制AGV移动"""
        if not self.nav_control_client.wait_for_service(timeout_sec=2.0):
            self.logger.error("AGV运动服务未就绪!")
            return
        self.logger.info(f"控制AGV移动: command={command}, x={x}, y={y}, radian={radian}, speed={speed}")
        request = NavControl.Request()
        request.command = int(command)
        request.x = x
        request.y = y
        request.radian = radian
        request.speed = speed
        
        future = self.nav_control_client.call_async(request)
        future.add_done_callback(
                self._agv_response_callback
            )

        time.sleep(1)
        # 等待状态正常
        self.get_agv_status()

        return True

# def test():

def main(args=None):
    rclpy.init(args=args)
    
    # 创建控制器实例
    controller = RobotController()
    
    # 使用多线程执行器
    # executor = SingleThreadedExecutor()
    
    executor = MultiThreadedExecutor()
    executor.add_node(controller)
    
    # 在后台运行执行器
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    # 注册信号处理
    def signal_handler(sig, frame):
        controller.logger.info("接收到终止信号，关闭中...")
        executor.shutdown()
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # controller.logger.info("机器人控制开始...")
        # if not controller.wait_for_services():
        #     controller.logger.error("服务初始化失败，退出程序")
        #     return
        
        # 初始化动作
        if not controller.line_move(angle=0.0, pos=0.0, head_angle=0.0):
            controller.logger.error("初始化线运动失败!")
            return
        if not controller.robot_move_origin():
            controller.logger.error("手臂回原点失败!")
            return
        left_joints_f =  [3676,17837,17606,17654,17486,200]
        right_joints_f = [3676,17837,17606,17654,17486,200]
        controller.rhand_use(left_joints_f,right_joints_f)

        
        # 主控制循环
        # while True:
        # controller.logger.info("开始新循环...")
        # 数数位置
        left_joints = [-174.74, 127.50, 96.14, -84.34, -82.98, -24.95, 78.53]
        right_joints = [0.0] *7
        if not controller.arm_move(left_joints, right_joints):
            controller.logger.error("手臂动作失败!")
            time.sleep(5)
        # 数字1
        left_joints_f =  [276,17837,9781,10138,9886,200]
        right_joints_f = [3676,17837,17606,17654,17486,200]
        controller.rhand_use(left_joints_f,right_joints_f)
        time.sleep(1)
        # 数字2
        left_joints_f =  [276,17837,17606,10138,9886,200]
        right_joints_f = [3676,17837,17606,17654,17486,200]
        controller.rhand_use(left_joints_f,right_joints_f)
        time.sleep(1)
        # 数字3
        left_joints_f =  [276,17837,17606,17654,9886,200]
        right_joints_f = [3676,17837,17606,17654,17486,200]
        controller.rhand_use(left_joints_f,right_joints_f)
        time.sleep(1)
        left_joints_f =  [3676,17837,17606,17654,17486,200]
        right_joints_f = [3676,17837,17606,17654,17486,200]
        controller.rhand_use(left_joints_f,right_joints_f)

        # 单手挥
        # 位置1：
        left_joints = [-144.74, 44.13, 96.14, -48.34, -60.98, -8.23, 45.53]
        right_joints = [0.0] *7
        if not controller.arm_move(left_joints, right_joints):
            controller.logger.error("手臂动作失败!")
            time.sleep(5)

        # 位置2
        left_joints = [-144.74, 57.93, 96.14, -41.34, -60.98, 5.43, 45.53]
        right_joints = [0.0] *7
        if not controller.arm_move(left_joints, right_joints):
            controller.logger.error("手臂动作失败!")
            time.sleep(5)

        # 位置3
        left_joints = [-150.74, 28.06, 96.14, -48.34, -60.98, -12.97, 45.53]
        right_joints = [0.0] *7
        if not controller.arm_move(left_joints, right_joints):
            controller.logger.error("手臂动作失败!")
            time.sleep(5)
        # 位置4
        left_joints = [-144.74, 57.93, 96.14, -41.34, -60.98, 5.43, 45.53]
        right_joints = [0.0] *7
        if not controller.arm_move(left_joints, right_joints):
            controller.logger.error("手臂动作失败!")
            time.sleep(5)

        # 双手挥
        # 位置1：
        left_joints = [-144.74, 44.13, 96.14, -48.34, -60.98, -8.23, 45.53]
        right_joints = [-144.74, 44.13, 96.14, -48.34, -60.98, -8.23, 45.53]
        if not controller.arm_move(left_joints, right_joints):
            controller.logger.error("手臂动作失败!")
            time.sleep(5)

        # 位置2
        left_joints = [-144.74, 57.93, 96.14, -41.34, -60.98, 5.43, 45.53]
        right_joints = [-144.74, 57.93, 96.14, -41.34, -60.98, 5.43, 45.53]
        if not controller.arm_move(left_joints, right_joints):
            controller.logger.error("手臂动作失败!")
            time.sleep(5)

        # 位置3
        left_joints = [-150.74, 28.06, 96.14, -48.34, -60.98, -12.97, 45.53]
        right_joints = [-150.74, 28.06, 96.14, -48.34, -60.98, -12.97, 45.53]
        if not controller.arm_move(left_joints, right_joints):
            controller.logger.error("手臂动作失败!")
            time.sleep(5)
        # 位置4
        left_joints = [-144.74, 57.93, 96.14, -41.34, -60.98, 5.43, 45.53]
        right_joints = [-144.74, 57.93, 96.14, -41.34, -60.98, 5.43, 45.53]
        if not controller.arm_move(left_joints, right_joints):
            controller.logger.error("手臂动作失败!")
            time.sleep(5)
        # 点头
        if not controller.line_move(angle=0.0, pos=0.0, head_angle=35.0):
            controller.logger.error("升降动作失败!")
            time.sleep(5)
        time.sleep(0.7)
        if not controller.line_move(angle=0.0, pos=0.0, head_angle=0.0):
            controller.logger.error("升降动作失败!")
            time.sleep(5)

        left_joints = [0.0, 0.0, 0.0, -76.91, 0.0, 0.0, 0.0]
        right_joints = [0.0, 0.0, 0.0, -76.91, 0.0, 0.0, 0.0]
        if not controller.arm_move(left_joints, right_joints):
           controller.logger.error("手臂动作失败!")
        
        time.sleep(5)

        # 手臂回原点
        if not controller.arm_move([0.0]*7, [0.0]*7):
            controller.logger.error("手臂回原点失败!")
            time.sleep(5)
            # continue
        
        controller.logger.info("完成一个循环，等待5秒后重新开始...")
        # time.sleep(5)
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.logger.info("用户中断操作")
    except Exception as e:
        controller.logger.error(f"发生异常: {str(e)}")
    finally:
        print('1')
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
