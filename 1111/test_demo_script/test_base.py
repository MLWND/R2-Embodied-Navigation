from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
import time

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
        self.init_responce_status()
        
        # 状态变量
        self.robot_status = None
        self.motor_feedback = None
        self.nav_result = None
        
    def set_responce_status(self,name='enable', status=False):
        '''清除回复传递的状态'''
        if name == 'arm_enable':
            self.arm_enable_response = status
        elif name == 'arm_speed':
            self.arm_speed_response = status
        elif name == 'finger':
            self.finger_response = status
        elif name == 'arm_clear':
            self.arm_clear_response = status
        elif name == 'arm_move':
            self.arm_move_response = status
        elif name == 'motor':
            self.motor_response = status
        elif name == 'agv':
            self.agv_response = status


    def init_responce_status(self):
        '''服务调用回复状态结果（用于判断是否执行下一步）'''
        self.finger_response=None
        self.arm_enable_response=None
        self.arm_clear_response=None
        self.arm_speed_response=None
        self.arm_move_response=None
        self.motor_response=None
        self.agv_response=None


    def wait_for_services(self,service1):
        """等待服务可用，带超时处理"""
        service_name = None
        if service1 == 'finger':
            service_name = self.finger_client
        elif service1 == 'arm_clear':
            service_name = self.clear_error_client
        elif service1 == 'arm_enable':
            service_name = self.enable_control_client
        elif service1 == 'arm_speed':
            service_name = self.global_speed_client
        elif service1 == 'arm_move':
            service_name = self.move_joints_client
        elif service1 == 'motor':
            service_name = self.motor_control_client
        elif service1 == 'agv':
            service_name = self.nav_control_client
        else:
            return False,'input error!!'
        ret = service_name.wait_for_service(timeout_sec=3.0)
        msg = f'{service1} 服务已就绪'
        if not ret:
            msg = f'{service1} 服务未就绪'
        return ret,msg



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
                # print(f"手臂状态: is_alarming={is_alarming}, joint_move_complete={joint_move_complete}")
                if not is_alarming and joint_move_complete:
                    elapsed = time.time() - start_time
                    self.logger.info(f"手臂状态正常：is_alarming={is_alarming}, joint_move_complete={joint_move_complete}  耗时: {elapsed:.2f}秒")
                    return True
        time.sleep(0.05)
        self.logger.error("等待手臂状态超时!")
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
                    self.logger.info(f"升降腰头部电机状态正常：waist_ready={waist_ready}, ascend_ready={ascend_ready} 耗时: {elapsed:.2f}秒")
                    return True
            time.sleep(0.05)
    
        self.logger.error("-------------------------等待电机状态超时!-------------------------------------")
        return False
    
    def get_agv_status(self, timeout=120):
        """等待AGV状态正常"""
        start_time = time.time()
        self.logger.info("等待AGV状态正常...")
        
        while time.time() - start_time < timeout:
            if self.nav_result is not None:
                state = self.nav_result.state
                # self.logger.info(f"AGV状态: state={state}")
                if state in [0, 3]:
                    elapsed = time.time() - start_time
                    self.logger.info(f"AGV状态正常：state={state} 耗时: {elapsed:.2f}秒")
                    return True
        
            time.sleep(0.05)
        self.logger.error("等待AGV状态超时!")
        return False
    
    # 回调函数
    def _rhand_response_callback(self,future):
        try:
            response = future.result()
            self.finger_response=response.success
            self.logger.info(f'手爪服务发送结果：{response.success}')
        except Exception as e:
            self.logger.error(f'手爪服务发送结果：false')
    def _clear_response_callback(self,future):
        try:
            response = future.result()
            self.arm_clear_response=response.success
            self.logger.info(f'手臂清错服务发送结果：{response.success}')
        except Exception as e:
            self.logger.error(f'手臂清错服务发送结果：false')
    def _enable_response_callback(self,future):
        try:
            response = future.result()
            self.arm_enable_response=response.success
            self.logger.info(f'手臂使能服务发送结果：{response.success}')
        except Exception as e:
            self.logger.error(f'手臂使能服务发送结果：false')
    def _speed_response_callback(self,future):
        try:
            response = future.result()
            self.arm_speed_response=response.success
            self.logger.info(f'手臂速度服务发送结果：{response.success}')
        except Exception as e:
            self.logger.error(f'手臂速度服务发送结果：false')
    def _robot_response_callback(self,future):
        try:
            response = future.result()
            self.arm_move_response=response.success
            self.logger.info(f'手臂运动服务发送结果：{response.success}')
        except Exception as e:
            self.logger.error(f'手臂运动服务发送结果：false')
    def _line_response_callback(self,future):
        try:
            response = future.result()
            self.motor_response=response.success
            self.logger.info(f'升降腰头部服务发送结果：{response.success}')
        except Exception as e:
            self.logger.error(f'升降腰头部服务发送结果：false')
    def _agv_response_callback(self,future):
        try:
            response = future.result()
            self.agv_response=response.success
            self.logger.info(f'agv服务发送结果：{response.success}')
        except Exception as e:
            self.logger.error(f'agv服务发送结果：false')

    # ----------------------------------------------机器人控制方法------------------------------------
    def rhand_use(self,positions_left,positions_right):
        """发送手爪运动命令"""
        if not self.wait_for_services(service1='finger'):return

        self.logger.info("发送手爪运动命令...")
        request = SetFingerPositions.Request()
        request.positions_left = positions_left
        request.positions_right = positions_right
        
        # 服务调用返回
        future = self.finger_client.call_async(request)
        future.add_done_callback(
                self._rhand_response_callback
            )
        while self.finger_response is None:
            # 等待更新
            time.sleep(0.1)
        success1=self.finger_response
        self.finger_response=None
        return success1
    def robot_move_origin(self,speed):
        """手臂回原点"""
        self.logger.info("手臂回原点...")
        success = True

        if not self.wait_for_services(service1='arm_clear'):return
        # 清除错误
        clear_request = ClearError.Request(clear_error=True)
        clear_future = self.clear_error_client.call_async(clear_request)
        clear_future.add_done_callback(
                self._clear_response_callback
            )
        while self.arm_clear_response is None:
            # 等待更新
            time.sleep(0.1)
        success1=self.arm_clear_response
        self.arm_clear_response=None
        if success1 is not True:
            return success1

        if not self.wait_for_services(service1='arm_enable'):return
        # 使能控制
        enable_request = RobotEnableControl.Request(enable=True)
        enable_future = self.enable_control_client.call_async(enable_request)
        enable_future.add_done_callback(
                self._enable_response_callback
            )
        while self.arm_enable_response is None:
            # 等待更新
            time.sleep(0.1)
        success1=self.arm_enable_response
        self.arm_enable_response=None
        if success1 is not True:
            return success1

        if not self.wait_for_services(service1='arm_speed'):return
        # 设置速度
        speed_request = GlobalSpeedSet.Request(speed=speed)
        speed_future = self.global_speed_client.call_async(speed_request)
        speed_future.add_done_callback(
                self._speed_response_callback
            )
        while self.arm_speed_response is None:
            # 等待更新
            time.sleep(0.1)
        success1=self.arm_speed_response
        self.arm_speed_response=None
        if success1 is not True:
            return success1

        return self.arm_move([0.0]*7, [0.0]*7)

    def arm_move(self, left_joints, right_joints):
        """控制手臂运动"""
        if not self.wait_for_services(service1='arm_move'):return

        self.logger.info(f"控制手臂运动: left={left_joints}, right={right_joints}")
        request = MoveToJointPositions.Request()
        request.left_joints = left_joints
        request.right_joints = right_joints
        
        future = self.move_joints_client.call_async(request)
        future.add_done_callback(
                self._robot_response_callback
            )
        while self.arm_move_response is None:
            # 等待更新
            time.sleep(0.1)
        success1=self.arm_move_response
        self.arm_move_response=None
        if success1 is not True:
            return success1

        time.sleep(0.5)

        # 等待状态正常
        ret = self.get_robot_status()
        return ret
    
    def line_move(self, angle=0.0, pos=0.0, head_angle=0.0, 
                 speed_waist=1, speed_ascend=1, speed_head=1):
        """控制腰部、升降和头部运动"""
        if not self.wait_for_services(service1='motor'):return

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
        while self.motor_response is None:
            # 等待更新
            time.sleep(0.1)
        success1=self.motor_response
        self.motor_response=None
        if success1 is not True:
            return success1
        time.sleep(0.5)
        # 等待状态正常
        ret = self.get_motor_status()
        return ret
    
    def agv_move(self, command, x, y, radian, speed):
        """控制AGV移动"""
        if not self.wait_for_services(service1='agv'):return

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
        while self.agv_response is None:
            # 等待更新
            time.sleep(0.1)
        success1=self.agv_response
        self.agv_response=None
        if success1 is not True:
            return success1

        time.sleep(1)
        # 等待状态正常
        ret = self.get_agv_status()

        return ret
