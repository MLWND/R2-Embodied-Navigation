import rclpy
from rclpy.executors import  MultiThreadedExecutor
import threading
import time
import signal
import sys
from test_base import RobotController


def main(args=None):
    rclpy.init(args=args)
    
    # 创建控制器实例
    controller = RobotController()
    
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
        if not controller.line_move(angle=0.0, pos=0.0, head_angle=0.0):return
        if not controller.robot_move_origin(speed=40.0):return
        left_joints_f =  [3676,17837,17606,17654,17486,200]
        right_joints_f = [3676,17837,17606,17654,17486,200]
        controller.rhand_use(left_joints_f,right_joints_f)

        # agv移动到放置为止
        # if not controller.agv_move(command = 1, x = 0.79, y = 2.37, radian = 1.6 ,speed= 0.6):return
        
        
        # 主控制循环
        # while True:
        # controller.logger.info("开始新循环...")
        # 数数位置
        left_joints = [-174.74, 127.50, 96.14, -84.34, -82.98, -24.95, 78.53]
        right_joints = [0.0] *7
        if not controller.arm_move(left_joints, right_joints):return
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
        if not controller.arm_move(left_joints, right_joints):return

        # 位置2
        left_joints = [-144.74, 57.93, 96.14, -41.34, -60.98, 5.43, 45.53]
        right_joints = [0.0] *7
        if not controller.arm_move(left_joints, right_joints):return

        # 位置3
        left_joints = [-150.74, 28.06, 96.14, -48.34, -60.98, -12.97, 45.53]
        right_joints = [0.0] *7
        if not controller.arm_move(left_joints, right_joints):return
        # 位置4
        left_joints = [-144.74, 57.93, 96.14, -41.34, -60.98, 5.43, 45.53]
        right_joints = [0.0] *7
        if not controller.arm_move(left_joints, right_joints):return

        # 双手挥
        # 位置1：
        left_joints = [-144.74, 44.13, 96.14, -48.34, -60.98, -8.23, 45.53]
        right_joints = [-144.74, 44.13, 96.14, -48.34, -60.98, -8.23, 45.53]
        if not controller.arm_move(left_joints, right_joints):return

        # 位置2
        left_joints = [-144.74, 57.93, 96.14, -41.34, -60.98, 5.43, 45.53]
        right_joints = [-144.74, 57.93, 96.14, -41.34, -60.98, 5.43, 45.53]
        if not controller.arm_move(left_joints, right_joints):return

        # 位置3
        left_joints = [-150.74, 28.06, 96.14, -48.34, -60.98, -12.97, 45.53]
        right_joints = [-150.74, 28.06, 96.14, -48.34, -60.98, -12.97, 45.53]
        if not controller.arm_move(left_joints, right_joints):return
        # 位置4
        left_joints = [-144.74, 57.93, 96.14, -41.34, -60.98, 5.43, 45.53]
        right_joints = [-144.74, 57.93, 96.14, -41.34, -60.98, 5.43, 45.53]
        if not controller.arm_move(left_joints, right_joints):return
        # 点头
        if not controller.line_move(angle=0.0, pos=0.0, head_angle=35.0):return
        time.sleep(0.7)
        if not controller.line_move(angle=0.0, pos=0.0, head_angle=0.0):return

        left_joints = [0.0, 0.0, 0.0, -76.91, 0.0, 0.0, 0.0]
        right_joints = [0.0, 0.0, 0.0, -76.91, 0.0, 0.0, 0.0]
        if not controller.arm_move(left_joints, right_joints):return

        # 手臂回原点
        if not controller.arm_move([0.0]*7, [0.0]*7):return
        
        controller.logger.info("完成一个循环...")

        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.logger.info("用户中断操作")
    except Exception as e:
        controller.logger.error(f"发生异常: {str(e)}")
    finally:
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()