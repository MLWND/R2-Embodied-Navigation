"""R2 里程计闭环运动控制 (模拟 R1 真机底盘控制器)

参考 R1 真机 ROS2 接口 (1111/):
- /move (Move.srv): distance(mm) / angle(度) 位置命令
- pose_2d: line_speed / palstance / x / y / radian 反馈闭环

关键:
1. diff_drive: 线速度/角速度 → 轮子速度 (右轮轴 -y, 需取反)
2. 里程计闭环 (P 控制): 位置误差 → 速度命令
3. 航向保持: yaw 误差 → 角速度命令
4. caster μ=0.05 + swivel 摩擦=0: 自旋阻力最小
5. 累积角度: 避免 yaw ±180° 环绕误判
"""
import os
import numpy as np
import sapien

URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_swivel.urdf"
MESH = "/home/xujinlong/test/robot_example1_EN_demo/robot_example1_EN_demo/robot_model/r2_urdf_v01/meshes"
TMP = "/tmp/r2_swivel_tmp.urdf"

WHEEL_R = 0.085    # 驱动轮半径
TRACK = 0.458      # 轮距
MAX_V = 0.6        # 最大线速度 (R1: 0.3-0.6 m/s)
MAX_W = 1.5        # 最大角速度
Kp = 1.5           # 位置环增益
Kw = 2.0           # 航向环增益

def euler(q):
    w, x, y, z = q
    roll = np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y-x*z), -1, 1)))
    yaw = np.degrees(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    return roll, pitch, yaw

def make_scene():
    with open(URDF) as f:
        urdf = f.read()
    urdf = urdf.replace("package://r2_urdf_v01/meshes/", MESH + "/")
    with open(TMP, "w") as f:
        f.write(urdf)
    scene = sapien.Scene()
    scene.set_timestep(0.005)
    scene.add_ground(altitude=0, render=False)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = False
    loader.set_material(1.5, 1.5, 0.0)
    robot = loader.load(TMP)
    robot.set_pose(sapien.Pose(p=[0, 0, 0.0]))
    robot.set_qpos(np.zeros(robot.dof))
    # 禁用自碰撞 (base_link STL 覆盖躯干与手臂碰撞的翻倒根因)
    for l in robot.get_links():
        for sh in l.get_collision_shapes():
            sh.set_collision_groups([1, 1, 1<<29, 0])
    # swivel 转向轴摩擦=0 (fork 自由转向)
    for j in robot.get_joints():
        if 'swivel' in j.get_name():
            j.friction = 0.0
    # caster 小轮 μ=0.05 (自旋阻力最小)
    for l in robot.get_links():
        if 'caster' in l.get_name() and 'wheel' in l.get_name():
            for sh in l.get_collision_shapes():
                sh.set_physical_material(sapien.physx.PhysxMaterial(0.05, 0.05, 0.0))
    wl_j = [j for j in robot.get_joints() if j.get_name() == "wheel_left_joint"][0]
    wr_j = [j for j in robot.get_joints() if j.get_name() == "wheel_right_joint"][0]
    base = [l for l in robot.get_links() if l.get_name() == "base_link"][0]
    wl_j.set_drive_properties(0.0, 500.0, 2000.0, "force")
    wr_j.set_drive_properties(0.0, 500.0, 2000.0, "force")
    for i in range(200):
        scene.step()
    return scene, robot, wl_j, wr_j, base

def diff_drive(v, omega):
    """差速运动学: 线速度/角速度 → 轮子速度 (右轮轴 -y, 取反)"""
    v_l = (v - omega * TRACK / 2) / WHEEL_R
    v_r = -(v + omega * TRACK / 2) / WHEEL_R
    return v_l, v_r

def move_linear(scene, wl_j, wr_j, base, target_dist, label):
    """里程计闭环直线移动: target_dist>0 前进, <0 后退"""
    p0 = base.get_pose().p.copy()
    yaw0 = euler(base.get_pose().q)[2]
    n = 0
    fwd = np.array([np.cos(np.radians(yaw0)), np.sin(np.radians(yaw0))])
    for i in range(2000):
        n += 1
        p = base.get_pose().p
        dx = p[0] - p0[0]
        dy = p[1] - p0[1]
        # 沿初始航向的有符号位移
        dist = float(dx * fwd[0] + dy * fwd[1])
        dyaw = (euler(base.get_pose().q)[2] - yaw0 + 180) % 360 - 180
        err = target_dist - dist
        if abs(err) < 0.005:
            break
        v_cmd = np.clip(Kp * err, -MAX_V, MAX_V)
        yaw_err = (0 - dyaw + 180) % 360 - 180
        omega_cmd = np.clip(Kw * np.radians(yaw_err), -0.8, 0.8)
        v_l, v_r = diff_drive(v_cmd, omega_cmd)
        wl_j.set_drive_velocity_target(v_l)
        wr_j.set_drive_velocity_target(v_r)
        scene.step()
    wl_j.set_drive_velocity_target(0.0)
    wr_j.set_drive_velocity_target(0.0)
    for i in range(50):
        scene.step()
    p1 = base.get_pose().p
    d = float(np.linalg.norm(p1[:2]-p0[:2]))
    dyaw = (euler(base.get_pose().q)[2] - yaw0 + 180) % 360 - 180
    print(f"[{label}] 目标={target_dist:.2f}m 实际={d:.3f}m 步数={n} Δyaw={dyaw:+.1f}°", flush=True)
    return d, dyaw

def move_spin(scene, wl_j, wr_j, base, target_deg, label):
    """里程计闭环原地旋转 (累积角度避免环绕)"""
    p0 = base.get_pose().p.copy()
    prev_yaw = euler(base.get_pose().q)[2]
    cum_yaw = 0.0
    n = 0
    for i in range(2000):
        n += 1
        yaw = euler(base.get_pose().q)[2]
        delta = (yaw - prev_yaw + 180) % 360 - 180
        cum_yaw += delta
        prev_yaw = yaw
        err = target_deg - cum_yaw
        if err > 180: err -= 360
        elif err < -180: err += 360
        if abs(err) < 2.0:
            break
        omega_cmd = np.clip(Kp * np.radians(err), -MAX_W, MAX_W)
        v_l, v_r = diff_drive(0.0, omega_cmd)
        wl_j.set_drive_velocity_target(v_l)
        wr_j.set_drive_velocity_target(v_r)
        scene.step()
    wl_j.set_drive_velocity_target(0.0)
    wr_j.set_drive_velocity_target(0.0)
    for i in range(50):
        scene.step()
    p1 = base.get_pose().p
    d = float(np.linalg.norm(p1[:2]-p0[:2]))
    print(f"[{label}] 目标={target_deg}° 实际={cum_yaw:+.1f}° 步数={n} 位移={d:.3f}m", flush=True)
    return cum_yaw, d

if __name__ == "__main__":
    scene, robot, wl_j, wr_j, base = make_scene()
    print(f"[SETTLE] base_z={base.get_pose().p[2]:.4f}", flush=True)
    move_linear(scene, wl_j, wr_j, base, +0.5, "前进")
    move_linear(scene, wl_j, wr_j, base, -0.5, "后退")
    move_spin(scene, wl_j, wr_j, base, +90, "左转90")
    move_spin(scene, wl_j, wr_j, base, -90, "右转90")
    move_spin(scene, wl_j, wr_j, base, +180, "原地左转180")
    move_spin(scene, wl_j, wr_j, base, -180, "原地右转180")
