"""R2 完整基础运动验证 (纯 SAPIEN): 前进/后退/左转/右转/掉头

符号约定 (URDF):
  左轮轴 +y, 右轮轴 -y
  前进 = 左轮 +ω, 右轮 -ω (各自轴正方向)
  原地左转(逆时针) = 左轮 -ω, 右轮 -ω
  原地右转 = 左轮 +ω, 右轮 +ω
"""
import os
import numpy as np
import sapien

URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_cylwheel.urdf"
TARGET = float(os.environ.get("TARGET", "10.0"))  # 目标轮速 (SAPIEN 半速, 加倍补偿)
RAMP = int(os.environ.get("RAMP", "150"))
DAMPING = float(os.environ.get("DAMPING", "500"))
FORCE_LIMIT = float(os.environ.get("FL", "2000"))

scene = sapien.Scene()
scene.set_timestep(0.005)
scene.add_ground(altitude=0, render=False)

loader = scene.create_urdf_loader()
loader.fix_root_link = False
loader.set_material(1.0, 1.0, 0.0)
robot = loader.load(URDF)
robot.set_pose(sapien.Pose(p=[0, 0, 0.0]))
robot.set_qpos(np.zeros(robot.dof))

wl_j = [j for j in robot.get_joints() if j.get_name() == "wheel_left_joint"][0]
wr_j = [j for j in robot.get_joints() if j.get_name() == "wheel_right_joint"][0]
base = [l for l in robot.get_links() if l.get_name() == "base_link"][0]

wl_j.set_drive_properties(0.0, DAMPING, FORCE_LIMIT, "force")
wr_j.set_drive_properties(0.0, DAMPING, FORCE_LIMIT, "force")

# settle
for i in range(200):
    scene.step()


def euler(q):
    w, x, y, z = q
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - x * z), -1, 1)))
    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return roll, pitch, yaw


def drive(steps, wl_cmd, wr_cmd, label, settle_between=100):
    """差速驱动: wl_cmd/wr_cmd 为各自轴正方向的目标速度 (rad/s)"""
    p0 = base.get_pose().p.copy()
    q0 = base.get_pose().q
    yaw0 = euler(q0)[2]
    for i in range(steps):
        tgt = min(TARGET, TARGET * (i + 1) / RAMP)
        wl_j.set_drive_velocity_target(wl_cmd * tgt / TARGET)
        wr_j.set_drive_velocity_target(wr_cmd * tgt / TARGET)
        scene.step()
    p1 = base.get_pose().p
    yaw1 = euler(base.get_pose().q)[2]
    d = float(np.linalg.norm(p1[:2] - p0[:2]))
    dyaw = (yaw1 - yaw0 + 180) % 360 - 180  # 归一化到 [-180,180]
    z = p1[2]
    print(f"[MOVE] {label}: 位移={d:.3f}m 方向=({p1[0]-p0[0]:+.3f},{p1[1]-p0[1]:+.3f}) Δyaw={dyaw:+.1f}° base_z={z:.4f}", flush=True)
    # settle 回正
    wl_j.set_drive_velocity_target(0.0)
    wr_j.set_drive_velocity_target(0.0)
    for i in range(settle_between):
        scene.step()
    return d, dyaw


print(f"[MOVE] settle: base_z={base.get_pose().p[2]:.4f}", flush=True)

# 前进: 左轮+ 右轮- (各轴正方向)
drive(300, +TARGET, -TARGET, "前进")
# 后退
drive(300, -TARGET, +TARGET, "后退")
# 左转 (逆时针): 左轮慢 右轮快
drive(300, +TARGET * 0.3, -TARGET, "左转")
# 右转: 左轮快 右轮慢
drive(300, +TARGET, -TARGET * 0.3, "右转")
# 原地左转 (掉头): 左轮- 右轮-
drive(600, -TARGET, -TARGET, "原地左转(掉头)")
# 原地右转
drive(600, +TARGET, +TARGET, "原地右转")
