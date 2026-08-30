"""R2 前进运动验证 — 纯 SAPIEN 路径 (已验证可用)

背景: ManiSkill env 中的轮式驱动存在未定位的悬浮问题(轮子空转, 牵引力 ~0)。
纯 SAPIEN 路径 + 速度斜坡可稳定驱动 R2 前进 (无后翻, 贴地, ~50% 效率)。

关键修复 (2026-08-17):
1. 轮子碰撞体: STL 凸包(132顶点棱柱) → 精确圆柱 (r=0.085, len=0.13, rpy=90°绕x使轴沿y)
   - 之前生成的 URDF 把 <origin rpy> 插在 <geometry> 内部(无效顺序), 圆柱未旋转 → 后翻根因
2. caster: 球 r=0.05 顶高 0.10 深入底盘(底 0.063) → r=0.03, steer z=0.08 (顶高 0.06)
3. 速度斜坡: 0→5 rad/s 分 150 步 (0.75s), 避免瞬时加速导致后翻
4. 驱动参数: PDJointVel damping=500, force_limit=2000 (SAPIEN 隐式 PD)
5. 目标速度加倍: SAPIEN 驱动存在半速特性(目标10实际~7.4), 需加倍补偿

用法:
  source /opt/miniconda3/etc/profile.d/conda.sh && conda activate maniskill
  export PXR_WORK_THREAD_LIMIT=8
  taskset -c 0-7 python r2_drive_pure_sapien.py
"""
import os
import numpy as np
import sapien

URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_cylwheel.urdf"
TARGET = float(os.environ.get("TARGET", "10.0"))  # 目标轮速 rad/s (SAPIEN 半速特性: 实际≈一半, 加倍补偿)
RAMP = int(os.environ.get("RAMP", "150"))          # 斜坡步数 (0.75s 到满速)
HOLD = int(os.environ.get("HOLD", "400"))          # 满速保持步数
DAMPING = float(os.environ.get("DAMPING", "500"))
FORCE_LIMIT = float(os.environ.get("FL", "2000"))

scene = sapien.Scene()
scene.set_timestep(0.005)
scene.add_ground(altitude=0, render=False)

loader = scene.create_urdf_loader()
loader.fix_root_link = False
loader.set_material(1.0, 1.0, 0.0)  # 轮子 μ=1.0 (地面默认 0.3)
robot = loader.load(URDF)
robot.set_pose(sapien.Pose(p=[0, 0, 0.0]))
robot.set_qpos(np.zeros(robot.dof))

wl_j = [j for j in robot.get_joints() if j.get_name() == "wheel_left_joint"][0]
wr_j = [j for j in robot.get_joints() if j.get_name() == "wheel_right_joint"][0]
base = [l for l in robot.get_links() if l.get_name() == "base_link"][0]

wl_j.set_drive_properties(0.0, DAMPING, FORCE_LIMIT, "force")
wr_j.set_drive_properties(0.0, DAMPING, FORCE_LIMIT, "force")

# settle 200 步
for i in range(200):
    scene.step()
p0 = base.get_pose().p.copy()
print(f"[DRIVE] settle: base_z={p0[2]:.4f}", flush=True)


def euler(q):
    w, x, y, z = q
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - x * z), -1, 1)))
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    return roll, pitch


# 前进: 斜坡 + 保持
total = RAMP + HOLD
for i in range(total):
    tgt = min(TARGET, TARGET * (i + 1) / RAMP)
    wl_j.set_drive_velocity_target(tgt)
    wr_j.set_drive_velocity_target(-tgt)
    scene.step()
    if i % 100 == 0:
        p = base.get_pose().p
        roll, pitch = euler(base.get_pose().q)
        print(f"[DRIVE] step{i}: base=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) pitch={pitch:+.1f}° roll={roll:+.1f}°", flush=True)

p1 = base.get_pose().p
d = float(np.linalg.norm(p1[:2] - p0[:2]))
t = total * 0.005
expect = TARGET * 0.085 * t
print(f"[DRIVE] 结果: 位移={d:.3f}m 时间={t:.2f}s 平均v={d/t:.3f}m/s (目标{TARGET*0.085:.2f}) 效率={d/max(expect,1e-6)*100:.0f}%", flush=True)
print(f"[DRIVE] base_z {p0[2]:.4f}->{p1[2]:.4f}  (应保持贴地 ~0)", flush=True)
