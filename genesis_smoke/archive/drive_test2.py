#!/usr/bin/env python
"""驱动测试 v2: 位置控制轮子 + 长 settle, 验证机器人能否前进/转向"""
import numpy as np
import genesis as gs

R2_URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf"

def to_np(x):
    if hasattr(x, "cpu"):
        x = x.cpu()
    if isinstance(x, (tuple, list)):
        return [to_np(i) for i in x]
    return np.asarray(x)

def flat(x):
    parts = []
    for i in to_np(x):
        parts.append(np.asarray(i).reshape(-1))
    return np.concatenate(parts)

gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/60.0, substeps=4), show_viewer=False)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=R2_URDF), surface=gs.surfaces.Default())
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0.0, 0.0, 0.0)), surface=gs.surfaces.Default())
scene.build()
print("[drive2] build 完成")

wl = flat(r2.get_joint('wheel_left_joint').dofs_idx_local).astype(int).tolist()
wr = flat(r2.get_joint('wheel_right_joint').dofs_idx_local).astype(int).tolist()

# 长 settle
for _ in range(100):
    scene.step()
print(f"[drive2] settle 后 base z={to_np(r2.get_pos()).reshape(-1)[2]:.4f}")

# 位置控制: 轮子转 3 圈 (18.85 rad), 分 200 步
target = 18.85
r2.set_dofs_kp(50.0)
r2.set_dofs_kv(5.0)
for i in range(200):
    r2.set_dofs_position([target * (i+1)/200, target * (i+1)/200], wl + wr)
    scene.step()
    if i % 50 == 0:
        p = to_np(r2.get_pos()).round(3).tolist()
        print(f"[drive2] step {i}: base={p}")

p_end = to_np(r2.get_pos()).round(3).tolist()
print(f"[drive2] 前进测试结束: base={p_end} (x 应显著增大)")

# 转向: 左轮正转右轮反转
r2.set_dofs_position([0.0, 0.0], wl + wr)
for _ in range(50):
    scene.step()
for i in range(200):
    r2.set_dofs_position([target * (i+1)/200, -target * (i+1)/200], wl + wr)
    scene.step()
    if i % 50 == 0:
        q = to_np(r2.get_quat()).reshape(-1)
        yaw = 2 * np.arctan2(q[3], q[0])
        print(f"[drive2] 转向 step {i}: yaw={yaw:.3f} rad")
print("[drive2] 完成")
