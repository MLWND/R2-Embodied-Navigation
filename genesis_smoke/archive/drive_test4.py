#!/usr/bin/env python
"""驱动测试 v4: 查驱动时基座姿态(俯仰/翻滚)+ 轮子接触"""
import numpy as np
import genesis as gs

R2_URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf"

def to_np(x):
    if hasattr(x, "cpu"): x = x.cpu()
    if isinstance(x, (tuple, list)): return [to_np(i) for i in x]
    return np.asarray(x)

def flat(x):
    parts = []
    for i in to_np(x):
        parts.append(np.asarray(i).reshape(-1))
    return np.concatenate(parts)

def quat_to_euler(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    # pitch (y轴), roll (x轴), yaw (z轴)
    pitch = np.arcsin(2*(w*y - z*x))
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw

gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/60.0, substeps=4), show_viewer=False)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=R2_URDF), surface=gs.surfaces.Default())
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default())
scene.build()
print("[drive4] build 完成")
wl = flat(r2.get_joint('wheel_left_joint').dofs_idx_local).astype(int).tolist()
wr = flat(r2.get_joint('wheel_right_joint').dofs_idx_local).astype(int).tolist()

for _ in range(100):
    scene.step()
q = flat(r2.get_quat())
print(f"[drive4] settle: pos={to_np(r2.get_pos()).round(4).tolist()} euler(roll,pitch,yaw)={np.round(quat_to_euler(q),3).tolist()}")

r2.set_dofs_kp(50.0); r2.set_dofs_kv(5.0)
target = -18.85
for i in range(200):
    r2.set_dofs_position([target*(i+1)/200, target*(i+1)/200], wl+wr)
    scene.step()
    if i % 25 == 0:
        p = to_np(r2.get_pos()).round(3).tolist()
        q = flat(r2.get_quat())
        e = np.round(quat_to_euler(q), 3).tolist()
        print(f"[drive4] step {i}: pos={p} euler={e}")
print("[drive4] 完成")
