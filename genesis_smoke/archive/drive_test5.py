#!/usr/bin/env python
"""驱动测试 v5: 速度控制 + 高增益 + 高轮速"""
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

gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/60.0, substeps=4), show_viewer=False)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=R2_URDF), surface=gs.surfaces.Default())
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default())
scene.build()
print("[d5] build 完成")
wl = flat(r2.get_joint('wheel_left_joint').dofs_idx_local).astype(int).tolist()
wr = flat(r2.get_joint('wheel_right_joint').dofs_idx_local).astype(int).tolist()

for _ in range(100):
    scene.step()
print(f"[d5] settle: base={to_np(r2.get_pos()).round(4).tolist()}")

# 速度控制: 高增益 + 高轮速 (反转向前进)
r2.set_dofs_kp(1000.0, wl + wr)
r2.set_dofs_kv(100.0, wl + wr)
r2.set_dofs_velocity([-10.0, -10.0], wl + wr)  # 反转向前进
for i in range(150):
    scene.step()
    if i % 30 == 0:
        p = to_np(r2.get_pos()).round(3).tolist()
        v = flat(r2.get_dofs_velocity(wl + wr)).round(2).tolist()
        print(f"[d5] step {i}: base={p} 轮速={v}")

r2.set_dofs_velocity([0.0, 0.0], wl + wr)
print("[d5] 完成")
