#!/usr/bin/env python
"""验证 v2: 手臂自然下垂 + 高 kp 保持 0 位姿, 手不插地"""
import numpy as np
import genesis as gs
from PIL import Image

URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf"
OUT = "/home/xujinlong/test/genesis_smoke/output"

def to_np(x):
    if hasattr(x, "cpu"): x = x.cpu()
    if isinstance(x, (tuple, list)): return [to_np(i) for i in x]
    return np.asarray(x)

def flat(x):
    parts = []
    for i in to_np(x):
        parts.append(np.asarray(i).reshape(-1))
    return np.concatenate(parts)

def jidx(name):
    return flat(r2.get_joint(name).dofs_idx_local).astype(int).tolist()

gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/60.0, substeps=4), show_viewer=False)
cam = scene.add_camera(res=(800, 600), pos=(2.5, 0, 1.2), lookat=(0, 0, 0.9), fov=55)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=URDF), surface=gs.surfaces.Default())
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default())
scene.build()
print(f"[nat2] build 完成, n_dofs={r2.n_dofs}")

# 手臂/手指/腰部/头部: 高 kp 保持 0 位姿; 轮子: 自由
arm_dofs = []
for j in r2.joints:
    n = j.name
    if n in ('wheel_left_joint', 'wheel_right_joint'):
        continue
    arm_dofs += jidx(n)
arm_dofs = list(set(arm_dofs))
r2.set_dofs_kp(5000.0, arm_dofs)
r2.set_dofs_kv(200.0, arm_dofs)
wl, wr = jidx('wheel_left_joint'), jidx('wheel_right_joint')
r2.set_dofs_kp(0.0, wl+wr); r2.set_dofs_kv(100.0, wl+wr)

# settle: 每步强制手臂 0 位姿 (set_dofs_position 直接设状态, 保证不弯)
for i in range(200):
    r2.set_dofs_position([0.0]*len(arm_dofs), arm_dofs)
    scene.step()

links = scene.rigid_solver.links
name2idx = {l.name: i for i, l in enumerate(links)}
for name in ['J2_left_Link', 'J5_left_Link', 'J7_left_Link', 'hand_left_Link', 'index_left_Link', 'pinky_left_Link']:
    if name in name2idx:
        p = to_np(links[name2idx[name]].get_pos()).reshape(-1)
        print(f"[nat2] {name}: z={p[2]:.4f}")

c = r2.get_contacts(with_entity=ground)
from collections import Counter
names = [links[int(c['link_b'][i])].name for i in range(len(c['position']))]
print(f"[nat2] 接地: {dict(Counter(names))}")
print(f"[nat2] base pos = {to_np(r2.get_pos()).reshape(-1).round(4).tolist()}")

rgb, _, _, _ = cam.render(rgb=True, depth=True)
Image.fromarray(np.asarray(rgb)).save(f"{OUT}/natural_pose2.png")
print(f"[nat2] 已保存 natural_pose2.png")
print("[nat2] 完成")
