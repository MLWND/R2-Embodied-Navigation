#!/usr/bin/env python
"""渲染自然下垂姿态: 正面 + 侧面, 确认手指排列和手掌朝向"""
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
cam_front = scene.add_camera(res=(800, 600), pos=(0, -2.5, 1.0), lookat=(0, 0, 0.9), fov=55)
cam_side = scene.add_camera(res=(800, 600), pos=(2.5, 0, 1.0), lookat=(0, 0, 0.9), fov=55)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=URDF), surface=gs.surfaces.Default())
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default())
scene.build()
print(f"[fp] build 完成, n_dofs={r2.n_dofs}")

arm_dofs = []
for j in r2.joints:
    if j.name not in ('wheel_left_joint', 'wheel_right_joint'):
        arm_dofs += jidx(j.name)
arm_dofs = list(set(arm_dofs))
r2.set_dofs_kp(5000.0, arm_dofs); r2.set_dofs_kv(200.0, arm_dofs)
wl, wr = jidx('wheel_left_joint'), jidx('wheel_right_joint')
r2.set_dofs_kp(0.0, wl+wr); r2.set_dofs_kv(100.0, wl+wr)

for i in range(150):
    r2.set_dofs_position([0.0]*len(arm_dofs), arm_dofs)
    scene.step()

links = scene.rigid_solver.links
name2idx = {l.name: i for i, l in enumerate(links)}
# 手指位置 (x 方向确认排开)
for name in ['thumb_flex_left_Link', 'index_left_Link', 'middle_left_Link', 'ring_left_Link', 'pinky_left_Link']:
    if name in name2idx:
        p = to_np(links[name2idx[name]].get_pos()).reshape(-1)
        print(f"[fp] {name}: x={p[0]:.4f} z={p[2]:.4f}")

rgb, _, _, _ = cam_front.render(rgb=True, depth=True)
Image.fromarray(np.asarray(rgb)).save(f"{OUT}/natural_front.png")
rgb, _, _, _ = cam_side.render(rgb=True, depth=True)
Image.fromarray(np.asarray(rgb)).save(f"{OUT}/natural_side.png")
print("[fp] 已保存 natural_front.png / natural_side.png")
print("[fp] 完成")
