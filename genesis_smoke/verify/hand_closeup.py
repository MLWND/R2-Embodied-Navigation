#!/usr/bin/env python
"""手部特写渲染: 相机对准左手, 确认手指微弯/拇指内收/手腕"""
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
# 相机对准左手 (左臂在 y>0 侧, 手在 x≈0.14, y≈0.25, z≈0.54)
cam = scene.add_camera(res=(900, 700), pos=(0.3, 0.6, 0.5), lookat=(0.14, 0.25, 0.5), fov=40)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=URDF), surface=gs.surfaces.Default())
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default())
scene.build()
print(f"[hand] build 完成, n_dofs={r2.n_dofs}")

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

rgb, _, _, _ = cam.render(rgb=True, depth=True)
Image.fromarray(np.asarray(rgb)).save(f"{OUT}/hand_closeup.png")
print(f"[hand] 已保存 hand_closeup.png")
print("[hand] 完成")
