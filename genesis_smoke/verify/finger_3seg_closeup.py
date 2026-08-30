#!/usr/bin/env python
"""手指 3 段特写: 相机对准手指侧面, 确认指节分段"""
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
# 相机对准左手手指 (index 在 x≈0.12, y≈0.28, z≈0.4), 从侧面看
cam = scene.add_camera(res=(1000, 800), pos=(0.5, 0.15, 0.35), lookat=(0.12, 0.28, 0.4), fov=30)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=URDF), surface=gs.surfaces.Default())
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default())
scene.build()
print(f"[3s] build 完成, n_dofs={r2.n_dofs}")

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

# 手指微弯 (index/middle/ring/pinky=0.5, 展示 3 段)
ctrl = []
for side in ['left', 'right']:
    for name in ['index', 'middle', 'ring', 'pinky']:
        ctrl += jidx(f'{name}_{side}_joint')
ctrl = list(set(ctrl))
for i in range(80):
    t = (i+1)/80
    r2.set_dofs_position([0.5*t]*len(ctrl), ctrl)
    scene.step()

rgb, _, _, _ = cam.render(rgb=True, depth=True)
Image.fromarray(np.asarray(rgb)).save(f"{OUT}/finger_3seg_closeup.png")
print(f"[3s] 已保存 finger_3seg_closeup.png")
print("[3s] 完成")
