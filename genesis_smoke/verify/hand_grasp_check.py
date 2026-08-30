#!/usr/bin/env python
"""验证灵巧手可驱动: 手指弯曲 (抓取动作)"""
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
print(f"[hand] build 完成, n_dofs={r2.n_dofs}")

# 手臂/手指 kp, 轮子自由
arm_dofs = []
for j in r2.joints:
    if j.name not in ('wheel_left_joint', 'wheel_right_joint'):
        arm_dofs += jidx(j.name)
arm_dofs = list(set(arm_dofs))
r2.set_dofs_kp(5000.0, arm_dofs); r2.set_dofs_kv(200.0, arm_dofs)
wl, wr = jidx('wheel_left_joint'), jidx('wheel_right_joint')
r2.set_dofs_kp(0.0, wl+wr); r2.set_dofs_kv(100.0, wl+wr)

# settle: 手臂 0 位姿
for i in range(150):
    r2.set_dofs_position([0.0]*len(arm_dofs), arm_dofs)
    scene.step()

# 手指弯曲: index/middle/ring/pinky = 1.0, thumb_flex = 0.5 (抓取)
finger_joints = []
for side in ['left', 'right']:
    for name in ['index', 'middle', 'ring', 'pinky']:
        finger_joints += jidx(f'{name}_{side}_joint')
    finger_joints += jidx(f'thumb_flex_{side}_joint')
finger_joints = list(set(finger_joints))
targets = [1.0]*len(finger_joints)
for i in range(100):
    t = (i+1)/100
    r2.set_dofs_position([x*t for x in targets], finger_joints)
    scene.step()

links = scene.rigid_solver.links
name2idx = {l.name: i for i, l in enumerate(links)}
for name in ['index_left_Link', 'middle_left_Link', 'pinky_left_Link', 'thumb_flex_left_Link']:
    if name in name2idx:
        p = to_np(links[name2idx[name]].get_pos()).reshape(-1)
        print(f"[hand] {name}: pos={p.round(3).tolist()}")

c = r2.get_contacts(with_entity=ground)
from collections import Counter
names = [links[int(c['link_b'][i])].name for i in range(len(c['position']))]
print(f"[hand] 接地: {dict(Counter(names))}")

rgb, _, _, _ = cam.render(rgb=True, depth=True)
Image.fromarray(np.asarray(rgb)).save(f"{OUT}/hand_grasp.png")
print(f"[hand] 已保存 hand_grasp.png")
print("[hand] 完成")
