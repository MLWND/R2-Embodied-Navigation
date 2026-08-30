#!/usr/bin/env python
"""验证 3 段手指: 6 位置量控制, intermediate/distal 通过 mimic 联动"""
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
cam = scene.add_camera(res=(900, 700), pos=(0.3, 0.6, 0.5), lookat=(0.14, 0.25, 0.5), fov=40)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=URDF), surface=gs.surfaces.Default())
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default())
scene.build()
print(f"[3s] build 完成, n_dofs={r2.n_dofs}")

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

# 6 位置量控制: 手指弯曲 (index/middle/ring/pinky=1.0, thumb_flex=0.5)
ctrl = []
for side in ['left', 'right']:
    for name in ['index', 'middle', 'ring', 'pinky']:
        ctrl += jidx(f'{name}_{side}_joint')
    ctrl += jidx(f'thumb_flex_{side}_joint')
ctrl = list(set(ctrl))
targets = [1.0]*len(ctrl)
for i in range(100):
    t = (i+1)/100
    r2.set_dofs_position([x*t for x in targets], ctrl)
    scene.step()

# 检查 3 段联动: index proximal/intermediate/distal 的关节角
links = scene.rigid_solver.links
name2idx = {l.name: i for i, l in enumerate(links)}
for name in ['index_proximal_left_Link', 'index_intermediate_left_Link', 'index_distal_left_Link']:
    if name in name2idx:
        p = to_np(links[name2idx[name]].get_pos()).reshape(-1)
        print(f"[3s] {name}: pos={p.round(3).tolist()}")

# 检查 mimic 关节角 (intermediate/distal 应跟随 proximal)
for jn in ['index_left_joint', 'index_intermediate_left_joint', 'index_distal_left_joint']:
    q = flat(r2.get_dofs_position(jidx(jn)))
    print(f"[3s] {jn}: q={q.round(3).tolist()}")

rgb, _, _, _ = cam.render(rgb=True, depth=True)
Image.fromarray(np.asarray(rgb)).save(f"{OUT}/hand_3seg.png")
print(f"[3s] 已保存 hand_3seg.png")
print("[3s] 完成")
