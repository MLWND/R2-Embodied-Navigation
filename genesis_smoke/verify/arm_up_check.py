#!/usr/bin/env python
"""加载 + J2 抬臂 ±90° 后渲染, 确认手臂不插地、姿态正确 (侧视 + 正视)"""
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
# 两个相机: 正视(前方) + 侧视
cam_front = scene.add_camera(res=(800, 600), pos=(0, -2.5, 1.0), lookat=(0, 0, 0.9), fov=55)
cam_side = scene.add_camera(res=(800, 600), pos=(2.5, 0, 1.0), lookat=(0, 0, 0.9), fov=55)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=URDF), surface=gs.surfaces.Default())
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default())
scene.build()
print("[fin] build 完成")

wl, wr = jidx('wheel_left_joint'), jidx('wheel_right_joint')
j2l, j2r = jidx('J2_left_joint'), jidx('J2_right_joint')
r2.set_dofs_kp(3000.0, j2l + j2r); r2.set_dofs_kv(200.0, j2l + j2r)
r2.set_dofs_kp(1000.0, wl+wr); r2.set_dofs_kv(100.0, wl+wr)

# settle + 抬臂 (J2_left=+1.57, J2_right=-1.57)
for i in range(150):
    t = (i+1)/150
    r2.set_dofs_position([1.57*t, -1.57*t], j2l + j2r)
    scene.step()

# 接触检查
links = scene.rigid_solver.links
c = r2.get_contacts(with_entity=ground)
from collections import Counter
names = [links[int(c['link_b'][i])].name for i in range(len(c['position']))]
print(f"[fin] 接地 link: {dict(Counter(names))}")
print(f"[fin] base pos = {to_np(r2.get_pos()).reshape(-1).round(4).tolist()}")

rgb, _, _, _ = cam_front.render(rgb=True, depth=True)
Image.fromarray(np.asarray(rgb)).save(f"{OUT}/arm_up_front.png")
rgb, _, _, _ = cam_side.render(rgb=True, depth=True)
Image.fromarray(np.asarray(rgb)).save(f"{OUT}/arm_up_side.png")
print(f"[fin] 已保存 arm_up_front.png / arm_up_side.png")

# 手臂各 link 位置
name2idx = {l.name: i for i, l in enumerate(links)}
for name in ['J2_left_Link', 'J5_left_Link', 'J7_left_Link', 'hand_left_Link']:
    if name in name2idx:
        p = to_np(links[name2idx[name]].get_pos()).reshape(-1)
        print(f"[fin] {name}: z={p[2]:.4f}")
print("[fin] 完成")
