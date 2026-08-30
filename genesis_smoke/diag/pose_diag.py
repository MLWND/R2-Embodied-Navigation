#!/usr/bin/env python
"""姿态诊断: 加载 URDF 后检查各 link 世界位置 + 轮底/底座高度 + 接触"""
import sys
import numpy as np
import genesis as gs

URDF = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf"

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
r2 = scene.add_entity(morph=gs.morphs.URDF(file=URDF), surface=gs.surfaces.Default())
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default())
scene.build()
print(f"[diag] build 完成, URDF={URDF.split('/')[-1]}")

links = scene.rigid_solver.links
name2idx = {l.name: i for i, l in enumerate(links)}

# settle
for _ in range(200):
    scene.step()

base_pos = to_np(r2.get_pos()).reshape(-1)
print(f"[diag] base 世界位置: {base_pos.round(4).tolist()}")

# 各关键 link 世界位置
for name in ['base_link', 'wheel_left_Link', 'wheel_right_Link',
             'caster_front_left_wheel_Link', 'caster_rear_left_wheel_Link']:
    if name in name2idx:
        p = to_np(links[name2idx[name]].get_pos()).reshape(-1)
        print(f"[diag] {name}: pos={p.round(4).tolist()}")

# 轮底高度 (wheel 半径 0.085, caster 半径 0.05)
wl = to_np(links[name2idx['wheel_left_Link']].get_pos()).reshape(-1)
wr = to_np(links[name2idx['wheel_right_Link']].get_pos()).reshape(-1)
print(f"[diag] 左轮底 = {wl[2]-0.085:.4f}, 右轮底 = {wr[2]-0.085:.4f}")

# base mesh 底 (mesh z 底 -0.0068 + base origin 0.07)
print(f"[diag] base mesh 底 ≈ {base_pos[2] + 0.07 - 0.0068:.4f}")

# 接触
c = r2.get_contacts(with_entity=ground)
from collections import Counter
names = [links[int(c['link_b'][i])].name for i in range(len(c['position']))]
print(f"[diag] 接地 link: {dict(Counter(names))}")
print(f"[diag] 接触点数: {len(c['position'])}")
print("[diag] 完成")
