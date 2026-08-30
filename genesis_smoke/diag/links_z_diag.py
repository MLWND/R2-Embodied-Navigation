#!/usr/bin/env python
"""列出关键 link 世界 z 位置, 检查手臂 0 位姿是否插地/穿 base"""
import numpy as np
import genesis as gs

URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf"

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

for _ in range(100):
    scene.step()

links = scene.rigid_solver.links
name2idx = {l.name: i for i, l in enumerate(links)}

print(f"{'link':<28} {'x':>8} {'y':>8} {'z':>8}")
for name in ['base_link', 'lift_Link', 'waist_Link',
             'J1_left_Link', 'J2_left_Link', 'J3_left_Link', 'J4_left_Link',
             'J5_left_Link', 'J6_left_Link', 'J7_left_Link', 'hand_left_Link',
             'J1_right_Link', 'J2_right_Link', 'hand_right_Link',
             'wheel_left_Link', 'caster_front_left_wheel_Link']:
    if name in name2idx:
        p = to_np(links[name2idx[name]].get_pos()).reshape(-1)
        print(f"{name:<28} {p[0]:>8.4f} {p[1]:>8.4f} {p[2]:>8.4f}")

c = r2.get_contacts(with_entity=ground)
from collections import Counter
names = [links[int(c['link_b'][i])].name for i in range(len(c['position']))]
print(f"\n接地: {dict(Counter(names))}")
print("[diag] 完成")
