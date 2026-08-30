#!/usr/bin/env python
"""列出所有 link 碰撞体世界 bbox, 找穿模/重叠 + 检查手臂 0 位姿"""
import sys
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
print(f"[bbox] 共 {len(links)} 个 link")
print(f"[bbox] {'link':<28} {'z_min':>8} {'z_max':>8}  备注")
for i, l in enumerate(links):
    if l.collision:
        # 碰撞体 bbox (通过 link 变换 + 各 geoms)
        p = to_np(l.get_pos()).reshape(-1)
        zmin = zmax = p[2]
        ngeom = 0
        # 尝试获取碰撞几何
        try:
            geoms = l.collision.geoms
            ngeom = len(geoms)
        except Exception:
            pass
        note = ''
        if p[2] - 0.085 < -0.01: note = ' <-- 轮子/低部件?'
        if 'J' in l.name and 'hand' in l.name: note = ' <-- 手爪'
        print(f"[bbox] {l.name:<28} {p[2]:>8.4f} {p[2]:>8.4f}  ngeom={ngeom}{note}")

print()
print("[bbox] 接触地面:", end=' ')
c = r2.get_contacts(with_entity=ground)
from collections import Counter
names = [links[int(c['link_b'][i])].name for i in range(len(c['position']))]
print(dict(Counter(names)))

# 各 link 世界位置汇总
print()
print("[bbox] 手臂链位置:")
for name in ['lift_Link', 'waist_Link', 'J1_left_Link', 'J2_left_Link', 'J3_left_Link', 'J4_left_Link', 'J5_left_Link', 'J6_left_Link', 'J7_left_Link', 'hand_left_Link']:
    for i, l in enumerate(links):
        if l.name == name:
            p = to_np(l.get_pos()).reshape(-1)
            print(f"  {name}: z={p[2]:.4f}")
            break
print("[bbox] 完成")
