#!/usr/bin/env python
"""接触几何诊断: 驱动轮接触点的 position/normal/force, 验证摩擦约束"""
import numpy as np
import genesis as gs

R2_URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf"

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
r2 = scene.add_entity(morph=gs.morphs.URDF(file=R2_URDF), surface=gs.surfaces.Default())
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default())
scene.build()
print("[cg] build 完成")

for _ in range(100):
    scene.step()

links = scene.rigid_solver.links
name2idx = {l.name: i for i, l in enumerate(links)}
print(f"[cg] wheel_left z={float(flat(links[name2idx['wheel_left_Link']].get_pos())[2]):.4f}")

# 驱动轮接触点详情
contacts = r2.get_contacts(with_entity=ground)
n = len(contacts['position'])
print(f"[cg] 接触点: {n}")
for i in range(n):
    la, lb = int(contacts['link_a'][i]), int(contacts['link_b'][i])
    na = links[la].name if la < len(links) else f'link{la}'
    nb = links[lb].name if lb < len(links) else f'link{lb}'
    if 'wheel' not in na and 'wheel' not in nb:
        continue
    pos = np.asarray(to_np(contacts['position'][i])).reshape(-1)
    norm = np.asarray(to_np(contacts['normal'][i])).reshape(-1)
    fa = np.asarray(to_np(contacts['force_a'][i])).reshape(-1)
    fb = np.asarray(to_np(contacts['force_b'][i])).reshape(-1)
    pen = float(to_np(contacts['penetration'][i]))
    print(f"  {na} <-> {nb}")
    print(f"    pos={pos.round(4).tolist()} normal={norm.round(4).tolist()} pen={pen:.5f}")
    print(f"    force_a={fa.round(2).tolist()} force_b={fb.round(2).tolist()}")

# 轮子碰撞几何朝向: 打印 geom 信息
print("[cg] === 轮子 geom ===")
for g in scene.rigid_solver.geoms:
    if 'wheel' in g.name:
        print(f"  geom {g.name}: type={g.type} pos={to_np(g.get_pos()).round(4).tolist() if hasattr(g,'get_pos') else '?'}")
print("[cg] 完成")
