#!/usr/bin/env python
"""着地诊断: 各轮子 link 高度 + 接触力, 验证驱动轮是否悬空 (caster 撑起)"""
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
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=1/60.0, substeps=10),
    rigid_options=gs.options.RigidOptions(noslip_iterations=5, enable_self_collision=False),
    show_viewer=False,
)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=R2_URDF), surface=gs.surfaces.Default(),
                      material=gs.materials.Rigid(friction=5.0))
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default(),
                          material=gs.materials.Rigid(friction=5.0))
scene.build()
print("[gd] build 完成")

for _ in range(150):
    scene.step()
print(f"[gd] settle 后 base z={to_np(r2.get_pos()).reshape(-1)[2]:.4f}")

# 各轮子 link 高度 (轮子底部 = z - 半径)
WHEEL_LINKS = {
    'wheel_left_Link':        0.085,
    'wheel_right_Link':       0.085,
    'caster_front_left_wheel_Link':  0.05,
    'caster_front_right_wheel_Link': 0.05,
    'caster_rear_left_wheel_Link':   0.05,
    'caster_rear_right_wheel_Link':  0.05,
}
links = scene.rigid_solver.links
name2idx = {l.name: i for i, l in enumerate(links)}
for name, r in WHEEL_LINKS.items():
    li = name2idx[name]
    z = float(flat(links[li].get_pos())[2])
    print(f"[gd] {name}: link_z={z:.4f}  轮底={z-r:.4f}  着地={'是' if abs(z-r) < 0.012 else '否!'}")

# 接触力
print("[gd] === 与地面接触 ===")
contacts = r2.get_contacts(with_entity=ground)
n = len(contacts['position'])
print(f"[gd] 接触点总数: {n}")
for i in range(n):
    la, lb = int(contacts['link_a'][i]), int(contacts['link_b'][i])
    fa = to_np(contacts['force_a'][i])
    name_a = scene.rigid_solver.links[la].name if la < len(scene.rigid_solver.links) else f'link{la}'
    name_b = scene.rigid_solver.links[lb].name if lb < len(scene.rigid_solver.links) else f'link{lb}'
    f = np.asarray(fa).reshape(-1)
    fz = f[2] if f.size >= 3 else 0.0
    print(f"  {name_a} <-> {name_b}  fz={fz:.1f}  f={f.round(1).tolist()}")

# 汇总: 驱动轮 vs caster 接触力
print("[gd] === link 净接触力 ===")
force = to_np(r2.get_links_net_contact_force())
for name in WHEEL_LINKS:
    li = name2idx[name]
    f = np.asarray(force[li]).reshape(-1)
    print(f"  {name}: 净力={f.round(1).tolist()}")

print("[gd] 完成")
