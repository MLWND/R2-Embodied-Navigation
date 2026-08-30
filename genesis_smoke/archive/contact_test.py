#!/usr/bin/env python
"""接触诊断: 轮子是否接触地面 + 接触力 + 驱动时力变化"""
import numpy as np
import genesis as gs

R2_URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf"

def to_np(x):
    if hasattr(x, "cpu"):
        x = x.cpu()
    if isinstance(x, (tuple, list)):
        return [to_np(i) for i in x]
    return np.asarray(x)

def flat(x):
    parts = []
    for i in to_np(x):
        parts.append(np.asarray(i).reshape(-1))
    return np.concatenate(parts)

gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/60.0, substeps=4), show_viewer=False)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=R2_URDF), surface=gs.surfaces.Default())
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0.0, 0.0, 0.0)), surface=gs.surfaces.Default())
scene.build()
print("[contact] build 完成")

wl = flat(r2.get_joint('wheel_left_joint').dofs_idx_local).astype(int).tolist()
wr = flat(r2.get_joint('wheel_right_joint').dofs_idx_local).astype(int).tolist()

for _ in range(100):
    scene.step()
print(f"[contact] settle 后 base z={to_np(r2.get_pos()).reshape(-1)[2]:.4f}")

# 接触检查
contacts = r2.get_contacts(with_entity=ground)
n = len(contacts['position'])
print(f"[contact] 与地面的接触数: {n}")
for i in range(min(n, 12)):
    la, lb = int(contacts['link_a'][i]), int(contacts['link_b'][i])
    fa = to_np(contacts['force_a'][i])
    name_a = scene.rigid_solver.links[la].name if la < len(scene.rigid_solver.links) else f'link{la}'
    name_b = scene.rigid_solver.links[lb].name if lb < len(scene.rigid_solver.links) else f'link{lb}'
    print(f"  {name_a} <-> {name_b}  force_a={np.asarray(fa).round(1).tolist()}")

# 轮子接触力
force = to_np(r2.get_links_net_contact_force())
print(f"[contact] 各 link 接触力(前8): {np.asarray(force).reshape(-1)[:8].round(2).tolist()}")

# 驱动时接触力变化
r2.set_dofs_kp(50.0); r2.set_dofs_kv(5.0)
for i in range(100):
    r2.set_dofs_position([3.0*(i+1)/100, 3.0*(i+1)/100], wl + wr)
    scene.step()
    if i % 25 == 0:
        p = to_np(r2.get_pos()).round(3).tolist()
        f = np.asarray(to_np(r2.get_links_net_contact_force())).reshape(-1)[:6].round(1).tolist()
        print(f"[contact] step {i}: base={p} 接触力(前6)={f}")
print("[contact] 完成")
