#!/usr/bin/env python
"""速度控制时轮子实际角速度 + 接触力: 验证摩擦是否随速度产生"""
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

def jidx(name):
    return flat(r2.get_joint(name).dofs_idx_local).astype(int).tolist()

gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/60.0, substeps=4), show_viewer=False)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=R2_URDF), surface=gs.surfaces.Default())
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default())
scene.build()
print("[sv] build 完成")

wl, wr = jidx('wheel_left_joint'), jidx('wheel_right_joint')
r2.set_dofs_kp(1000.0, wl+wr); r2.set_dofs_kv(100.0, wl+wr)
for _ in range(100): scene.step()

links = scene.rigid_solver.links
name2idx = {l.name: i for i, l in enumerate(links)}
wli = name2idx['wheel_left_Link']

def wheel_contact_force():
    c = r2.get_contacts(with_entity=ground)
    fsum = np.zeros(3)
    for i in range(len(c['position'])):
        la, lb = int(c['link_a'][i]), int(c['link_b'][i])
        if links[la].name == 'wheel_left_Link':
            fsum += np.asarray(to_np(c['force_a'][i])).reshape(-1)
        elif links[lb].name == 'wheel_left_Link':
            fsum += np.asarray(to_np(c['force_b'][i])).reshape(-1)
    return fsum

# 速度控制
r2.set_dofs_velocity([5.0, 5.0], wl + wr)
for i in range(60):
    scene.step()
    if i % 10 == 0:
        v = flat(r2.get_dofs_velocity(wl+wr))
        p = to_np(r2.get_pos()).reshape(-1)
        f = wheel_contact_force()
        print(f"[sv] step {i}: 轮速={v.round(2).tolist()} base_x={p[0]:.3f} 左轮接触力={f.round(1).tolist()}")

r2.set_dofs_velocity([0.0, 0.0], wl + wr)
print("[sv] 完成")
