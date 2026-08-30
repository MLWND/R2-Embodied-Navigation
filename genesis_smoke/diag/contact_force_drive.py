#!/usr/bin/env python
"""驱动时轮子接触力诊断: 位置控制转轮子, 监控切向力是否随驱动增大"""
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
print("[cf] build 完成")

wl, wr = jidx('wheel_left_joint'), jidx('wheel_right_joint')
r2.set_dofs_kp(50.0, wl+wr); r2.set_dofs_kv(5.0, wl+wr)
for _ in range(100): scene.step()

links = scene.rigid_solver.links
name2idx = {l.name: i for i, l in enumerate(links)}
wli = name2idx['wheel_left_Link']

def wheel_contact_force():
    """返回左轮所有接触点的合力"""
    c = r2.get_contacts(with_entity=ground)
    fsum = np.zeros(3)
    for i in range(len(c['position'])):
        la, lb = int(c['link_a'][i]), int(c['link_b'][i])
        if links[la].name == 'wheel_left_Link':
            fsum += np.asarray(to_np(c['force_a'][i])).reshape(-1)
        elif links[lb].name == 'wheel_left_Link':
            fsum += np.asarray(to_np(c['force_b'][i])).reshape(-1)
    return fsum

# 驱动前
print(f"[cf] 驱动前: base={to_np(r2.get_pos()).round(3).tolist()} 左轮接触力={wheel_contact_force().round(1).tolist()}")

# 位置控制转轮子 (2圈)
target = 12.57
for i in range(200):
    r2.set_dofs_position([target*(i+1)/200, target*(i+1)/200], wl + wr)
    scene.step()
    if i % 40 == 0:
        f = wheel_contact_force()
        p = to_np(r2.get_pos()).reshape(-1)
        wpos = flat(r2.get_dofs_position(wl+wr))
        print(f"[cf] step {i}: base_x={p[0]:.3f} 轮角={wpos.round(2).tolist()} 左轮接触力={f.round(1).tolist()} (fz={f[2]:.0f} fx={f[0]:.1f})")

print("[cf] 完成")
