#!/usr/bin/env python
"""最终诊断: 驱动时轮子接触力的切向分量"""
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
print("[cf] build 完成")
wl = flat(r2.get_joint('wheel_left_joint').dofs_idx_local).astype(int).tolist()
wr = flat(r2.get_joint('wheel_right_joint').dofs_idx_local).astype(int).tolist()

for _ in range(100):
    scene.step()

r2.set_dofs_kp(50.0); r2.set_dofs_kv(5.0)
target = -18.85
for i in range(100):
    r2.set_dofs_position([target*(i+1)/100, target*(i+1)/100], wl+wr)
    scene.step()
    if i == 50:
        contacts = r2.get_contacts(with_entity=ground)
        print(f"[cf] step {i}: 接触数={len(contacts['position'])}")
        for k in range(len(contacts['position'])):
            la, lb = int(contacts['link_a'][k]), int(contacts['link_b'][k])
            name_a = scene.rigid_solver.links[la].name
            fa = to_np(contacts['force_a'][k])
            fa = np.asarray(fa).reshape(-1)
            if len(fa) >= 3:
                print(f"  {name_a} force=[{fa[0]:.1f}, {fa[1]:.1f}, {fa[2]:.1f}]  (x,y,z)")
        p = to_np(r2.get_pos()).round(3).tolist()
        print(f"[cf] base pos={p}")
print("[cf] 完成")
