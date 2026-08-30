#!/usr/bin/env python
"""打印接触点 friction 值"""
import numpy as np
import genesis as gs

URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf"

def to_np(x):
    if hasattr(x, "cpu"): x = x.cpu()
    if isinstance(x, (tuple, list)): return [to_np(i) for i in x]
    return np.asarray(x)

gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/60.0, substeps=4), show_viewer=False)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=URDF), surface=gs.surfaces.Default())
g = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default())
scene.build()
for _ in range(100): scene.step()

cs = scene.rigid_solver.collider._collider_state
n = int(cs.n_contacts[0])
print(f"接触数: {n}")
for i in range(n):
    fr = float(cs.contact_data.friction[i, 0])
    pen = float(cs.contact_data.penetration[i, 0])
    pos = np.asarray(to_np(cs.contact_data.pos[i, 0])).reshape(-1)
    print(f"  contact {i}: friction={fr:.3f} pen={pen:.6f} pos={pos.round(3).tolist()}")
print("完成")
