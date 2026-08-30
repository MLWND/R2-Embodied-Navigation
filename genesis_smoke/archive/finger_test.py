#!/usr/bin/env python
"""验证 6 指手爪: 加载 + 手指关节控制"""
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
r2 = scene.add_entity(morph=gs.morphs.URDF(file=R2_URDF), surface=gs.surfaces.Default(),
                      material=gs.materials.Rigid(friction=5.0, gravity_compensation=1.0))
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default(),
                          material=gs.materials.Rigid(friction=5.0))
scene.build()
print(f"[fg] build 完成, n_dofs={r2.n_dofs}")

# 手指关节
finger_joints = [j.name for j in r2.joints if 'thumb' in j.name or j.name in
                 ('index_left_joint','middle_left_joint','ring_left_joint','pinky_left_joint')]
print(f"[fg] 手指关节: {finger_joints}")

# 测试左手指: 全部弯曲到 0.5
dofs = []
for name in finger_joints:
    d = flat(r2.get_joint(name).dofs_idx_local).astype(int).tolist()
    dofs.extend(d)
r2.set_dofs_kp(100.0, dofs); r2.set_dofs_kv(10.0, dofs)
r2.set_dofs_position([0.5]*len(dofs), dofs)
for _ in range(60):
    scene.step()
pos = flat(r2.get_dofs_position(dofs)).round(3).tolist()
print(f"[fg] 手指位置: {pos} (应≈0.5)")
print("[fg] 完成")
