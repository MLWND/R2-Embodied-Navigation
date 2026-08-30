#!/usr/bin/env python
"""验证三对轮子都能独立转动"""
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
                      material=gs.materials.Rigid(friction=5.0))
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default(),
                          material=gs.materials.Rigid(friction=5.0))
scene.build()
print("[wr] build 完成")

# 关节分组
def jidx(name):
    return flat(r2.get_joint(name).dofs_idx_local).astype(int).tolist()

drive = jidx('wheel_left_joint') + jidx('wheel_right_joint')
steer = [jidx(f'caster_{pos}_{side}_steer_joint') for pos in ['front','rear'] for side in ['left','right']]
steer = [d[0] for d in steer]
cwheel = [jidx(f'caster_{pos}_{side}_wheel_joint') for pos in ['front','rear'] for side in ['left','right']]
cwheel = [d[0] for d in cwheel]

print(f"[wr] 驱动轮 DOF: {drive}")
print(f"[wr] caster steer DOF: {steer}")
print(f"[wr] caster wheel DOF: {cwheel}")

for _ in range(50):
    scene.step()

# 1. 驱动轮转动
r2.set_dofs_kp(1000.0, drive); r2.set_dofs_kv(100.0, drive)
r2.set_dofs_velocity([5.0, 5.0], drive)
for _ in range(30):
    scene.step()
a0 = flat(r2.get_dofs_position(drive)).round(3).tolist()
print(f"[wr] 驱动轮角度(30步后): {a0} (应≠0, 说明在转)")
r2.set_dofs_velocity([0.0, 0.0], drive)

# 2. caster steer 转动 (绕z)
r2.set_dofs_kp(100.0, steer); r2.set_dofs_kv(10.0, steer)
r2.set_dofs_position([0.5]*len(steer), steer)
for _ in range(30):
    scene.step()
a1 = flat(r2.get_dofs_position(steer)).round(3).tolist()
print(f"[wr] caster steer 角度: {a1} (应≈0.5, 说明绕z能转)")

# 3. caster wheel 转动 (绕y)
r2.set_dofs_kp(100.0, cwheel); r2.set_dofs_kv(10.0, cwheel)
r2.set_dofs_position([3.0]*len(cwheel), cwheel)
for _ in range(30):
    scene.step()
a2 = flat(r2.get_dofs_position(cwheel)).round(3).tolist()
print(f"[wr] caster wheel 角度: {a2} (应≈3.0, 说明滚动能转)")
print("[wr] 完成")
