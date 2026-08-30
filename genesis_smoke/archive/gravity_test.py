#!/usr/bin/env python
"""重力补偿测试: gravity_compensation=1.0 + ramp + 适中增益"""
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
r2 = scene.add_entity(
    morph=gs.morphs.URDF(file=R2_URDF),
    surface=gs.surfaces.Default(),
    material=gs.materials.Rigid(friction=5.0, gravity_compensation=1.0),  # 重力补偿
)
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default(),
                          material=gs.materials.Rigid(friction=5.0))
scene.build()
print("[gc] build 完成")

def jidx(name):
    return flat(r2.get_joint(name).dofs_idx_local).astype(int).tolist()

head, lift, waist = jidx('head_joint'), jidx('lift_joint'), jidx('waist_joint')
arm_l = [jidx(f'J{i}_left_joint') for i in range(1,8)]

pos_joints = head + lift + waist + [d[0] for d in arm_l]
r2.set_dofs_kp(500.0, pos_joints)
r2.set_dofs_kv(50.0, pos_joints)

for _ in range(50):
    scene.step()

def ramp_to(dofs, target, steps=100, hold=30):
    cur = flat(r2.get_dofs_position(dofs))
    start = cur.copy()
    for i in range(steps):
        t = start + (np.array(target) - start) * ((i+1)/steps)
        r2.set_dofs_position(t.tolist(), dofs)
        scene.step()
    for _ in range(hold):
        scene.step()
    return flat(r2.get_dofs_position(dofs)).round(3).tolist()

print(f"[gc] 升降: {ramp_to(lift, [0.3])} (应≈0.3)")
print(f"[gc] 腰部: {ramp_to(waist, [0.5])} (应≈0.5)")
print(f"[gc] 头部: {ramp_to(head, [0.4])} (应≈0.4)")
for i, d in enumerate(arm_l):
    r = ramp_to(d, [0.5], steps=60)
    print(f"[gc] 左臂 J{i+1}: {r} (应≈0.5)")
print("[gc] 完成")
