#!/usr/bin/env python
"""无重力补偿: 高kp强PD + ramp 抗重力, 保持驱动"""
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
                      material=gs.materials.Rigid(friction=5.0))  # 无重力补偿
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default(),
                          material=gs.materials.Rigid(friction=5.0))
scene.build()
print("[ng] build 完成")

def jidx(name):
    return flat(r2.get_joint(name).dofs_idx_local).astype(int).tolist()

wl, wr = jidx('wheel_left_joint'), jidx('wheel_right_joint')
head, lift, waist = jidx('head_joint'), jidx('lift_joint'), jidx('waist_joint')
arm_l = [jidx(f'J{i}_left_joint') for i in range(1,8)]

r2.set_dofs_kp(1000.0, wl+wr); r2.set_dofs_kv(100.0, wl+wr)
pos_joints = head + lift + waist + [d[0] for d in arm_l]
r2.set_dofs_kp(3000.0, pos_joints); r2.set_dofs_kv(300.0, pos_joints)

for _ in range(50):
    scene.step()

# 前进
r2.set_dofs_velocity([5.0, 5.0], wl+wr)
for _ in range(120):
    scene.step()
r2.set_dofs_velocity([0.0, 0.0], wl+wr)
print(f"[ng] 前进: base={to_np(r2.get_pos()).round(3).tolist()}")

# 转向
for _ in range(30):
    scene.step()
r2.set_dofs_velocity([5.0, -5.0], wl+wr)
for _ in range(120):
    scene.step()
r2.set_dofs_velocity([0.0, 0.0], wl+wr)
q = flat(r2.get_quat()); yaw = 2*np.arctan2(q[3], q[0])
print(f"[ng] 转向: yaw={yaw:.3f} rad")

# 关节 (ramp)
def ramp_to(dofs, target, steps=80, hold=20):
    cur = flat(r2.get_dofs_position(dofs)); start = cur.copy()
    for i in range(steps):
        t = start + (np.array(target) - start) * ((i+1)/steps)
        r2.set_dofs_position(t.tolist(), dofs)
        scene.step()
    for _ in range(hold):
        scene.step()
    return flat(r2.get_dofs_position(dofs)).round(3).tolist()

print(f"[ng] 升降: {ramp_to(lift, [0.3])}")
print(f"[ng] 腰部: {ramp_to(waist, [0.5])}")
print(f"[ng] 头部: {ramp_to(head, [0.4])}")
for i, d in enumerate(arm_l):
    ramp_to(d, [0.5], steps=40)
print(f"[ng] 左臂: {[round(float(flat(r2.get_dofs_position(d))[0]),2) for d in arm_l]}")
print("[ng] 完成")
