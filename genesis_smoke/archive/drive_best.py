#!/usr/bin/env python
"""最优组合: v6驱动增益 + InternUtopia位置增益 + caster阻尼 + neutral_collision"""
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
    sim_options=gs.options.SimOptions(dt=1/60.0, substeps=4),
    rigid_options=gs.options.RigidOptions(enable_neutral_collision=True),
    show_viewer=False,
)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=R2_URDF), surface=gs.surfaces.Default(),
                      material=gs.materials.Rigid(friction=5.0))
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default(),
                          material=gs.materials.Rigid(friction=5.0))
scene.build()
print("[db] build 完成")

wl = flat(r2.get_joint('wheel_left_joint').dofs_idx_local).astype(int).tolist()
wr = flat(r2.get_joint('wheel_right_joint').dofs_idx_local).astype(int).tolist()
head = flat(r2.get_joint('head_joint').dofs_idx_local).astype(int).tolist()
lift = flat(r2.get_joint('lift_joint').dofs_idx_local).astype(int).tolist()
waist = flat(r2.get_joint('waist_joint').dofs_idx_local).astype(int).tolist()
# caster 关节
caster_steer = []; caster_wheel = []
for j in r2.joints:
    if 'caster' in j.name:
        d = flat(j.dofs_idx_local).astype(int).tolist()
        (caster_steer if 'steer' in j.name else caster_wheel).extend(d)

# 增益: 轮子 v6 (kp=1000 kv=100); caster 阻尼; 位置关节 kp=2000 kv=100
r2.set_dofs_kp(1000.0, wl+wr); r2.set_dofs_kv(100.0, wl+wr)
if caster_steer: r2.set_dofs_kp(0.0, caster_steer); r2.set_dofs_kv(1000.0, caster_steer)
if caster_wheel: r2.set_dofs_kp(0.0, caster_wheel); r2.set_dofs_kv(100.0, caster_wheel)
r2.set_dofs_kp(2000.0, head+lift+waist); r2.set_dofs_kv(100.0, head+lift+waist)

for _ in range(100):
    scene.step()
print(f"[db] settle: base={to_np(r2.get_pos()).round(3).tolist()}")

# 前进
r2.set_dofs_velocity([5.0, 5.0], wl+wr)
for i in range(120):
    scene.step()
    if i % 40 == 0:
        print(f"[db] 前进 step {i}: base={to_np(r2.get_pos()).round(3).tolist()}")
r2.set_dofs_velocity([0.0, 0.0], wl+wr)
print(f"[db] 前进结束: base={to_np(r2.get_pos()).round(3).tolist()}")

# 转向
for _ in range(30):
    scene.step()
r2.set_dofs_velocity([5.0, -5.0], wl+wr)
for i in range(120):
    scene.step()
    if i % 40 == 0:
        q = flat(r2.get_quat()); yaw = 2*np.arctan2(q[3], q[0])
        print(f"[db] 转向 step {i}: yaw={yaw:.3f}")
r2.set_dofs_velocity([0.0, 0.0], wl+wr)
q = flat(r2.get_quat()); yaw = 2*np.arctan2(q[3], q[0])
print(f"[db] 转向结束: yaw={yaw:.3f} rad")

# 头部/升降/腰部
r2.set_dofs_position([0.4], head)
for _ in range(60):
    scene.step()
print(f"[db] 头部: pos={flat(r2.get_dofs_position(head)).round(3).tolist()} (应≈0.4)")
r2.set_dofs_position([0.3], lift)
for _ in range(60):
    scene.step()
print(f"[db] 升降: pos={flat(r2.get_dofs_position(lift)).round(3).tolist()} (应≈0.3)")
r2.set_dofs_position([0.5], waist)
for _ in range(60):
    scene.step()
print(f"[db] 腰部: pos={flat(r2.get_dofs_position(waist)).round(3).tolist()} (应≈0.5)")
print("[db] 完成")
