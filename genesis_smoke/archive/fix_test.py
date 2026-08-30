#!/usr/bin/env python
"""修复测试: 转向(强差速) + 头部(高力限) + 位置控制(温和增益)"""
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
print("[fx] build 完成")

wl = flat(r2.get_joint('wheel_left_joint').dofs_idx_local).astype(int).tolist()
wr = flat(r2.get_joint('wheel_right_joint').dofs_idx_local).astype(int).tolist()
head = flat(r2.get_joint('head_joint').dofs_idx_local).astype(int).tolist()
lift = flat(r2.get_joint('lift_joint').dofs_idx_local).astype(int).tolist()
waist = flat(r2.get_joint('waist_joint').dofs_idx_local).astype(int).tolist()

for _ in range(100):
    scene.step()

# ===== 1. 转向: 强差速 [10,-10] =====
r2.set_dofs_kp(1000.0, wl+wr); r2.set_dofs_kv(100.0, wl+wr)
r2.set_dofs_velocity([10.0, -10.0], wl+wr)
for i in range(150):
    scene.step()
    if i % 50 == 0:
        q = flat(r2.get_quat())
        yaw = 2*np.arctan2(q[3], q[0])
        v = flat(r2.get_dofs_velocity(wl+wr)).round(2).tolist()
        print(f"[fx] 转向 step {i}: yaw={yaw:.3f} 轮速={v}")
r2.set_dofs_velocity([0.0, 0.0], wl+wr)
q = flat(r2.get_quat()); yaw = 2*np.arctan2(q[3], q[0])
print(f"[fx] 转向结束: yaw={yaw:.3f} rad (应明显≠0)")

# ===== 2. 头部: 力限500 + 高增益 =====
r2.set_dofs_force_range([-500.0, 500.0], head)
r2.set_dofs_kp(100.0, head); r2.set_dofs_kv(10.0, head)
r2.set_dofs_position([0.4], head)
for i in range(60):
    scene.step()
print(f"[fx] 头部: pos={flat(r2.get_dofs_position(head)).round(3).tolist()} (应≈0.4)")

# ===== 3. 升降/腰部: 温和增益 =====
r2.set_dofs_kp(50.0, lift+waist); r2.set_dofs_kv(15.0, lift+waist)
r2.set_dofs_position([0.3], lift)
for _ in range(60):
    scene.step()
print(f"[fx] 升降: pos={flat(r2.get_dofs_position(lift)).round(3).tolist()} (应≈0.3)")
r2.set_dofs_position([0.5], waist)
for _ in range(60):
    scene.step()
print(f"[fx] 腰部: pos={flat(r2.get_dofs_position(waist)).round(3).tolist()} (应≈0.5)")
print("[fx] 完成")
