#!/usr/bin/env python
"""驱动修复: noslip + 小dt + 删底座碰撞 (参考 Genesis issue #1610)"""
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
    sim_options=gs.options.SimOptions(dt=1/60.0, substeps=10),  # 小 dt
    rigid_options=gs.options.RigidOptions(
        noslip_iterations=5,          # 抑制打滑
        enable_self_collision=False,  # 关自碰撞
    ),
    show_viewer=False,
)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=R2_URDF), surface=gs.surfaces.Default(),
                      material=gs.materials.Rigid(friction=5.0))
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default(),
                          material=gs.materials.Rigid(friction=5.0))
scene.build()
print("[ns] build 完成")

def jidx(name):
    return flat(r2.get_joint(name).dofs_idx_local).astype(int).tolist()

wl, wr = jidx('wheel_left_joint'), jidx('wheel_right_joint')
r2.set_dofs_kp(1000.0, wl+wr); r2.set_dofs_kv(100.0, wl+wr)

for _ in range(50):
    scene.step()
print(f"[ns] settle: base={to_np(r2.get_pos()).round(3).tolist()}")

# 前进
r2.set_dofs_velocity([5.0, 5.0], wl+wr)
for i in range(120):
    scene.step()
    if i % 40 == 0:
        print(f"[ns] 前进 step {i}: base={to_np(r2.get_pos()).round(3).tolist()}")
r2.set_dofs_velocity([0.0, 0.0], wl+wr)
print(f"[ns] 前进结束: base={to_np(r2.get_pos()).round(3).tolist()}")

# 转向
for _ in range(30):
    scene.step()
r2.set_dofs_velocity([5.0, -5.0], wl+wr)
for i in range(120):
    scene.step()
    if i % 40 == 0:
        q = flat(r2.get_quat()); yaw = 2*np.arctan2(q[3], q[0])
        print(f"[ns] 转向 step {i}: yaw={yaw:.3f}")
r2.set_dofs_velocity([0.0, 0.0], wl+wr)
q = flat(r2.get_quat()); yaw = 2*np.arctan2(q[3], q[0])
print(f"[ns] 转向结束: yaw={yaw:.3f} rad")
print("[ns] 完成")
