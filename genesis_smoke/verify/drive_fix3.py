#!/usr/bin/env python
"""速度控制 + substeps=4 + 无 caster 阻尼: 前进 + 转向测试"""
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
print("[sv2] build 完成")

wl, wr = jidx('wheel_left_joint'), jidx('wheel_right_joint')
r2.set_dofs_kp(1000.0, wl+wr); r2.set_dofs_kv(100.0, wl+wr)
for _ in range(100): scene.step()
print(f"[sv2] settle 后 base={to_np(r2.get_pos()).round(3).tolist()}")

# === 前进: 左+右- (右轮轴 -y, 必须取反) ===
p0 = to_np(r2.get_pos()).reshape(-1)[:2].copy()
for i in range(120):
    r2.set_dofs_velocity([5.0, -5.0], wl + wr)
    scene.step()
    if i % 30 == 0:
        print(f"[sv2] 前进 step {i}: base={to_np(r2.get_pos()).round(3).tolist()}")
r2.set_dofs_velocity([0.0, 0.0], wl + wr)
p1 = to_np(r2.get_pos()).reshape(-1)[:2]
print(f"[sv2] 前进位移 dx={p1[0]-p0[0]:.3f} dy={p1[1]-p0[1]:.3f}")

# === 转向: 左+右+ (差速) ===
for _ in range(30): scene.step()
q = flat(r2.get_quat()); yaw0 = 2*np.arctan2(q[3], q[0])
for i in range(120):
    r2.set_dofs_velocity([5.0, 5.0], wl + wr)
    scene.step()
    if i % 30 == 0:
        q = flat(r2.get_quat()); yaw = 2*np.arctan2(q[3], q[0])
        print(f"[sv2] 转向 step {i}: yaw={yaw:.3f} base={to_np(r2.get_pos()).round(3).tolist()}")
r2.set_dofs_velocity([0.0, 0.0], wl + wr)
q = flat(r2.get_quat()); yaw = 2*np.arctan2(q[3], q[0])
print(f"[sv2] 转向 yaw={yaw:.3f} (Δ={yaw-yaw0:.3f})")
print("[sv2] 完成")
