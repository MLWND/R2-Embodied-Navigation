#!/usr/bin/env python
"""综合修复 v2: 修 J2 右臂符号 + 接触打印 + 转向锁死诊断 (监控轮子转动)"""
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
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=1/60.0, substeps=10),
    rigid_options=gs.options.RigidOptions(
        enable_self_collision=False,
    ),
    show_viewer=False,
)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=R2_URDF), surface=gs.surfaces.Default(),
                      material=gs.materials.Rigid(friction=5.0))
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default(),
                          material=gs.materials.Rigid(friction=5.0))
scene.build()
print("[fx2] build 完成")

wl, wr = jidx('wheel_left_joint'), jidx('wheel_right_joint')
j2l, j2r = jidx('J2_left_joint'), jidx('J2_right_joint')
steers = []
for s in ['front_left', 'front_right', 'rear_left', 'rear_right']:
    steers += jidx(f'caster_{s}_steer_joint')
steers = list(set(steers))

r2.set_dofs_kp(3000.0, j2l + j2r); r2.set_dofs_kv(200.0, j2l + j2r)
r2.set_dofs_kp(0.0, steers); r2.set_dofs_kv(1000.0, steers)
r2.set_dofs_kp(1000.0, wl+wr); r2.set_dofs_kv(100.0, wl+wr)

# settle + 抬臂 (J2_left=+1.57, J2_right=-1.57)
for i in range(150):
    t = (i+1)/150
    r2.set_dofs_position([1.57*t, -1.57*t], j2l + j2r)
    scene.step()
print(f"[fx2] settle 后 base={to_np(r2.get_pos()).round(3).tolist()}")

# 接触检查 (正确: link_b 是机器人侧)
contacts = r2.get_contacts(with_entity=ground)
names = [scene.rigid_solver.links[int(contacts['link_b'][i])].name for i in range(len(contacts['position']))]
from collections import Counter
print(f"[fx2] 接地 link: {dict(Counter(names))}")

# === 转向: 左+右- , 监控轮子 ===
q = flat(r2.get_quat()); yaw0 = 2*np.arctan2(q[3], q[0])
print(f"[fx2] 转向前 yaw={yaw0:.3f} 轮角={flat(r2.get_dofs_position(wl+wr)).round(2).tolist()}")
r2.set_dofs_velocity([5.0, -5.0], wl + wr)
for i in range(200):
    scene.step()
    if i % 20 == 0:
        q = flat(r2.get_quat()); yaw = 2*np.arctan2(q[3], q[0])
        wpos = flat(r2.get_dofs_position(wl+wr)).round(2).tolist()
        print(f"[fx2] 转向 step {i}: yaw={yaw:.3f} 轮角={wpos} base={to_np(r2.get_pos()).round(3).tolist()}")
r2.set_dofs_velocity([0.0, 0.0], wl + wr)
q = flat(r2.get_quat()); yaw = 2*np.arctan2(q[3], q[0])
print(f"[fx2] 转向结束 yaw={yaw:.3f} (Δ={yaw-yaw0:.3f})")
print("[fx2] 完成")
