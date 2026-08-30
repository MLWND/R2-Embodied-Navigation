#!/usr/bin/env python
"""综合修复测试:
1. 驱动轮 z=0.085 (已在 URDF 修正, 轮底归零)
2. J2 抬平手臂 (J5 前臂 0 位姿插地 10.8cm, 必须抬臂)
3. caster steer 阻尼 kd=1000 (官方参考, 防抖)
4. 验证前进 + 转向
"""
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
        noslip_iterations=5,
        enable_self_collision=False,
    ),
    show_viewer=False,
)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=R2_URDF), surface=gs.surfaces.Default(),
                      material=gs.materials.Rigid(friction=5.0))
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default(),
                          material=gs.materials.Rigid(friction=5.0))
scene.build()
print("[fx] build 完成")

wl, wr = jidx('wheel_left_joint'), jidx('wheel_right_joint')
# 抬臂: J2 左右 = ±90° (前臂 0 位姿插地 10.8cm)
j2l, j2r = jidx('J2_left_joint'), jidx('J2_right_joint')
# caster steer 阻尼防抖
steers = []
for s in ['front_left', 'front_right', 'rear_left', 'rear_right']:
    steers += jidx(f'caster_{s}_steer_joint')
steers = list(set(steers))

r2.set_dofs_kp(3000.0, j2l + j2r); r2.set_dofs_kv(200.0, j2l + j2r)
r2.set_dofs_kp(0.0, steers); r2.set_dofs_kv(1000.0, steers)   # caster 阻尼
r2.set_dofs_kp(1000.0, wl+wr); r2.set_dofs_kv(100.0, wl+wr)   # 驱动轮速度控制

# settle + 抬臂 (ramp)
for i in range(150):
    t = (i+1)/150
    r2.set_dofs_position([1.57*t, 1.57*t], j2l + j2r)  # 注意: 左右同名关节一起设置
    scene.step()
print(f"[fx] settle 后 base={to_np(r2.get_pos()).round(3).tolist()}")

# 接触检查: J5 是否还擦地
contacts = r2.get_contacts(with_entity=ground)
names = [f"{scene.rigid_solver.links[int(contacts['link_a'][i])].name}" for i in range(len(contacts['position']))]
print(f"[fx] 接地 link: {sorted(set(names))}")

# === 前进: 轮速 +5 (观察方向符号) ===
p0 = to_np(r2.get_pos()).reshape(-1)[:2].copy()
r2.set_dofs_velocity([5.0, 5.0], wl + wr)
for i in range(150):
    scene.step()
    if i % 50 == 0:
        print(f"[fx] 前进 step {i}: base={to_np(r2.get_pos()).round(3).tolist()}")
r2.set_dofs_velocity([0.0, 0.0], wl + wr)
p1 = to_np(r2.get_pos()).reshape(-1)[:2]
print(f"[fx] 前进位移 dx={p1[0]-p0[0]:.3f} dy={p1[1]-p0[1]:.3f} (x 应显著增大)")

# === 转向: 左+右- ===
for _ in range(30): scene.step()
r2.set_dofs_velocity([5.0, -5.0], wl + wr)
for i in range(150):
    scene.step()
    if i % 50 == 0:
        q = flat(r2.get_quat()); yaw = 2*np.arctan2(q[3], q[0])
        print(f"[fx] 转向 step {i}: yaw={yaw:.3f}")
r2.set_dofs_velocity([0.0, 0.0], wl + wr)
q = flat(r2.get_quat()); yaw = 2*np.arctan2(q[3], q[0])
print(f"[fx] 转向 yaw={yaw:.3f} rad")
print("[fx] 完成")
