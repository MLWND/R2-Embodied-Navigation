#!/usr/bin/env python
"""轮子驱动测试: 前进更远 + 转向 90° + 掉头 180°, 验证位移和朝向
参考 R1 ros2 NavControl (command/x/y/radian/speed) -> Genesis 差速控制"""
import numpy as np
import genesis as gs

URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf"

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

def yaw_of():
    q = flat(r2.get_quat())
    return 2*np.arctan2(q[3], q[0])

gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/60.0, substeps=8), rigid_options=gs.options.RigidOptions(friction_cone=gs.friction_cone.elliptic, impratio=100, contact_resolution=gs.contact_resolution.signorini), show_viewer=False)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=URDF, pos=(0,0,-0.001)), surface=gs.surfaces.Default(),
                      material=gs.materials.Rigid(friction=1.5))
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default(),
                          material=gs.materials.Rigid(friction=1.5))
scene.build()
print(f"[drv] build 完成, n_dofs={r2.n_dofs}")

# 手臂/躯干 kp 保持 0 位姿; 轮子/万向轮自由
arm_dofs = []
for j in r2.joints:
    n = j.name
    if n not in ('wheel_left_joint', 'wheel_right_joint') and not any(k in n for k in ['thumb','index','middle','ring','pinky']):
        arm_dofs += jidx(n)
arm_dofs = list(set(arm_dofs))
r2.set_dofs_kp(5000.0, arm_dofs); r2.set_dofs_kv(200.0, arm_dofs)
wl, wr = jidx('wheel_left_joint'), jidx('wheel_right_joint')
r2.set_dofs_kp(1000.0, wl+wr); r2.set_dofs_kv(100.0, wl+wr)
# 万向轮: steer 高阻尼, wheel 中阻尼 (官方 r2.py)
for j in r2.joints:
    if 'caster' in j.name:
        idx = jidx(j.name)
        if 'steer' in j.name:
            r2.set_dofs_kp(0.0, idx); r2.set_dofs_kv(1000.0, idx)
        else:
            r2.set_dofs_kp(0.0, idx); r2.set_dofs_kv(100.0, idx)
j2l, j2r = jidx('J2_left_joint'), jidx('J2_right_joint')
r2.set_dofs_kp(3000.0, j2l + j2r); r2.set_dofs_kv(200.0, j2l + j2r)
for i in range(150):
    t = (i+1)/150
    r2.control_dofs_position([1.57*t, -1.57*t], j2l + j2r)
    scene.step()
print(f"[drv] settle: base={to_np(r2.get_pos()).reshape(-1).round(3).tolist()} yaw={yaw_of():.3f}")

# ===== 1. 前进 3 秒 (速度 3 rad/s) =====
p0 = to_np(r2.get_pos()).reshape(-1)[:2].copy()
yaw0 = yaw_of()
for i in range(180):
    r2.set_dofs_velocity([3.0, -3.0], wl + wr)
    scene.step()
r2.set_dofs_velocity([0.0, 0.0], wl + wr)
p1 = to_np(r2.get_pos()).reshape(-1)[:2]
disp = np.linalg.norm(p1 - p0)
print(f"[drv] 前进3s: 位移={disp:.3f} m, yaw变化={yaw_of()-yaw0:.3f} rad {'OK' if disp>0.3 else 'FAIL'}")

# ===== 2. 转向 90° (原地旋转) =====
yaw1 = yaw_of()
for i in range(120):
    r2.set_dofs_velocity([8.0, 8.0], wl + wr)
    scene.step()
r2.set_dofs_velocity([0.0, 0.0], wl + wr)
yaw2 = yaw_of()
d1 = yaw2 - yaw1
print(f"[drv] 转向90°: yaw变化={d1:.3f} rad ({np.degrees(d1):.1f}°) {'OK' if abs(d1)>0.5 else 'FAIL'}")

# ===== 3. 掉头 180° (再转 90°) =====
for i in range(120):
    r2.set_dofs_velocity([8.0, 8.0], wl + wr)
    scene.step()
r2.set_dofs_velocity([0.0, 0.0], wl + wr)
yaw3 = yaw_of()
d2 = yaw3 - yaw2
total = yaw3 - yaw1
print(f"[drv] 掉头: 再转={np.degrees(d2):.1f}°, 总转向={np.degrees(total):.1f}° {'OK' if abs(total)>2.5 else 'FAIL'}")

# ===== 4. 掉头后前进 (验证朝向正确) =====
p2 = to_np(r2.get_pos()).reshape(-1)[:2].copy()
for i in range(120):
    r2.set_dofs_velocity([3.0, -3.0], wl + wr)
    scene.step()
r2.set_dofs_velocity([0.0, 0.0], wl + wr)
p3 = to_np(r2.get_pos()).reshape(-1)[:2]
disp2 = np.linalg.norm(p3 - p2)
# 前进方向应沿 yaw3 方向
move_dir = np.arctan2(p3[1]-p2[1], p3[0]-p2[0])
print(f"[drv] 掉头后前进: 位移={disp2:.3f} m, 方向={np.degrees(move_dir):.1f}°, 朝向={np.degrees(yaw3):.1f}°")
print(f"[drv] 最终: base={to_np(r2.get_pos()).reshape(-1).round(3).tolist()} yaw={np.degrees(yaw3):.1f}°")
print("[drv] 完成")
