#!/usr/bin/env python
"""完整工作配置: 重力补偿 + 驱动 + 转向 + 全部关节 + 渲染"""
import os
import numpy as np
from PIL import Image
import genesis as gs

R2_URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf"
OUT = "/home/xujinlong/test/genesis_smoke/output"
os.makedirs(OUT, exist_ok=True)

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
    material=gs.materials.Rigid(friction=5.0, gravity_compensation=1.0),
)
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default(),
                          material=gs.materials.Rigid(friction=5.0))
cam = scene.add_camera(model="pinhole", res=(1280, 720), pos=(2.5, 1.5, 1.5), lookat=(0,0,0.5), fov=60)
scene.build()
print("[ok] build 完成, n_dofs=", r2.n_dofs)

def save(name):
    rgb, depth, seg, normal = cam.render(rgb=True, depth=True)
    Image.fromarray(np.asarray(rgb)).save(os.path.join(OUT, f"ok_{name}.png"))
    print(f"[ok] 已保存 ok_{name}.png")

def jidx(name):
    return flat(r2.get_joint(name).dofs_idx_local).astype(int).tolist()

wl, wr = jidx('wheel_left_joint'), jidx('wheel_right_joint')
head, lift, waist = jidx('head_joint'), jidx('lift_joint'), jidx('waist_joint')
arm_l = [jidx(f'J{i}_left_joint') for i in range(1,8)]
arm_r = [jidx(f'J{i}_right_joint') for i in range(1,8)]

# 增益: 轮子速度控制; 位置关节 kp=500 kd=50
r2.set_dofs_kp(1000.0, wl+wr); r2.set_dofs_kv(100.0, wl+wr)
pos_joints = head + lift + waist + [d[0] for d in arm_l] + [d[0] for d in arm_r]
r2.set_dofs_kp(500.0, pos_joints); r2.set_dofs_kv(50.0, pos_joints)

for _ in range(50):
    scene.step()
save("0_initial")

# ===== 1. 前进 =====
r2.set_dofs_velocity([5.0, 5.0], wl+wr)
for _ in range(120):
    scene.step()
r2.set_dofs_velocity([0.0, 0.0], wl+wr)
print(f"[ok] 前进: base={to_np(r2.get_pos()).round(3).tolist()}")
save("1_drive")

# ===== 2. 转向 =====
for _ in range(30):
    scene.step()
r2.set_dofs_velocity([5.0, -5.0], wl+wr)
for _ in range(120):
    scene.step()
r2.set_dofs_velocity([0.0, 0.0], wl+wr)
q = flat(r2.get_quat()); yaw = 2*np.arctan2(q[3], q[0])
print(f"[ok] 转向: yaw={yaw:.3f} rad")
save("2_turn")

# ===== 3. 升降/腰部/头部 (ramp) =====
def ramp_to(dofs, target, steps=80, hold=20):
    cur = flat(r2.get_dofs_position(dofs)); start = cur.copy()
    for i in range(steps):
        t = start + (np.array(target) - start) * ((i+1)/steps)
        r2.set_dofs_position(t.tolist(), dofs)
        scene.step()
    for _ in range(hold):
        scene.step()
    return flat(r2.get_dofs_position(dofs)).round(3).tolist()

print(f"[ok] 升降: {ramp_to(lift, [0.3])}")
print(f"[ok] 腰部: {ramp_to(waist, [0.5])}")
print(f"[ok] 头部: {ramp_to(head, [0.4])}")
save("3_upper")

# ===== 4. 双臂 =====
for i, d in enumerate(arm_l):
    ramp_to(d, [0.5], steps=40)
for i, d in enumerate(arm_r):
    ramp_to(d, [-0.5], steps=40)
print(f"[ok] 左臂: {[round(float(flat(r2.get_dofs_position(d))[0]),2) for d in arm_l]}")
print(f"[ok] 右臂: {[round(float(flat(r2.get_dofs_position(d))[0]),2) for d in arm_r]}")
save("4_arms")
print("[ok] 完成")
