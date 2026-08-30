#!/usr/bin/env python
"""完整机器人调试: 驱动 + 转向 + 头部 + 手臂 (隔离场景, 摩擦5.0)"""
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

def yaw_of(q):
    return 2 * np.arctan2(q[3], q[0])

gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/60.0, substeps=4), show_viewer=False)
r2 = scene.add_entity(
    morph=gs.morphs.URDF(file=R2_URDF),
    surface=gs.surfaces.Default(),
    material=gs.materials.Rigid(friction=5.0),
)
ground = scene.add_entity(
    morph=gs.morphs.Plane(pos=(0,0,0)),
    surface=gs.surfaces.Default(),
    material=gs.materials.Rigid(friction=5.0),
)
cam = scene.add_camera(model="pinhole", res=(1280, 720), pos=(2.5, 1.5, 1.5), lookat=(0,0,0.5), fov=60)
scene.build()
print("[rt] build 完成, n_dofs=", r2.n_dofs)

def save(name):
    rgb, depth, seg, normal = cam.render(rgb=True, depth=True)
    Image.fromarray(np.asarray(rgb)).save(os.path.join(OUT, f"rt_{name}.png"))
    print(f"[rt] 已保存 rt_{name}.png")

wl = flat(r2.get_joint('wheel_left_joint').dofs_idx_local).astype(int).tolist()
wr = flat(r2.get_joint('wheel_right_joint').dofs_idx_local).astype(int).tolist()
head = flat(r2.get_joint('head_joint').dofs_idx_local).astype(int).tolist()
lift = flat(r2.get_joint('lift_joint').dofs_idx_local).astype(int).tolist()
waist = flat(r2.get_joint('waist_joint').dofs_idx_local).astype(int).tolist()

for _ in range(100):
    scene.step()
print(f"[rt] settle: base={to_np(r2.get_pos()).round(3).tolist()}")
save("0_initial")

# ===== 1. 前进 =====
r2.set_dofs_kp(1000.0, wl + wr); r2.set_dofs_kv(100.0, wl + wr)
r2.set_dofs_velocity([5.0, 5.0], wl + wr)
for _ in range(120):
    scene.step()
p = to_np(r2.get_pos()).round(3).tolist()
print(f"[rt] 前进: base={p} (x 应增大)")
r2.set_dofs_velocity([0.0, 0.0], wl + wr)
save("1_drive")

# ===== 2. 转向: 左轮前右轮后 (差速) =====
for _ in range(30):
    scene.step()
r2.set_dofs_velocity([5.0, -5.0], wl + wr)
for _ in range(120):
    scene.step()
q = flat(r2.get_quat())
yaw = yaw_of(q)
print(f"[rt] 转向: yaw={yaw:.3f} rad (应≠0)")
r2.set_dofs_velocity([0.0, 0.0], wl + wr)
save("2_turn")

# ===== 3. 头部 (effort 已修) =====
r2.set_dofs_kp(200.0); r2.set_dofs_kv(20.0)
r2.set_dofs_position([0.4], head)
for _ in range(60):
    scene.step()
print(f"[rt] 头部: pos={flat(r2.get_dofs_position(head)).round(3).tolist()} (应≈0.4)")

# ===== 4. 升降/腰部 =====
r2.set_dofs_position([0.3], lift)
for _ in range(40):
    scene.step()
print(f"[rt] 升降: pos={flat(r2.get_dofs_position(lift)).round(3).tolist()} (应≈0.3)")
r2.set_dofs_position([0.5], waist)
for _ in range(40):
    scene.step()
print(f"[rt] 腰部: pos={flat(r2.get_dofs_position(waist)).round(3).tolist()} (应≈0.5)")
save("3_upper")

# ===== 5. 左臂 =====
arm_l = [flat(r2.get_joint(f'J{i}_left_joint').dofs_idx_local).astype(int).tolist() for i in range(1,8)]
for i, d in enumerate(arm_l):
    r2.set_dofs_position([0.5], d)
    for _ in range(30):
        scene.step()
print(f"[rt] 左臂: {[round(float(flat(r2.get_dofs_position(d))[0]),2) for d in arm_l]}")
save("4_arms")
print("[rt] 完成")
