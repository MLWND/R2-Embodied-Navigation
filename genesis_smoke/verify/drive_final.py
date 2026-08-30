#!/usr/bin/env python
"""最终综合验证: 前进 → 转向 → 前进, 渲染确认"""
import numpy as np
import genesis as gs
from PIL import Image

R2_URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf"
OUT = "/home/xujinlong/test/genesis_smoke/output"

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

def save_frame(name):
    rgb, _, _, _ = cam.render(rgb=True, depth=True)
    Image.fromarray(np.asarray(rgb)).save(f"{OUT}/{name}.png")
    print(f"[fin] 已保存 {name}.png")

gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/60.0, substeps=4), show_viewer=False)
cam = scene.add_camera(res=(640, 480), pos=(2.5, 0, 1.2), lookat=(0, 0, 0.5), fov=60)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=R2_URDF), surface=gs.surfaces.Default())
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default())
scene.build()
print("[fin] build 完成")

wl, wr = jidx('wheel_left_joint'), jidx('wheel_right_joint')
j2l, j2r = jidx('J2_left_joint'), jidx('J2_right_joint')
r2.set_dofs_kp(3000.0, j2l + j2r); r2.set_dofs_kv(200.0, j2l + j2r)
r2.set_dofs_kp(1000.0, wl+wr); r2.set_dofs_kv(100.0, wl+wr)

# settle + 抬臂
for i in range(150):
    t = (i+1)/150
    r2.set_dofs_position([1.57*t, -1.57*t], j2l + j2r)
    scene.step()
print(f"[fin] settle: base={to_np(r2.get_pos()).round(3).tolist()} yaw={yaw_of():.3f}")

# 1. 前进 2s
for i in range(120):
    r2.set_dofs_velocity([5.0, -5.0], wl + wr)
    scene.step()
p = to_np(r2.get_pos()).reshape(-1)
print(f"[fin] 前进后: base={p.round(3).tolist()} yaw={yaw_of():.3f}")
save_frame('fin_1_forward')

# 2. 转向 2s
for i in range(120):
    r2.set_dofs_velocity([5.0, 5.0], wl + wr)
    scene.step()
p = to_np(r2.get_pos()).reshape(-1)
print(f"[fin] 转向后: base={p.round(3).tolist()} yaw={yaw_of():.3f}")
save_frame('fin_2_turn')

# 3. 再前进 2s
for i in range(120):
    r2.set_dofs_velocity([5.0, -5.0], wl + wr)
    scene.step()
p = to_np(r2.get_pos()).reshape(-1)
print(f"[fin] 再前进后: base={p.round(3).tolist()} yaw={yaw_of():.3f}")
save_frame('fin_3_forward2')

print("[fin] 完成")
