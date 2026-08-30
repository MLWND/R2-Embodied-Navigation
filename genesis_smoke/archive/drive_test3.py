#!/usr/bin/env python
"""驱动测试 v3: 反转轮向(前进) + 高摩擦 + 验证轮子转角"""
import numpy as np
import genesis as gs

R2_URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf"

def to_np(x):
    if hasattr(x, "cpu"):
        x = x.cpu()
    if isinstance(x, (tuple, list)):
        return [to_np(i) for i in x]
    return np.asarray(x)

def flat(x):
    parts = []
    for i in to_np(x):
        parts.append(np.asarray(i).reshape(-1))
    return np.concatenate(parts)

gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/60.0, substeps=4), show_viewer=False)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=R2_URDF), surface=gs.surfaces.Default())
# 高摩擦地面
ground = scene.add_entity(
    morph=gs.morphs.Plane(pos=(0.0, 0.0, 0.0)),
    surface=gs.surfaces.Default(),
    material=gs.materials.Rigid(friction=2.0),
)
scene.build()
print("[drive3] build 完成")

wl = flat(r2.get_joint('wheel_left_joint').dofs_idx_local).astype(int).tolist()
wr = flat(r2.get_joint('wheel_right_joint').dofs_idx_local).astype(int).tolist()

for _ in range(100):
    scene.step()
print(f"[drive3] settle 后 base={to_np(r2.get_pos()).round(4).tolist()}")

# 反转轮向: 负角度 = 前进 (+x)
r2.set_dofs_kp(50.0); r2.set_dofs_kv(5.0)
target = -18.85  # 反转
for i in range(200):
    r2.set_dofs_position([target * (i+1)/200, target * (i+1)/200], wl + wr)
    scene.step()
    if i % 50 == 0:
        p = to_np(r2.get_pos()).round(3).tolist()
        wa = flat(r2.get_dofs_position(wl + wr)).round(2).tolist()
        print(f"[drive3] step {i}: base={p} 轮角={wa}")

p_end = to_np(r2.get_pos()).round(3).tolist()
print(f"[drive3] 前进测试结束: base={p_end} (x 应显著增大)")
print("[drive3] 完成")
