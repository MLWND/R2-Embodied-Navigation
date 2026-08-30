#!/usr/bin/env python
"""诊断: 关节限位 + 驱动时轮子实际速度 + 底座接触"""
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
    """任意嵌套 tensor/tuple/list -> 1D numpy"""
    parts = []
    for i in to_np(x):
        parts.append(np.asarray(i).reshape(-1))
    return np.concatenate(parts)

gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/60.0, substeps=4), show_viewer=False)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=R2_URDF), surface=gs.surfaces.Default())
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0.0, 0.0, 0.0)), surface=gs.surfaces.Default())
scene.build()
print("[diag] build 完成")

# 1. 关节限位 (直接从 URDF 读, 不查 Genesis API)
import xml.etree.ElementTree as ET
tree = ET.parse(R2_URDF)
for j in tree.getroot().findall('joint'):
    name = j.get('name')
    lim = j.find('limit')
    if lim is not None:
        print(f"[diag] {name:24s} limit lower={lim.get('lower')} upper={lim.get('upper')}")

# 2. settle
for _ in range(30):
    scene.step()
print(f"[diag] settle 后 base z={to_np(r2.get_pos()).reshape(-1)[2]:.4f}")

# 3. 驱动: 设轮速, 查实际轮速 + 基座位移
wl = flat(r2.get_joint('wheel_left_joint').dofs_idx_local).astype(int).tolist()
wr = flat(r2.get_joint('wheel_right_joint').dofs_idx_local).astype(int).tolist()
r2.set_dofs_velocity([3.0, 3.0], wl + wr)
for i in range(30):
    scene.step()
    if i % 10 == 0:
        v = flat(r2.get_dofs_velocity(wl + wr)).round(3).tolist()
        p = to_np(r2.get_pos()).round(4).tolist()
        print(f"[diag] step {i}: 轮速={v} base={p}")
r2.set_dofs_velocity([0.0, 0.0], wl + wr)
print("[diag] 完成")
