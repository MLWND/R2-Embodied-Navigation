#!/usr/bin/env python
"""测量手臂长度: URDF 理论值 vs 仿真实际值"""
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

# ============ URDF 理论长度 (0 位姿, 自然下垂) ============
import xml.etree.ElementTree as ET
tree = ET.parse(URDF); root = tree.getroot()
joints = {}
for j in root.findall('joint'):
    joints[j.get('name')] = j

def origin_xyz(name):
    o = joints[name].find('origin')
    return [float(x) for x in o.get('xyz').split()]

# 左臂链 (0 位姿, 各关节 origin 的 z 分量累加 = 下垂长度)
chain = ['J1_left_joint', 'J2_left_joint', 'J3_left_joint', 'J4_left_joint',
         'J5_left_joint', 'J6_left_joint', 'J7_left_joint']
# 0 位姿时, 各段沿 z 的位移 (J2 绕 x, J3 绕 z, J4 绕 y, J5 绕 z, J6 绕 x, J7 绕 z)
# 0 位姿: 所有关节角 0, 各段 origin 的 z 直接累加
z_drop = 0.0
for name in chain:
    xyz = origin_xyz(name)
    z_drop += xyz[2]  # 0 位姿时 z 分量直接累加 (下垂)
hand_xyz = origin_xyz('hand_left_joint')
z_drop += hand_xyz[2]
print(f"[len] URDF 0位姿: 从 J1 到 hand 下垂长度 = {abs(z_drop):.4f} m")

# J1 相对 base 高度
j1 = origin_xyz('J1_left_joint')
lift = origin_xyz('lift_joint')
waist = origin_xyz('waist_joint')
j1_h = lift[2] + waist[2] + j1[2]
print(f"[len] URDF: J1(肩) 相对 base 高度 = {j1_h:.4f} m")
print(f"[len] URDF: 手底(0位姿) 相对 base = {j1_h + z_drop:.4f} m")

# ============ 仿真实际长度 ============
gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/60.0, substeps=4), show_viewer=False)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=URDF), surface=gs.surfaces.Default())
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default())
scene.build()

# 保持 0 位姿
arm_dofs = []
for j in r2.joints:
    if j.name not in ('wheel_left_joint', 'wheel_right_joint'):
        arm_dofs += jidx(j.name)
arm_dofs = list(set(arm_dofs))
r2.set_dofs_kp(5000.0, arm_dofs); r2.set_dofs_kv(200.0, arm_dofs)
wl, wr = jidx('wheel_left_joint'), jidx('wheel_right_joint')
r2.set_dofs_kp(0.0, wl+wr); r2.set_dofs_kv(100.0, wl+wr)
for i in range(100):
    r2.set_dofs_position([0.0]*len(arm_dofs), arm_dofs)
    scene.step()

links = scene.rigid_solver.links
name2idx = {l.name: i for i, l in enumerate(links)}
j1p = to_np(links[name2idx['J1_left_Link']].get_pos()).reshape(-1)
handp = to_np(links[name2idx['hand_left_Link']].get_pos()).reshape(-1)
basep = to_np(r2.get_pos()).reshape(-1)
print(f"[len] 仿真: J1_left pos = {j1p.round(4).tolist()}")
print(f"[len] 仿真: hand_left pos = {handp.round(4).tolist()}")
print(f"[len] 仿真: J1->hand 距离 = {np.linalg.norm(handp - j1p):.4f} m")
print(f"[len] 仿真: J1 相对 base 高度 = {j1p[2] - basep[2]:.4f} m")
print(f"[len] 仿真: hand 相对 base 高度 = {handp[2] - basep[2]:.4f} m")
print("[len] 完成")
