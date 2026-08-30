#!/usr/bin/env python
"""轮高扫描: 对指定轮高测 settle + 位置控制前进距离 + 打滑率"""
import sys, xml.etree.ElementTree as ET, numpy as np
import genesis as gs

WH = float(sys.argv[1])
R2_URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf"

# 临时修改轮高
tree = ET.parse(R2_URDF)
root = tree.getroot()
for j in root.findall('joint'):
    if j.get('name') in ('wheel_left_joint', 'wheel_right_joint'):
        o = j.find('origin')
        xyz = o.get('xyz').split(); xyz[2] = str(WH)
        o.set('xyz', ' '.join(xyz))
tree.write(R2_URDF, encoding='utf-8', xml_declaration=True)

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
    sim_options=gs.options.SimOptions(dt=1/60.0, substeps=4),
    show_viewer=False,
)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=R2_URDF), surface=gs.surfaces.Default())
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default())
scene.build()

wl, wr = jidx('wheel_left_joint'), jidx('wheel_right_joint')
r2.set_dofs_kp(50.0, wl+wr); r2.set_dofs_kv(5.0, wl+wr)

for _ in range(100):
    scene.step()

# 轮底高度检查
links = scene.rigid_solver.links
name2idx = {l.name: i for i, l in enumerate(links)}
z_wl = float(flat(links[name2idx['wheel_left_Link']].get_pos())[2])
z_cast = float(flat(links[name2idx['caster_front_left_wheel_Link']].get_pos())[2])

# 前进: 位置控制 2 圈 (12.57 rad), 200 步
target = 12.57
p0 = to_np(r2.get_pos()).reshape(-1)[:2].copy()
w0 = flat(r2.get_dofs_position(wl+wr)).copy()
for i in range(200):
    r2.set_dofs_position([target*(i+1)/200, target*(i+1)/200], wl + wr)
    scene.step()
p1 = to_np(r2.get_pos()).reshape(-1)[:2]
w1 = flat(r2.get_dofs_position(wl+wr))
dx = p1[0]-p0[0]
dtheta = np.abs(w1[0]-w0[0])
roll = dtheta * 0.085  # 理论滚动距离
slip = 1.0 - (dx / roll) if roll > 0.001 else 1.0
print(f"[scan] WH={WH}: 轮底(左)={z_wl-0.085:.4f} caster底={z_cast-0.05:.4f} 前进 dx={dx:.3f} 理论roll={roll:.3f} 打滑率={slip*100:.1f}%")
