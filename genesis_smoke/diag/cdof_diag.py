#!/usr/bin/env python
"""验证摩擦约束的接触点速度计算: cdof_ang/cdof_vel/root_COM"""
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

gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/60.0, substeps=4), show_viewer=False)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=URDF), surface=gs.surfaces.Default())
g = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default())
scene.build()
for _ in range(100): scene.step()

links = scene.rigid_solver.links
name2idx = {l.name: i for i, l in enumerate(links)}
wli = name2idx['wheel_left_Link']

# 轮子 dof 索引
wj = flat(r2.get_joint('wheel_left_joint').dofs_idx_local).astype(int).tolist()
print(f"wheel_left_joint dof: {wj}")

ds = scene.rigid_solver.dyn_state
# cdof_ang / cdof_vel 是 (n_dofs, 3)
import genesis.utils.misc as gum
cdof_ang = np.asarray(gum.qd_to_torch(ds.dofs.cdof_ang, transpose=True, copy=False).cpu())
cdof_vel = np.asarray(gum.qd_to_torch(ds.dofs.cdof_vel, transpose=True, copy=False).cpu())
print(f"cdof_ang shape: {cdof_ang.shape}")

for d in wj:
    print(f"dof {d}: cdof_ang={cdof_ang[0, d].round(4).tolist()} cdof_vel={cdof_vel[0, d].round(4).tolist()}")

# root_COM
root_com = np.asarray(gum.qd_to_torch(ds.links.root_COM, transpose=True, copy=False).cpu())
print(f"links.root_COM shape: {root_com.shape}")
print(f"wheel_left root_COM={root_com[0, wli].round(4).tolist()}")

# 轮子 link 位置
wpos = flat(links[wli].get_pos())
print(f"wheel_left pos={wpos.round(4).tolist()}")

# 接触点 (轮子)
cs = scene.rigid_solver.collider._collider_state
n = int(cs.n_contacts[0])
for i in range(n):
    pos = np.asarray(to_np(cs.contact_data.pos[i, 0])).reshape(-1)
    if abs(pos[2]) < 0.01 and abs(pos[1]) > 0.15:  # 轮子接触点
        print(f"接触点 pos={pos.round(4).tolist()}")
        # 手算: 假设轮子角速度 1 rad/s
        t_pos = pos - root_com[0, wli]
        v = cdof_vel[0, wj[0]] + np.cross(cdof_ang[0, wj[0]], t_pos)
        print(f"  t_pos(相对root_COM)={t_pos.round(4).tolist()}")
        print(f"  接触点速度(ω=1)={v.round(4).tolist()}  (理论滚动方向应≈[-0.085,0,0])")
print("完成")
