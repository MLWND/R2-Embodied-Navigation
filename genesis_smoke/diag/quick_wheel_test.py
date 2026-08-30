#!/usr/bin/env python
"""官方 wheel.urdf 驱动测试: 验证 Genesis 对 MESH 碰撞轮子的摩擦是否生效"""
import numpy as np
import genesis as gs

WHEEL_URDF = "/home/xujinlong/test/genesis-doc/genesis-world-main/genesis/assets/urdf/wheel/wheel.urdf"

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
robot = scene.add_entity(morph=gs.morphs.URDF(file=WHEEL_URDF), surface=gs.surfaces.Default())
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default())
scene.build()
print("[qw] build 完成")

# 轮子关节
wj = flat(robot.get_joint('baseHinge_1').dofs_idx_local).astype(int).tolist()
print(f"[qw] n_dofs={robot.n_dofs} 轮子dof={wj}")

# 位置控制转轮子
robot.set_dofs_kp(50.0, wj); robot.set_dofs_kv(5.0, wj)
for _ in range(100): scene.step()
print(f"[qw] settle 后 base={to_np(robot.get_pos()).round(3).tolist()}")

target = 12.57
for i in range(200):
    robot.set_dofs_position([target*(i+1)/200], wj)
    scene.step()
    if i % 50 == 0:
        p = to_np(robot.get_pos()).reshape(-1)
        wpos = flat(robot.get_dofs_position(wj))
        print(f"[qw] step {i}: base={p.round(3).tolist()} 轮角={wpos.round(2).tolist()}")

p = to_np(robot.get_pos()).reshape(-1)
print(f"[qw] 结束: base_x={p[0]:.3f} (轮子转 2 圈, 理论滚动 {12.57*0.5:.2f}m)")
print("[qw] 完成")
