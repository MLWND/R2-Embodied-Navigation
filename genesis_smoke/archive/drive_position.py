#!/usr/bin/env python
"""位置控制驱动测试 (轮高已修 0.085 + J2 抬臂): 对比 drive_test3 的效果"""
import numpy as np
import genesis as gs

R2_URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf"

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
    sim_options=gs.options.SimOptions(dt=1/60.0, substeps=10),
    # 无 noslip: noslip 会把 6 点着地的轮子锁死 (抓取任务才用)
    rigid_options=gs.options.RigidOptions(enable_self_collision=False),
    show_viewer=False,
)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=R2_URDF), surface=gs.surfaces.Default(),
                      material=gs.materials.Rigid(friction=5.0))
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default(),
                          material=gs.materials.Rigid(friction=5.0))
scene.build()
print("[pc] build 完成")

wl, wr = jidx('wheel_left_joint'), jidx('wheel_right_joint')
j2l, j2r = jidx('J2_left_joint'), jidx('J2_right_joint')

# 抬臂
r2.set_dofs_kp(3000.0, j2l + j2r); r2.set_dofs_kv(200.0, j2l + j2r)
# 轮子位置控制: 中刚度
r2.set_dofs_kp(50.0, wl+wr); r2.set_dofs_kv(5.0, wl+wr)

for i in range(150):
    t = (i+1)/150
    r2.set_dofs_position([1.57*t, -1.57*t], j2l + j2r)
    scene.step()
print(f"[pc] settle 后 base={to_np(r2.get_pos()).round(3).tolist()}")

# 前进: 轮子正转 18.85 rad (3圈), 分 300 步
target = 18.85
for i in range(300):
    r2.set_dofs_position([target*(i+1)/300, target*(i+1)/300], wl + wr)
    scene.step()
    if i % 60 == 0:
        print(f"[pc] 前进 step {i}: base={to_np(r2.get_pos()).round(3).tolist()}")
p_end = to_np(r2.get_pos()).reshape(-1)
print(f"[pc] 前进结束: x={p_end[0]:.3f} (Δx={p_end[0]:.3f})")

# 转向: 左正右反
for _ in range(50): scene.step()
r2.set_dofs_position([0.0, 0.0], wl+wr)
for i in range(300):
    r2.set_dofs_position([target*(i+1)/300, -target*(i+1)/300], wl + wr)
    scene.step()
    if i % 60 == 0:
        q = flat(r2.get_quat()); yaw = 2*np.arctan2(q[3], q[0])
        print(f"[pc] 转向 step {i}: yaw={yaw:.3f}")
q = flat(r2.get_quat()); yaw = 2*np.arctan2(q[3], q[0])
print(f"[pc] 转向结束 yaw={yaw:.3f} rad")
print("[pc] 完成")
