#!/usr/bin/env python
"""关节类型/dofs索引检查: 打印 URDF 全部关节的 name/type/dofs.
用途: 排查锁姿态时 FREE 根关节误入控制集(基座瞬移bug)等问题.
注意 j.type 是整数枚举(4=FREE), 判断必须用 j.type == gs.JOINT_TYPE.FREE."""
import os
os.environ.setdefault('PXR_WORK_THREAD_LIMIT', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import genesis as gs

def host(x):
    if hasattr(x, 'cpu'): x = x.cpu()
    return np.asarray(x).reshape(-1)

gs.init(backend=gs.gpu)
scene = gs.Scene(show_viewer=False)
r2 = scene.add_entity(morph=gs.morphs.URDF(
    file="/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf",
    pos=(1.5, 1.5, 0.005), euler=(0, 0, -135)))
scene.build()
print(f"[jc] n_dofs={r2.n_dofs} n_joints={len(r2.joints)}", flush=True)
for j in r2.joints:
    idx = host(j.dofs_idx_local).astype(int).tolist()
    print(f"[jc] {j.name:28s} type={str(j.type):22s} dofs={idx}", flush=True)
print(f"[jc] 出生位姿 base={host(r2.get_pos()).round(3).tolist()}", flush=True)
# 模拟 arm_dofs 计算看会不会包含基座
arm = []
for j in r2.joints:
    if j.name in ('wheel_left_joint', 'wheel_right_joint'): continue
    if 'FREE' in str(getattr(j, 'type', '')): continue
    arm += host(j.dofs_idx_local).astype(int).tolist()
arm = sorted(set(arm))
print(f"[jc] arm_dofs 数量={len(arm)} 范围=[{min(arm)},{max(arm)}] (n_dofs={r2.n_dofs})", flush=True)
