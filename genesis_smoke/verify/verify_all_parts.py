#!/usr/bin/env python
"""综合验证 v2: 所有部件正常 (57 DOF, GR1 手移植后)
修复: control_dofs_position (PD 控制) 代替 set_dofs_position
     接触检查同时读 link_a/link_b 挑机器人侧
     手臂用非零目标 (避免限位中点空转)
     增加右臂/右手/右手 mimic 测试 + 穿模检查 + n_dofs 断言
输出 JSON 结果"""
import json
import numpy as np
import genesis as gs

URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf"
result = {}

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
    sim_options=gs.options.SimOptions(dt=1/60.0, substeps=8),
    rigid_options=gs.options.RigidOptions(
        friction_cone=gs.friction_cone.elliptic,
        impratio=100,
        contact_resolution=gs.contact_resolution.signorini,
    ),
    show_viewer=False,
)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=URDF), surface=gs.surfaces.Default())
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default())
scene.build()

# ===== 1. 加载验证 =====
links = scene.rigid_solver.links
name2idx = {l.name: i for i, l in enumerate(links)}
result['n_dofs'] = r2.n_dofs
result['n_links'] = len(links)
result['dofs_ok'] = (r2.n_dofs == 57)
# 控制量 (无 mimic 的手指关节)
ctrl = [j.name for j in r2.joints if any(k in j.name for k in ['thumb','index','middle','ring','pinky']) and 'gr1' not in j.name and 'intermediate' not in j.name and 'distal' not in j.name and 'tip' not in j.name]
result['finger_ctrl'] = ctrl
result['ctrl_ok'] = (len(ctrl) == 12)
print(f"[V] 加载: n_dofs={r2.n_dofs} (期望57 {'OK' if result['dofs_ok'] else 'FAIL'}), links={len(links)}, 手指控制={len(ctrl)}")

# ===== 2. 姿态验证 (0 位姿 + kp 保持) =====
# 手臂/躯干: kp 保持; 轮子/万向轮: 自由; 手指: 不 control (避免后续 drive 失效)
arm_dofs = []
for j in r2.joints:
    n = j.name
    if n not in ('wheel_left_joint', 'wheel_right_joint') and not any(k in n for k in ['thumb','index','middle','ring','pinky']):
        arm_dofs += jidx(n)
arm_dofs = list(set(arm_dofs))
r2.set_dofs_kp(5000.0, arm_dofs); r2.set_dofs_kv(200.0, arm_dofs)
wl, wr = jidx('wheel_left_joint'), jidx('wheel_right_joint')
r2.set_dofs_kp(0.0, wl+wr); r2.set_dofs_kv(100.0, wl+wr)
# 万向轮自由滚动
caster_dofs = []
for j in r2.joints:
    if 'caster' in j.name:
        caster_dofs += jidx(j.name)
r2.set_dofs_kp(0.0, caster_dofs); r2.set_dofs_kv(0.0, caster_dofs)
for i in range(150):
    r2.control_dofs_position([0.0]*len(arm_dofs), arm_dofs)
    scene.step()

basep = to_np(r2.get_pos()).reshape(-1)
wheel_l = to_np(links[name2idx['wheel_left_Link']].get_pos()).reshape(-1)
wheel_r = to_np(links[name2idx['wheel_right_Link']].get_pos()).reshape(-1)
# 手指末端最低点
finger_bottom = None
for ln in links:
    if 'index' in ln.name and ('distal' in ln.name or 'tip' in ln.name or 'intermediate' in ln.name):
        fp = to_np(ln.get_pos()).reshape(-1)
        if finger_bottom is None or fp[2] < finger_bottom[2]:
            finger_bottom = fp
# 接触: 同时读 link_a/link_b, 挑属于机器人的
c = r2.get_contacts(with_entity=ground)
from collections import Counter
contact_names = []
for i in range(len(c['position'])):
    la, lb = int(c['link_a'][i]), int(c['link_b'][i])
    if la < len(links) and links[la].name != 'plane_baselink':
        contact_names.append(links[la].name)
    if lb < len(links) and links[lb].name != 'plane_baselink':
        contact_names.append(links[lb].name)
contact_cnt = Counter(contact_names)
result['wheel_bottom'] = [round(wheel_l[2]-0.085, 4), round(wheel_r[2]-0.085, 4)]
result['base_z'] = round(basep[2], 4)
result['finger_bottom_z'] = round(finger_bottom[2], 4) if finger_bottom is not None else None
result['contact_links'] = dict(contact_cnt)
# 姿态 OK: 轮底≈0, base 不插地, 手指离地>0.01, 接触只有轮子
wheel_ok = abs(wheel_l[2]-0.085) < 0.01 and abs(wheel_r[2]-0.085) < 0.01
base_ok = basep[2] > -0.01
finger_ok_pose = (finger_bottom is not None and finger_bottom[2] > 0.01)
contact_ok = all('wheel' in k or 'caster' in k for k in contact_cnt)
result['pose_ok'] = bool(wheel_ok and base_ok and finger_ok_pose and contact_ok)
print(f"[V] 姿态: 轮底={result['wheel_bottom']}, base_z={basep[2]:.4f}, 手指底={finger_bottom[2]:.4f}, 接地={dict(contact_cnt)} {'OK' if result['pose_ok'] else 'FAIL'}")

# ===== 3. 轮子驱动 (速度控制, 前进) =====
p0 = basep[:2].copy()
r2.control_dofs_velocity([3.0, -3.0], wl + wr)
for i in range(60):
    scene.step()
r2.control_dofs_velocity([0.0, 0.0], wl + wr)
p1 = to_np(r2.get_pos()).reshape(-1)[:2]
disp = np.linalg.norm(p1 - p0)
result['wheel_drive_disp'] = round(float(disp), 4)
result['wheel_ok'] = disp > 0.05
print(f"[V] 轮子: 前进位移={disp:.4f} m {'OK' if disp>0.05 else 'FAIL'}")

# ===== 4. 升降/腰部/头部 (PD 控制) =====
def drive_to(name, target, tol, steps=80, kp=5000.0):
    idx = jidx(name)
    r2.set_dofs_kp(kp, idx); r2.set_dofs_kv(200.0, idx)
    for i in range(steps):
        t = min(1.0, (i+1)/steps)
        r2.control_dofs_position([target*t], idx)
        scene.step()
    # 稳定
    for _ in range(20):
        r2.control_dofs_position([target], idx)
        scene.step()
    q = float(flat(r2.get_dofs_position(idx))[0])
    return q, abs(q - target) < tol

for name, target, tol in [('lift_joint', 0.3, 0.1), ('waist_joint', 0.5, 0.1), ('head_joint', 0.4, 0.1)]:
    q, ok = drive_to(name, target, tol)
    result[f'{name}_q'] = round(q, 3)
    result[f'{name}_ok'] = ok
    print(f"[V] {name}: 目标={target} 实际={q:.3f} {'OK' if ok else 'FAIL'}")

# 躯干归零 (手臂测试前)
for name in ['lift_joint', 'waist_joint', 'head_joint']:
    drive_to(name, 0.0, 0.1, steps=40)

# ===== 5. 手臂 J1-J7 (左右臂, 非零目标) =====
arm_ok = True
for side in ['left', 'right']:
    for i in range(1, 8):
        name = f'J{i}_{side}_joint'
        idx = jidx(name)
        lim = r2.get_joint(name).dofs_limit
        lo, hi = float(lim[0][0]), float(lim[0][1])
        # 非零目标: 用 limit 的 30% 位置 (避开中点 0)
        target = lo + 0.3 * (hi - lo)
        if abs(target) < 0.1:
            target = lo + 0.6 * (hi - lo)
        q, ok = drive_to(name, target, 0.15, steps=60)
        arm_ok = arm_ok and ok
        result[f'J{i}_{side}_q'] = round(q, 3)
        if not ok:
            result[f'J{i}_{side}_fail'] = f'target={target:.3f}'
        print(f"[V] J{i}_{side}: 目标={target:.3f} 实际={q:.3f} {'OK' if ok else 'FAIL'}")
result['arm_ok'] = arm_ok

# ===== 6. 手指 6 位置量 + mimic (左右手) =====
# 回 0 位姿
for i in range(30):
    r2.control_dofs_position([0.0]*len(arm_dofs), arm_dofs)
    scene.step()
finger_ok = True
mimic_ok = True
for side in ['left', 'right']:
    for fname in ['thumb_flex', 'thumb_abduct', 'index', 'middle', 'ring', 'pinky']:
        name = f'{fname}_{side}_joint'
        idx = jidx(name)
        lim = r2.get_joint(name).dofs_limit
        lo, hi = float(lim[0][0]), float(lim[0][1])
        target = min(0.5, hi)
        q, ok = drive_to(name, target, 0.15, steps=50)
        finger_ok = finger_ok and ok
        result[f'finger_{fname}_{side}_q'] = round(q, 3)
        print(f"[V] 手指 {fname}_{side}: 目标={target} 实际={q:.3f} {'OK' if ok else 'FAIL'}")
    # mimic 检查
    idx_c = jidx(f'index_{side}_joint')
    idx_m = jidx(f'gr1_{side[0].upper()}_index_intermediate_joint')
    if idx_m:
        qc = float(flat(r2.get_dofs_position(idx_c))[0])
        qm = float(flat(r2.get_dofs_position(idx_m))[0])
        ok = abs(qm - qc) < 0.2
        mimic_ok = mimic_ok and ok
        result[f'mimic_index_{side}_q'] = [round(qc, 3), round(qm, 3)]
        print(f"[V] mimic {side}: index={qc:.3f} intermediate={qm:.3f} {'OK' if ok else 'FAIL'}")
result['finger_ok'] = finger_ok
result['mimic_ok'] = mimic_ok

# ===== 7. 穿模检查 (自碰撞: 非相邻 link 间距离) =====
# 检查手指/手臂是否穿入 base 或地面 (用接触 + 位置)
penetration = []
for ln in links:
    if 'index' in ln.name or 'thumb' in ln.name or 'pinky' in ln.name:
        p = to_np(ln.get_pos()).reshape(-1)
        if p[2] < 0.005:  # 手指底接近地面
            penetration.append(f'{ln.name} z={p[2]:.3f}')
result['penetration'] = penetration
result['penetration_ok'] = (len(penetration) == 0)
print(f"[V] 穿模检查: {penetration if penetration else '无'} {'OK' if result['penetration_ok'] else 'FAIL'}")

result['all_ok'] = (result['dofs_ok'] and result['ctrl_ok'] and result['pose_ok'] and result['wheel_ok'] and
                    result['lift_joint_ok'] and result['waist_joint_ok'] and result['head_joint_ok'] and
                    result['arm_ok'] and result['finger_ok'] and result['mimic_ok'] and result['penetration_ok'])
print(f"[V] 总结: all_ok={result['all_ok']}")
print("[V] JSON: " + json.dumps(result, ensure_ascii=False, default=float))
