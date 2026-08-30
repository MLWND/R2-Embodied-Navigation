#!/usr/bin/env python
"""GRScenes navigation 场景: 碰撞物理验证 + R2 机器人站立/驱动

一次加载完成全部验证 (碰撞预处理 ~10min, 避免重复付费):
  1. 掉落探针: 3 个小球, 判定地板碰撞 (z≈0.1) / 墙凸包填屋 (z≈2.4+) / 穿透 (z<-1)
  2. R2 机器人: settle 后 base z≈0.002 (平面基准) -> 场景站立正常
  3. 差速前进 2s -> 位移>0.5m 且 z 稳定

加载方式:
  - 视觉: navigation_colored.usd (collision=False)
  - 碰撞: navigation_collision.usd (collision=True, visualization=False, fixed=True,
    无 CoACD 分解 — 单凸包; 凹形风险由探针#1判定)

注意 (Genesis 1.3.2 API):
  - 实体必须全部在 scene.build() 之前加入
  - 摩擦参数在 gs.options.RigidOptions (不是 SimOptions), 枚举需传 gs.friction_cone.elliptic 对象
  - n_envs=0 时 get_pos() 一维; GPU 张量先 .cpu()
"""
import os
# 线程配额保护: 共享服务器 ulimit -u 1024, 并行库不限制会撞配额,
# 且线程池部分创建失败会导致 build 阶段死锁(全部线程 futex 挂起). 必须在 import numpy 前设置.
os.environ.setdefault('PXR_WORK_THREAD_LIMIT', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '4')

# physics-only 模式: 跳过视觉实体 (GPU 被他人占用 ~76GB 时的省显存路径)
PHYSICS_ONLY = 'physics-only' in os.sys.argv

import numpy as np
import genesis as gs
from PIL import Image

SD = "/home/xujinlong/test/GRScenes/scenes/GRScenes-100/home_scenes/scenes/MWBGLKQKTKJZ2AABAAAAABA8_usd"
VIS = os.path.join(SD, "start_result_navigation_colored.usd")
COL = os.path.join(SD, "start_result_navigation_collision.usd")
R2_URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf"
OUT = "/home/xujinlong/test/genesis_smoke/output"

def host(x):
    if hasattr(x, "cpu"):
        x = x.cpu()
    if isinstance(x, (tuple, list)):
        return np.concatenate([np.asarray(i).reshape(-1) for i in x])
    return np.asarray(x).reshape(-1)

def jidx(e, name):
    return host(e.get_joint(name).dofs_idx_local).astype(int).tolist()

gs.init(backend=gs.gpu)
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=1 / 60.0, substeps=8),
    rigid_options=gs.options.RigidOptions(
        friction_cone=gs.friction_cone.elliptic,
        impratio=100,
        contact_resolution=gs.contact_resolution.signorini,
        use_gjk_collision=True,  # 巨型墙体 mesh 不建 SDF (14.8亿格爆掉), 凸 mesh 走 GJK
    ),
    show_viewer=False,
    vis_options=gs.options.VisOptions(background_color=(0.75, 0.78, 0.82), ambient_light=(0.45, 0.45, 0.45)),
)
cam = scene.add_camera(res=(800, 600), pos=(3.4, 3.4, 1.8), lookat=(0.9, 0.9, 0.4), fov=60)

if not PHYSICS_ONLY:
    print("[gs] 加载视觉场景 ...", flush=True)
    scene.add_entity(morph=gs.morphs.USD(file=VIS, collision=False))
else:
    print("[gs] physics-only: 跳过视觉实体", flush=True)
print("[gs] 加载碰撞场景 (无CoACD, 单凸包) ...", flush=True)
col_ent = scene.add_entity(
    morph=gs.morphs.USD(
        file=COL, collision=True, visualization=False, fixed=True,
        decimate=True, decimate_face_num=300,
        decompose_object_error_threshold=float('inf'),
    ),
    # 898 个家具 mesh 保持非凸会走网格 SDF, 默认 0.1m 格子总量 14.8 亿爆掉;
    # 0.25m 足够(墙/地板已是凸包, 走 GJK), 机器人和球用各自默认材质不受影响
    material=gs.materials.Rigid(sdf_cell_size=0.25),
)
print("[gs] 加载掉落探针 + R2 ...", flush=True)
SPOTS = [(0.0, 0.0), (1.8, 0.6), (3.5, 0.3)]
balls = [scene.add_entity(
    morph=gs.morphs.Sphere(radius=0.1, pos=(x, y, 1.5)),
    material=gs.materials.Rigid(rho=500),
) for x, y in SPOTS]
# 漂移判别探针: 无控制盒子, 与机器人同点位。盒子也滑走=地板碰撞面倾斜/异常
box_probe = scene.add_entity(
    morph=gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(1.5, 1.2, 0.15)),
    material=gs.materials.Rigid(rho=500),
)
r2 = scene.add_entity(
    # 贴地出生(轮底已归零): 高度出生+手臂伺服力会在低摩擦地板上把自由轮"踢"出去
    # (上轮 settle 漂了 2.1m 滑进餐桌区)
    morph=gs.morphs.URDF(file=R2_URDF, pos=(1.5, 1.5, 0.005), euler=(0, 0, -135)),
    surface=gs.surfaces.Default(),
)
print("[gs] build ...", flush=True)
scene.build()
# USD 绑定的 PhysicsMaterial 优先级高于 material 参数, 打滑(2s 0.26m vs 平面 0.72m),
# build 后对场景碰撞 geom 强制覆盖摩擦
_nfr = 0
for _g in col_ent.links[0].geoms:
    _g.set_friction(1.0)
    _nfr += 1
print(f"[gs] build 完成, 已覆盖 {_nfr} 个碰撞 geom 摩擦=1.0", flush=True)

wl, wr = jidx(r2, 'wheel_left_joint'), jidx(r2, 'wheel_right_joint')
# 姿态保持配方 (同 verify/natural_pose_check2.py): 非 wheel 全部关节高 kp 锁 0 位姿,
# 否则臂/腰在重力下塌垂拖地(身体贴地 + 驱动变慢且与摩擦无关)。
# 必须排除 FREE 根关节: 把它设 0 会把基座位姿强设为世界原点(瞬移 bug, 平面测试
# 出生在原点所以从未暴露)
arm_dofs = []
for j in r2.joints:
    if j.name in ('wheel_left_joint', 'wheel_right_joint'):
        continue
    # j.type 是整数枚举 (4=FREE), 必须与 gs.JOINT_TYPE.FREE 比较
    if j.type == gs.JOINT_TYPE.FREE:
        continue
    # 万向轮 (caster steer + wheel) 必须自由: 锁死会变成 8 个焊死圆盘在地上拖,
    # 导致驱动位移只有理论值 ~10% + 横向漂移(轮地牵引问题的根因)
    if 'caster' in j.name.lower():
        continue
    arm_dofs += jidx(r2, j.name)
arm_dofs = sorted(set(arm_dofs))
r2.set_dofs_kp(5000.0, arm_dofs)
r2.set_dofs_kv(200.0, arm_dofs)
r2.set_dofs_kp(0.0, wl + wr)   # 轮子纯速度控制
r2.set_dofs_kv(100.0, wl + wr)

def rpy_of():
    q = host(r2.get_quat())
    w, x, y, z = q
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    return roll, pitch

# ===== 1. 掉落 + 站立 settle =====
for i in range(200):
    r2.set_dofs_position([0.0] * len(arm_dofs), arm_dofs)
    scene.step()
    if (i + 1) % 50 == 0:
        p = host(r2.get_pos())
        bp = host(box_probe.get_pos())
        wq = host(r2.get_dofs_position())[wl + wr]
        print(f"[drift {i+1:3d}] base=({p[0]:+.3f},{p[1]:+.3f},{p[2]:.3f}) 盒子=({bp[0]:+.3f},{bp[1]:+.3f}) 轮转={wq.round(1).tolist()}", flush=True)
    if (i + 1) == 200:
        zs = [round(float(host(b.get_pos())[2]), 3) for b in balls]
        p = host(r2.get_pos())
        roll, pitch = rpy_of()
        sag = np.abs(host(r2.get_dofs_position())[arm_dofs]).max()
        print(f"[settle] 探针球 z={zs}", flush=True)
        print(f"[settle] R2 base={p.round(3).tolist()} (期望 z≈0.002) roll={roll:.1f}° pitch={pitch:.1f}°", flush=True)
        print(f"[settle] 姿态关节最大偏移={sag:.4f} rad (应≈0, 大则塌了)", flush=True)

rgb, _, _, _ = cam.render(rgb=True, depth=True)
Image.fromarray(np.asarray(rgb)).save(f"{OUT}/r2_scene_stand.png")
print(f"[settle] 已保存 r2_scene_stand.png", flush=True)

# ===== 2. 差速前进 2s (朝房间中心) =====
p0 = host(r2.get_pos()).copy()
for i in range(120):
    r2.set_dofs_position([0.0] * len(arm_dofs), arm_dofs)
    r2.set_dofs_velocity([5.0, -5.0], wl + wr)
    scene.step()
    if (i + 1) % 40 == 0:
        p = host(r2.get_pos())
        roll, pitch = rpy_of()
        print(f"[drive {i+1:3d}] base={p.round(3).tolist()} pitch={pitch:.1f}°", flush=True)
p = host(r2.get_pos())
moved = float(np.hypot(p[0] - p0[0], p[1] - p0[1]))
print(f"[drive] 位移={moved:.3f}m 最终 base={p.round(3).tolist()}", flush=True)

rgb, _, _, _ = cam.render(rgb=True, depth=True)
Image.fromarray(np.asarray(rgb)).save(f"{OUT}/r2_scene_drive.png")
print(f"[drive] 已保存 r2_scene_drive.png", flush=True)

zs = [float(host(b.get_pos())[2]) for b in balls]
ball_ok = all(z > -1.0 for z in zs)
stand_ok = abs(float(p[2])) < 0.1
drive_ok = moved > 0.5
print(f"[结论] 掉落{'OK' if ball_ok else '穿透!'} 站立{'OK' if stand_ok else '异常'} 驱动{'OK' if drive_ok else '异常'}")
