#!/usr/bin/env python
"""
R2 关节测试: 驱动/转向/升降/腰部/头部/双臂/手爪
用着色版 URDF (r2_urdf_v01_colored.urdf)。
"""
import os
import numpy as np
from PIL import Image
import genesis as gs

R2_URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf"
OUT = "/home/xujinlong/test/genesis_smoke/output"
os.makedirs(OUT, exist_ok=True)


def to_np(x):
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def main():
    gs.init(backend=gs.gpu)
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1 / 60.0, substeps=4), show_viewer=False)

    r2 = scene.add_entity(morph=gs.morphs.URDF(file=R2_URDF), surface=gs.surfaces.Default())
    ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0.0, 0.0, 0.0)), surface=gs.surfaces.Default())

    cam = scene.add_camera(model="pinhole", res=(1280, 720), pos=(2.5, 1.5, 1.5), lookat=(0.0, 0.0, 0.5), fov=60)
    scene.build()
    print(f"[joint] build 完成, n_dofs={r2.n_dofs}")

    def jidx(name):
        return to_np(r2.get_joint(name).dofs_idx_local).reshape(-1).tolist()

    # 关节 DOF 索引
    wheel_l, wheel_r = jidx("wheel_left_joint"), jidx("wheel_right_joint")
    lift, waist, head = jidx("lift_joint"), jidx("waist_joint"), jidx("head_joint")
    arm_l = [jidx(f"J{i}_left_joint") for i in range(1, 8)]
    arm_r = [jidx(f"J{i}_right_joint") for i in range(1, 8)]
    print(f"[joint] wheel_l={wheel_l} wheel_r={wheel_r} lift={lift} waist={waist} head={head}")
    print(f"[joint] 左臂 DOF: {[d[0] for d in arm_l]}  右臂 DOF: {[d[0] for d in arm_r]}")

    def save(name):
        rgb, depth, seg, normal = cam.render(rgb=True, depth=True)
        Image.fromarray(np.asarray(rgb)).save(os.path.join(OUT, f"joint_{name}.png"))
        print(f"[joint] 已保存 joint_{name}.png")

    def base_state():
        pos = to_np(r2.get_pos()).reshape(-1)
        quat = to_np(r2.get_quat()).reshape(-1)
        return pos, quat

    # ============ 0. 先 settle 让轮子接触地面 ============
    for _ in range(30):
        scene.step()

    # ============ 1. 驱动测试: 双轮同速前进 ============
    save("0_initial")
    r2.set_dofs_velocity([3.0, 3.0], wheel_l + wheel_r)
    for _ in range(120):
        scene.step()
    p0, _ = base_state()
    print(f"[joint] 驱动测试: 前进后 base pos={p0.round(3).tolist()} (x/y 应增大)")
    r2.set_dofs_velocity([0.0, 0.0], wheel_l + wheel_r)

    # ============ 2. 转向测试: 差速 ============
    r2.set_dofs_velocity([3.0, -3.0], wheel_l + wheel_r)
    for _ in range(120):
        scene.step()
    p1, q1 = base_state()
    yaw = 2 * np.arctan2(q1[3], q1[0])  # 简化 yaw
    print(f"[joint] 转向测试: base pos={p1.round(3).tolist()}, yaw≈{yaw:.2f} rad (应≠0)")
    r2.set_dofs_velocity([0.0, 0.0], wheel_l + wheel_r)
    save("1_after_drive_turn")

    # 位置控制统一调高 PD 增益
    r2.set_dofs_kp(200.0)
    r2.set_dofs_kv(20.0)

    # ============ 3. 升降测试 ============
    r2.set_dofs_position([0.3], lift)
    for _ in range(60):
        scene.step()
    print(f"[joint] 升降测试: lift pos={to_np(r2.get_dofs_position(lift)).round(3).tolist()} (应≈0.3)")

    # ============ 4. 腰部测试 ============
    r2.set_dofs_position([0.5], waist)
    for _ in range(60):
        scene.step()
    print(f"[joint] 腰部测试: waist pos={to_np(r2.get_dofs_position(waist)).round(3).tolist()} (应≈0.5)")

    # ============ 5. 头部测试 ============
    r2.set_dofs_position([0.4], head)
    for _ in range(60):
        scene.step()
    print(f"[joint] 头部测试: head pos={to_np(r2.get_dofs_position(head)).round(3).tolist()} (应≈0.4)")

    # ============ 6. 左臂测试: 逐关节 ============
    for i, d in enumerate(arm_l):
        r2.set_dofs_position([0.5], d)
        for _ in range(40):
            scene.step()
        cur = to_np(r2.get_dofs_position(d)).round(3).tolist()
        print(f"[joint] 左臂 J{i+1}: pos={cur} (应≈0.5)")
    save("2_arms_up")

    # ============ 7. 右臂测试 ============
    for i, d in enumerate(arm_r):
        r2.set_dofs_position([-0.5], d)
        for _ in range(40):
            scene.step()
        cur = to_np(r2.get_dofs_position(d)).round(3).tolist()
        print(f"[joint] 右臂 J{i+1}: pos={cur} (应≈-0.5)")

    # ============ 8. 手爪: 固定关节被 merge, 验证 hand link 存在 ============
    for name in ["hand_left_Link", "hand_right_Link"]:
        try:
            link = r2.get_link(name)
            print(f"[joint] {name}: 存在 (固定关节已合并进 J7, mesh 保留)")
        except Exception:
            print(f"[joint] {name}: 未找到 (被合并)")

    save("3_final")
    print("[joint] 全部测试完成")


if __name__ == "__main__":
    main()
