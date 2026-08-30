#!/usr/bin/env python
"""
Genesis 冒烟测试 (阶段 0)
目标: 验证 Genesis 能否加载 R2 URDF + table_scene.usd, 并渲染 RGB + 深度落盘。
预期: table_scene 的 MDL 材质回退到 displayColor(无贴图), 属正常现象。
用法: CUDA_VISIBLE_DEVICES=1 conda run -n genesis python smoke_test.py
"""
import os
import json
import time

import numpy as np
from PIL import Image

import genesis as gs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

R2_URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01.urdf"
TABLE_USD = "/home/xujinlong/test/InternUtopia/internutopia/assets/scenes/demo_scenes/franka_mocap_teleop/table_scene.usd"

RES = (1280, 720)
FOV = 60.0  # 垂直视场角(度)
N_STEPS = 10


def to_np(x):
    """torch tensor / list / ndarray -> numpy (CUDA tensor 先 .cpu())"""
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def pinhole_intrinsics(res, fov_deg):
    """pinhole 相机内参: fov 为垂直视场角, 像素方形。"""
    w, h = res
    fy = h / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    fx = fy
    return np.array([[fx, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]])


def main():
    t0 = time.time()
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1 / 60.0, substeps=4),
        show_viewer=False,
    )

    # 1) R2 机器人 (URDF-first 路线)
    r2 = scene.add_entity(
        morph=gs.morphs.URDF(file=R2_URDF),
        surface=gs.surfaces.Default(),
    )

    # 2) table_scene (USD 场景, MDL 材质会回退 displayColor)
    #    collision=False: 纯视觉加载, 跳过碰撞 SDF 预处理
    #    (复杂 USD 碰撞几何会触发 Genesis SDF 预处理崩溃 — 阶段 0 探明的问题)
    table_entities = scene.add_stage(
        morph=gs.morphs.USD(file=TABLE_USD, collision=False),
        vis_mode="visual",
    )

    # 3) 独立带碰撞地面 — 让 R2 能站住, 而不是自由落体穿模
    #    (table 场景的 ground 也因 collision=False 无碰撞, 故单独补一个)
    ground = scene.add_entity(
        morph=gs.morphs.Plane(pos=(0.0, 0.0, 0.0)),
        surface=gs.surfaces.Default(),
    )

    # 4) 相机
    cam = scene.add_camera(
        model="pinhole",
        res=RES,
        pos=(2.5, 1.5, 1.5),
        lookat=(0.0, 0.0, 0.5),
        up=(0.0, 0.0, 1.0),
        fov=FOV,
        near=0.1,
        far=50.0,
    )

    scene.build()
    print(f"[smoke] build 完成, 耗时 {time.time() - t0:.1f}s")
    print(f"[smoke] R2 实体: {r2.name}, 关节数: {r2.n_dofs}")
    print(f"[smoke] table 实体数: {len(table_entities)}")

    K = pinhole_intrinsics(RES, FOV)

    def render_frame(idx):
        rgb, depth, seg, normal = cam.render(rgb=True, depth=True)
        rgb = np.asarray(rgb)
        depth = np.asarray(depth, dtype=np.float32)

        rgb_path = os.path.join(OUT, f"rgb_{idx:04d}.png")
        depth_path = os.path.join(OUT, f"depth_{idx:04d}.npy")
        Image.fromarray(rgb).save(rgb_path)
        np.save(depth_path, depth)

        # 外参: 相机在世界系下的 4x4 变换
        T = to_np(cam.get_transform()).astype(np.float64)
        meta = {
            "frame": idx,
            "scene_id": "table_scene_demo",
            "camera": {
                "model": "pinhole",
                "res": list(RES),
                "fov_deg": FOV,
                "pos": to_np(cam.get_pos()).tolist(),
                "quat": to_np(cam.get_quat()).tolist(),
                "transform_4x4": T.tolist(),
                "intrinsics_K": K.tolist(),
            },
            "depth": {
                "unit": "meter",
                "dtype": "float32",
                "min": float(depth.min()),
                "max": float(depth.max()),
                "mean": float(depth.mean()),
            },
        }
        with open(os.path.join(OUT, f"meta_{idx:04d}.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[smoke] frame {idx}: rgb {rgb.shape} {rgb.dtype} -> {rgb_path}")
        print(f"[smoke] frame {idx}: depth {depth.shape} {depth.dtype} "
              f"min={depth.min():.3f} max={depth.max():.3f} -> {depth_path}")
        return rgb, depth

    # 帧 0: build 后立即渲染 (验证加载 + 渲染管线)
    render_frame(0)

    # 步进 N 步, 验证物理仿真不崩 (里程碑 b 的预检)
    for i in range(N_STEPS):
        scene.step()
    z_after = to_np(r2.get_pos())[2]
    print(f"[smoke] 步进 {N_STEPS} 步完成, R2 基座高度: {z_after:.3f} m (应 ≈0, 穿模则为负)")

    # 帧 1: 步进后渲染
    render_frame(1)

    print(f"[smoke] 全部完成, 总耗时 {time.time() - t0:.1f}s, 输出目录: {OUT}")


if __name__ == "__main__":
    main()
