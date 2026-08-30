#!/usr/bin/env python
"""轮子可见性验证: 近距离相机看 R2 底盘底部。"""
import os
import numpy as np
from PIL import Image
import genesis as gs

R2_URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01.urdf"
OUT = "/home/xujinlong/test/genesis_smoke/output"
os.makedirs(OUT, exist_ok=True)

gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1 / 60.0, substeps=4), show_viewer=False)

r2 = scene.add_entity(morph=gs.morphs.URDF(file=R2_URDF), surface=gs.surfaces.Default())
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0.0, 0.0, 0.0)), surface=gs.surfaces.Default())

# 三个视角: 左侧看轮子 / 正前方 / 正下(看底盘底部)
cam_left = scene.add_camera(model="pinhole", res=(800, 600), pos=(0.9, 0.55, 0.25), lookat=(0.0, 0.0, 0.2), fov=50)
cam_front = scene.add_camera(model="pinhole", res=(800, 600), pos=(1.5, 0.0, 0.4), lookat=(0.0, 0.0, 0.2), fov=50)

scene.build()
print("[wheel] build 完成")

for name, cam in [("left", cam_left), ("front", cam_front)]:
    rgb, depth, seg, normal = cam.render(rgb=True, depth=True)
    rgb = np.asarray(rgb)
    Image.fromarray(rgb).save(os.path.join(OUT, f"wheel_view_{name}.png"))
    print(f"[wheel] {name} 视角已保存")

print("[wheel] 完成")
