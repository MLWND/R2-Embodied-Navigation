#!/usr/bin/env python
"""渲染当前 URDF settle 后状态 (不抬臂), 确认手臂/身体是否掉地"""
import sys
import numpy as np
import genesis as gs
from PIL import Image

URDF = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/xujinlong/test/genesis_smoke/output/pose_check.png"

gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/60.0, substeps=4), show_viewer=False)
cam = scene.add_camera(res=(640, 480), pos=(2.5, 0, 1.2), lookat=(0, 0, 0.5), fov=60)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=URDF), surface=gs.surfaces.Default())
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default())
scene.build()
print(f"[chk] build 完成, URDF={URDF.split('/')[-1]}")

for _ in range(200):
    scene.step()

rgb, _, _, _ = cam.render(rgb=True, depth=True)
Image.fromarray(np.asarray(rgb)).save(OUT)
print(f"[chk] 已保存 {OUT}")
print(f"[chk] base pos = {np.asarray(r2.get_pos().cpu()).reshape(-1).round(4).tolist()}")
print("[chk] 完成")
