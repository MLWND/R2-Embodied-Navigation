#!/usr/bin/env python
"""侧视图: 看轮子与地面关系"""
import numpy as np
from PIL import Image
import genesis as gs

R2_URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf"

def to_np(x):
    if hasattr(x, "cpu"): x = x.cpu()
    if isinstance(x, (tuple, list)): return [to_np(i) for i in x]
    return np.asarray(x)

gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/60.0, substeps=4), show_viewer=False)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=R2_URDF), surface=gs.surfaces.Default(),
                      material=gs.materials.Rigid(friction=5.0))
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default(),
                          material=gs.materials.Rigid(friction=5.0))
# 侧视相机
cam = scene.add_camera(model="pinhole", res=(800, 600), pos=(2.0, 0.0, 0.9), lookat=(0,0,0.5), fov=50)
scene.build()
for _ in range(100):
    scene.step()
print(f"[side] settle base={to_np(r2.get_pos()).round(4).tolist()}")
rgb, depth, seg, normal = cam.render(rgb=True, depth=True)
Image.fromarray(np.asarray(rgb)).save("/home/xujinlong/test/genesis_smoke/output/side_view.png")
print("[side] 已保存 side_view.png")
