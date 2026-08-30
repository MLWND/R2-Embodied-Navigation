#!/usr/bin/env python
"""
Track B: 物体碰撞方案实验
目标: 验证 convexify=True 能否让 table 场景(含复杂微波炉几何)安全开碰撞,
      且机器人撞到物体时被物理阻挡(不穿模)。
"""
import os
import numpy as np
from PIL import Image
import genesis as gs

R2_URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01.urdf"
TABLE_USD = "/home/xujinlong/test/InternUtopia/internutopia/assets/scenes/demo_scenes/franka_mocap_teleop/table_scene.usd"
OUT = "/home/xujinlong/test/genesis_smoke/output"
os.makedirs(OUT, exist_ok=True)

def to_np(x):
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1 / 60.0, substeps=4), show_viewer=False)

# 机器人放在微波炉(x=0.02, y=0.38, z=1.19)正下方偏侧, 上半身会顶到微波炉
r2 = scene.add_entity(
    morph=gs.morphs.URDF(file=R2_URDF, pos=(0.0, 0.30, 0.0)),
    surface=gs.surfaces.Default(),
)
ground = scene.add_entity(
    morph=gs.morphs.Plane(pos=(0.0, 0.0, 0.0)),
    surface=gs.surfaces.Default(),
)
# 关键: convexify=True 把碰撞几何凸包化(简单几何), 避开复杂 instanced 几何的 SDF 崩溃
table_entities = scene.add_stage(
    morph=gs.morphs.USD(file=TABLE_USD, convexify=True),
    vis_mode="visual",
)

cam = scene.add_camera(
    model="pinhole", res=(1280, 720), pos=(2.5, 1.5, 1.5),
    lookat=(0.0, 0.0, 0.5), fov=60,
)
scene.build()
print(f"[collision] build 成功, table 实体数: {len(table_entities)}")

def save_frame(idx):
    rgb, depth, seg, normal = cam.render(rgb=True, depth=True)
    Image.fromarray(np.asarray(rgb)).save(os.path.join(OUT, f"collision_rgb_{idx:04d}.png"))
    print(f"[collision] frame {idx} 已保存, R2 基座 z={float(to_np(r2.get_pos()).reshape(-1)[2]):.3f}")

save_frame(0)

# 步进 60 步: 若碰撞生效, 机器人被微波炉顶住, 位置不会穿越到物体内部
for i in range(60):
    scene.step()
    if i % 20 == 0:
        print(f"[collision] step {i}, R2 pos: {to_np(r2.get_pos()).round(3).tolist()}")

save_frame(1)
print("[collision] 完成")
