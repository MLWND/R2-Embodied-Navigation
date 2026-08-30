#!/usr/bin/env python
"""生成手掌 mesh (palm_left/right.stl): 掌心鼓起 + 拇指根部 + 四指根部 + 腕部过渡
坐标系: hand link 局部系, +z 朝上(连接手腕), 手指从 -z 伸出, 拇指在 -x 侧"""
import trimesh
import numpy as np

OUT_DIR = '/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/meshes'

def make_palm():
    parts = []
    # 1. 手掌主体 (box): 宽0.1(x) 深0.06(y) 高0.1(z), 中心 (0,0,-0.09) -> z∈[-0.14,-0.04]
    palm = trimesh.creation.box(extents=(0.1, 0.06, 0.1))
    palm.apply_translation([0, 0, -0.09])
    parts.append(palm)

    # 2. 掌心鼓起 (椭球, 朝 -z 鼓起): 中心 (0,0,-0.11), 加大
    knuckle = trimesh.creation.icosphere(subdivisions=2, radius=0.03)
    knuckle.apply_scale([2.0, 1.5, 1.2])  # 椭球: 宽>深>高
    knuckle.apply_translation([0, 0, -0.11])
    parts.append(knuckle)

    # 3. 拇指根部 (球, 手掌 -x 侧): 中心 (-0.045, 0, -0.12), 加大
    thumb_root = trimesh.creation.icosphere(subdivisions=2, radius=0.032)
    thumb_root.apply_scale([1.0, 1.4, 1.3])
    thumb_root.apply_translation([-0.045, 0, -0.12])
    parts.append(thumb_root)

    # 4. 四指根部 (4 球, 手掌底部 x 方向): 中心 (x, 0, -0.12), 加大更突出
    for x in [-0.025, 0.0, 0.025, 0.05]:
        fr = trimesh.creation.icosphere(subdivisions=2, radius=0.025)
        fr.apply_scale([1.0, 1.3, 1.2])
        fr.apply_translation([x, 0, -0.12])
        parts.append(fr)

    # 5. 腕部过渡 (圆台, 手掌顶部连接 J7): 从手掌 0.045 渐变到手腕 0.03
    wrist = trimesh.creation.revolve(
        linestring=[[0, 0], [0.045, 0], [0.03, 0.07], [0, 0.07]],
        sections=24,
    )
    wrist.apply_translation([0, 0, -0.01])
    parts.append(wrist)

    mesh = trimesh.util.concatenate(parts)
    return mesh

# 左手
left = make_palm()
left.export(f'{OUT_DIR}/palm_left_Link.STL')
print(f'左手掌: {left.vertices.shape[0]} 顶点, 已保存 palm_left_Link.STL')

# 右手 (x 镜像)
right = left.copy()
right.apply_transform(np.diag([-1.0, 1.0, 1.0, 1.0]))
right.export(f'{OUT_DIR}/palm_right_Link.STL')
print(f'右手掌: {right.vertices.shape[0]} 顶点, 已保存 palm_right_Link.STL')
