#!/usr/bin/env python
"""从 GRScenes navigation.usd 提取碰撞子树 -> 纯碰撞 USD

GRScenes navigation 变体把 CollisionAPI 标在模型 Instance 上(PhysX 语义=
整个子树参与碰撞), 但 Genesis 的 mesh 级碰撞/视觉分离只认名字模式,
不看 CollisionAPI, 直接加载会把 4000+ 视觉 mesh 全变碰撞体.

本脚本把 /Root/Meshes 下不含 CollisionAPI 的模型子树 deactivate,
只保留碰撞模型, 导出 *_collision.usd. 配合:

    scene.add_entity(morph=gs.morphs.USD(
        file='..._collision.usd', collision=True, visualization=False, fixed=True))

用法: python grscene_nav_collision.py [navigation.usd] [输出.usd]
"""
import os
import sys
os.environ.setdefault('PXR_WORK_THREAD_LIMIT', '8')
from pxr import Usd, UsdPhysics

SCENE = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/xujinlong/test/GRScenes/scenes/GRScenes-100/home_scenes/scenes/MWBGLKQKTKJZ2AABAAAAABA8_usd/start_result_navigation.usd"
OUT = sys.argv[2] if len(sys.argv) > 2 else SCENE.replace('.usd', '_collision.usd')
# 可选: 只保留原点 RADIUS 米内的碰撞模型 (米, USD 单位换算 mpu=0.01), 加速加载
RADIUS = float(sys.argv[3]) if len(sys.argv) > 3 else None

stage = Usd.Stage.Open(SCENE)

# 非几何/非碰撞的顶层分支直接关掉 (HDR球/灯光等, 纯碰撞场景用不到)
for extra in ('/Root/__default_setting', '/Root/lights', '/Root/Looks'):
    p = stage.GetPrimAtPath(extra)
    if p.IsValid():
        p.SetActive(False)

kept = dropped = 0
meshes_root = stage.GetPrimAtPath('/Root/Meshes')
if not meshes_root.IsValid():
    sys.exit(f'[err] 找不到 /Root/Meshes: {SCENE}')

# 模型节点名为 model_<hash>_<N>, 位于 /Root/Meshes/<cat>/<subcat>/model_* (层级可能少一层)
def is_model(prim):
    return prim.GetName().startswith('model_') and prim.GetChildren()

for prim in Usd.PrimRange(meshes_root):
    if not is_model(prim):
        continue
    # 只要最外层 model 节点 (嵌套时子 model 随父级一起处理)
    parent = prim.GetParent()
    nested = False
    while parent.IsValid() and str(parent.GetPath()) != '/Root/Meshes':
        if is_model(parent):
            nested = True
            break
        parent = parent.GetParent()
    if nested:
        continue
    has_col = any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(prim))
    if has_col and RADIUS is not None:
        # 模型任一顶点进入半径圆内才保留 (mpu=0.01, 单位 cm -> m)
        import numpy as np
        from pxr import UsdGeom
        xf = UsdGeom.XformCache()
        inside = False
        for c in Usd.PrimRange(prim):
            if not c.IsA(UsdGeom.Mesh):
                continue
            pts = c.GetAttribute('points').Get()
            if not pts or not len(pts):
                continue
            T = np.array(xf.GetLocalToWorldTransform(c))
            P = np.hstack([np.array(pts, dtype=np.float64), np.ones((len(pts), 1))])
            W = (T @ P.T).T[:, :3] / 100.0
            if ((W[:, 0])**2 + (W[:, 1])**2 <= (RADIUS + 1.0)**2).any():
                inside = True
                break
        if not inside:
            has_col = False
    if has_col:
        kept += 1
    else:
        prim.SetActive(False)
        dropped += 1

print(f'碰撞模型保留 {kept}, 非碰撞模型禁用 {dropped}')
stage.GetRootLayer().Export(OUT)
print(f'保存到: {OUT}')
