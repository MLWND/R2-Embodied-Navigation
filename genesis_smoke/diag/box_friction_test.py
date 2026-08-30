#!/usr/bin/env python
"""box 摩擦测试: 给 box 水平初速度, 看是否被摩擦减速 (验证摩擦约束是否工作)"""
import numpy as np
import genesis as gs

def to_np(x):
    if hasattr(x, "cpu"): x = x.cpu()
    if isinstance(x, (tuple, list)): return [to_np(i) for i in x]
    return np.asarray(x)

def flat(x):
    parts = []
    for i in to_np(x):
        parts.append(np.asarray(i).reshape(-1))
    return np.concatenate(parts)

gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/60.0, substeps=4), show_viewer=False)
box = scene.add_entity(morph=gs.morphs.Box(size=(0.5, 0.5, 0.5), pos=(0, 0, 0.3)),
                       surface=gs.surfaces.Default())
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default())
scene.build()
print("[bf] build 完成")

for _ in range(100): scene.step()
print(f"[bf] settle 后 box pos={to_np(box.get_pos()).round(4).tolist()}")

# 给 box 水平速度 1 m/s (x 方向)
box.set_dofs_velocity([1.0, 0, 0, 0, 0, 0])
for i in range(60):
    scene.step()
    if i % 10 == 0:
        p = to_np(box.get_pos()).reshape(-1)
        v = to_np(box.get_dofs_velocity()).reshape(-1)
        print(f"[bf] step {i}: x={p[0]:.4f} vx={v[0]:.3f}")

p = to_np(box.get_pos()).reshape(-1)
print(f"[bf] 结束: x={p[0]:.4f} (有摩擦应 <0.1m 停下, 无摩擦应 ~1m)")
print("[bf] 完成")
