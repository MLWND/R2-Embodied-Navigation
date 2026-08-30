#!/usr/bin/env python
"""验证: 手臂自然下垂(0位姿) + 关节 kp 保持, 手不插地 + 6指灵巧手"""
import numpy as np
import genesis as gs
from PIL import Image

URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf"
OUT = "/home/xujinlong/test/genesis_smoke/output"

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
cam = scene.add_camera(res=(800, 600), pos=(2.5, 0, 1.2), lookat=(0, 0, 0.9), fov=55)
r2 = scene.add_entity(morph=gs.morphs.URDF(file=URDF), surface=gs.surfaces.Default())
ground = scene.add_entity(morph=gs.morphs.Plane(pos=(0,0,0)), surface=gs.surfaces.Default())
scene.build()
print(f"[nat] build 完成, n_dofs={r2.n_dofs}")

# 所有关节设 kp/kv (手臂/手指保持 0 位姿, 轮子自由)
n = r2.n_dofs
kp = np.zeros(n); kv = np.zeros(n)
for i in range(n):
    kp[i] = 2000.0; kv[i] = 100.0
# 轮子: 0 刚度 (自由)
for name in ['wheel_left_joint', 'wheel_right_joint']:
    idx = flat(r2.get_joint(name).dofs_idx_local).astype(int).tolist()
    for j in idx: kp[j] = 0.0; kv[j] = 100.0
r2.set_dofs_kp(kp.tolist()); r2.set_dofs_kv(kv.tolist())

# settle
for _ in range(200):
    scene.step()

links = scene.rigid_solver.links
name2idx = {l.name: i for i, l in enumerate(links)}

# 手/手指位置
for name in ['hand_left_Link', 'thumb_flex_left_Link', 'index_left_Link', 'pinky_left_Link',
             'hand_right_Link', 'thumb_flex_right_Link']:
    if name in name2idx:
        p = to_np(links[name2idx[name]].get_pos()).reshape(-1)
        print(f"[nat] {name}: z={p[2]:.4f}")

# 接触
c = r2.get_contacts(with_entity=ground)
from collections import Counter
names = [links[int(c['link_b'][i])].name for i in range(len(c['position']))]
print(f"[nat] 接地: {dict(Counter(names))}")
print(f"[nat] base pos = {to_np(r2.get_pos()).reshape(-1).round(4).tolist()}")

rgb, _, _, _ = cam.render(rgb=True, depth=True)
Image.fromarray(np.asarray(rgb)).save(f"{OUT}/natural_pose.png")
print(f"[nat] 已保存 natural_pose.png")
print("[nat] 完成")
