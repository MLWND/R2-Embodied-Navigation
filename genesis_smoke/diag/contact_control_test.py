#!/usr/bin/env python
"""接触系统对照实验(30s): Plane+球+盒子应静止在 z=0.100.
用途: 排查场景碰撞前先确认 Genesis 接触本身正常."""
import os
os.environ.setdefault('PXR_WORK_THREAD_LIMIT', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
import genesis as gs

def h(x): return np.asarray(x.cpu() if hasattr(x, 'cpu') else x).reshape(-1)

gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/60.0, substeps=4), show_viewer=False)
scene.add_entity(morph=gs.morphs.Plane())
ball = scene.add_entity(morph=gs.morphs.Sphere(radius=0.1, pos=(0.3, 0.3, 1.0)), material=gs.materials.Rigid(rho=500))
box = scene.add_entity(morph=gs.morphs.Box(size=(0.2,0.2,0.2), pos=(0.8, 0.8, 1.0)), material=gs.materials.Rigid(rho=500))
scene.build()
for i in range(150):
    scene.step()
    if (i+1) % 50 == 0:
        print(f"[ctl] step {i+1}: ball z={float(h(ball.get_pos())[2]):.3f} box z={float(h(box.get_pos())[2]):.3f}", flush=True)
