#!/usr/bin/env python
"""GRScenes 场景加载测试: 用 Genesis 加载 home_scenes 场景 USD"""
import os
# 机器 112 核, usd-core 会按核数起工作线程; 用户线程配额(ulimit -u 1024)
# 不够时 pthread_create 失败会导致堆损坏/段错误. 必须在 import genesis/pxr 前设置.
os.environ.setdefault('PXR_WORK_THREAD_LIMIT', '8')

import sys
import numpy as np
import genesis as gs
from PIL import Image

SCENE = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/xujinlong/test/GRScenes/scenes/GRScenes-100/home_scenes/scenes/MWBGLKQKTKJZ2AABAAAAABA8_usd/start_result_raw.usd"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/xujinlong/test/genesis_smoke/output/grscene_test.png"

gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/60.0, substeps=4), show_viewer=False)
cam = scene.add_camera(res=(800, 600), pos=(2.0, 2.0, 1.5), lookat=(0, 0, 0.5), fov=60)
# 纯视觉加载 (避免复杂碰撞 SDF 崩溃)
scene.add_entity(morph=gs.morphs.USD(file=SCENE, collision=False))
scene.build()
print(f"[gs] build 完成: {SCENE.split('/')[-2]}")

rgb, _, _, _ = cam.render(rgb=True, depth=True)
Image.fromarray(np.asarray(rgb)).save(OUT)
print(f"[gs] 已保存 {OUT}")
print("[gs] 完成")
