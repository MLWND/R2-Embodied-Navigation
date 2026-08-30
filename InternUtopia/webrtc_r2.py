"""R2 实时流：WebRTC（Kit Remote 客户端）+ MJPEG（浏览器直接看）

浏览器打开: http://10.214.131.225:8080/
"""
import io
import threading
import numpy as np
from PIL import Image

from internutopia.core.config import Config, SimConfig
from internutopia.core.gym_env import Env
from internutopia.core.util import has_display
from internutopia.macros import gm
from internutopia_extension import import_extensions
from internutopia_extension.configs.robots.r2 import (
    R2RobotCfg,
    move_by_speed_cfg,
    left_arm_cfg,
    right_arm_cfg,
    camera_cfg,
)
from internutopia_extension.configs.tasks import SingleInferenceTaskCfg

headless = not has_display()

r2 = R2RobotCfg(
    position=(0.0, 0.0, 0.05),
    controllers=[move_by_speed_cfg, left_arm_cfg, right_arm_cfg],
    sensors=[camera_cfg],
)

config = Config(
    simulator=SimConfig(physics_dt=1 / 240, rendering_dt=1 / 240, use_fabric=False, headless=headless, webrtc=True),
    task_configs=[
        SingleInferenceTaskCfg(
            scene_asset_path=gm.ASSET_PATH + '/scenes/empty.usd',
            scene_scale=(0.01, 0.01, 0.01),
            robots=[r2],
        ),
    ],
)

import_extensions()
env = Env(config)
import omni.replicator.core as rep
obs, _ = env.reset()

# ============ 渲染设置诊断 ============
import carb
settings = carb.settings.get_settings()
for key in ['/rtx/post/aa/op', '/rtx/post/dlss/execMode', '/rtx/post/colorcorr/saturation',
            '/rtx/post/colorcorr/contrast', '/rtx/post/tonemap/op', '/rtx/hydra/rendermode',
            '/rtx/debugView/pixelDebug', '/rtx/raytracing/enabled', '/rtx/pathtracing/enabled']:
    try:
        print(f'[渲染设置] {key} = {settings.get(key)}')
    except Exception as e:
        print(f'[渲染设置] {key} = ERROR {e}')

# 检查场景灯光
from pxr import Usd
import omni.usd
stage = omni.usd.get_context().get_stage()
for prim in stage.Traverse():
    pn = prim.GetTypeName()
    if 'Light' in pn or 'Dome' in pn:
        print(f'[灯光] {prim.GetPath()} type={pn}')
        for attr in prim.GetAttributes():
            print(f'  {attr.GetName()} = {attr.Get()}')

# ============ 第三人称流相机 ============
from pxr import Usd, UsdGeom
import omni.usd
stage = omni.usd.get_context().get_stage()
cam_path = '/StreamCam'
if not stage.GetPrimAtPath(cam_path):
    cam = UsdGeom.Camera.Define(stage, cam_path)
    cam.AddRotateYOp().Set(-90.0)
    cam.CreateFocalLengthAttr(24.0)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(15.2908)

from omni.isaac.core.prims.xform_prim import XFormPrim
stream_cam = XFormPrim(cam_path)

# ============ 主视口绑定流相机（WebRTC 流显示主视口内容） ============
from omni.kit.viewport.utility import get_active_viewport
viewport = get_active_viewport()
if viewport is not None:
    viewport.camera_path = cam_path
    print(f'主视口已绑定到 {cam_path}')
else:
    print('警告: 无活动视口，WebRTC 流可能无法显示画面')

def quat_from_lookat(eye, target, up=(0, 0, 1)):
    fwd = np.array(target) - np.array(eye)
    fwd = fwd / np.linalg.norm(fwd)
    z = -fwd
    x = np.cross(np.array(up), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        w = np.sqrt(tr + 1) / 2
        qx = (R[2, 1] - R[1, 2]) / (4 * w)
        qy = (R[0, 2] - R[2, 0]) / (4 * w)
        qz = (R[1, 0] - R[0, 1]) / (4 * w)
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s; qx = s / 4
            qy = (R[0, 1] + R[1, 0]) / s; qz = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s; qx = (R[0, 1] + R[1, 0]) / s
            qy = s / 4; qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s; qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s; qz = s / 4
    return [w, qx, qy, qz]

stream_cam.set_world_pose(np.array([2.8, 2.2, 1.3]), np.array(quat_from_lookat((2.8, 2.2, 1.3), (0.0, 0.0, 0.8))))

# 流渲染产物（跟随相机）
rp = rep.create.render_product(cam_path, (1280, 720))
anno = rep.AnnotatorRegistry.get_annotator('LdrColor')
anno.attach(rp)
print('流相机已就绪')

# ============ MJPEG HTTP 服务 ============
latest_frame = [None]
frame_lock = threading.Lock()

import http.server

class MJPEGHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        jpg = latest_frame[0]
                    if jpg is not None:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
                    import time
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            html = '''<html><head><title>R2 实时仿真</title></head>
<body style="background:#111;margin:0;display:flex;flex-direction:column;align-items:center;font-family:sans-serif">
<h2 style="color:#ddd">R2 实时仿真画面</h2>
<img src="/stream" style="max-width:95vw;border:2px solid #444;border-radius:8px">
</body></html>'''.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html)

    def log_message(self, *args):
        pass

server = http.server.ThreadingHTTPServer(('0.0.0.0', 8080), MJPEGHandler)
threading.Thread(target=server.serve_forever, daemon=True).start()
print('MJPEG 服务已启动: http://10.214.131.225:8080/')
print('WebRTC 服务已启动: http://10.214.131.225:49100/ (需 Kit Remote 客户端)')
print('按 Ctrl+C 停止')

# ============ 运动循环（真实机器人手臂参数，来自 demo1.py，单位：度→弧度） ============
import numpy as np

def deg(angles):
    """度 → 弧度"""
    return np.radians(angles).tolist()

# demo1.py 真实动作（左臂大幅动作，右臂回原点——官方验证过的安全动作）
POSE_COUNT = deg([-174.74, 127.50, 96.14, -84.34, -82.98, -24.95, 78.53])  # 数数
POSE_WAVE1 = deg([-144.74, 44.13, 96.14, -48.34, -60.98, -8.23, 45.53])    # 单手挥1
POSE_WAVE2 = deg([-144.74, 57.93, 96.14, -41.34, -60.98, 5.43, 45.53])     # 单手挥2
POSE_WAVE3 = deg([-150.74, 28.06, 96.14, -48.34, -60.98, -12.97, 45.53])   # 单手挥3
POSE_BOW = deg([0.0, 0.0, 0.0, -76.91, 0.0, 0.0, 0.0])                     # 鞠躬
ZERO7 = [0.0] * 7

i = 0
try:
    while True:
        if i < 240 * 3:
            # 前进 3 秒
            action = {move_by_speed_cfg.name: [0.3, 0.0]}
        elif i < 240 * 6:
            # 后退 3 秒（回原点附近）
            action = {move_by_speed_cfg.name: [-0.3, 0.0]}
        elif i < 240 * 9:
            # 原地旋转 3 秒
            action = {move_by_speed_cfg.name: [0.0, 0.8]}
        elif i < 240 * 12:
            # 数数（左臂大幅动作）
            action = {left_arm_cfg.name: [POSE_COUNT], right_arm_cfg.name: [ZERO7]}
        elif i < 240 * 15:
            # 单手挥 1
            action = {left_arm_cfg.name: [POSE_WAVE1], right_arm_cfg.name: [ZERO7]}
        elif i < 240 * 18:
            # 单手挥 2
            action = {left_arm_cfg.name: [POSE_WAVE2], right_arm_cfg.name: [ZERO7]}
        elif i < 240 * 21:
            # 单手挥 3
            action = {left_arm_cfg.name: [POSE_WAVE3], right_arm_cfg.name: [ZERO7]}
        elif i < 240 * 24:
            # 双手挥（左右相同）
            action = {left_arm_cfg.name: [POSE_WAVE1], right_arm_cfg.name: [POSE_WAVE1]}
        elif i < 240 * 27:
            # 鞠躬
            action = {left_arm_cfg.name: [POSE_BOW], right_arm_cfg.name: [POSE_BOW]}
        elif i < 240 * 30:
            # 手臂回原点
            action = {left_arm_cfg.name: [ZERO7], right_arm_cfg.name: [ZERO7]}
        else:
            i = 0
            continue
        obs, _, _, _, _ = env.step(action=action)
        i += 1
        # 每 4 帧推一帧流
        if i % 4 == 0:
            rgba = anno.get_data()
            if rgba is not None and len(rgba) > 0:
                arr = np.array(rgba)
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                img = Image.fromarray(arr).convert('RGB')
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=80)
                with frame_lock:
                    latest_frame[0] = buf.getvalue()
        if i % 240 == 0:
            print(f'第{i//240}s: 位置=({obs["position"][0]:.2f},{obs["position"][1]:.2f})')
except KeyboardInterrupt:
    pass

env.close()
print('DONE')
