#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-Time 3D SAPIEN Robot Navigation Streaming Server (HTTP / MJPEG)
采用高保真 3D 拟人物理渲染 + 差速平稳动力学约束 + 3D 追随视口与 2D 拓扑雷达 HUD
"""

import os
import sys
import time
import socket
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import numpy as np
import cv2
import scipy.spatial.transform as spt
from PIL import Image, ImageDraw

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from grscenes_importer import extract_grscene_layout
from scene_semantic_graph import build_scene_semantic_graph
from ppo_nav_agent import PPONavAgent
import torch
import sapien
import sapien.render

global_frame_jpeg = None
lock = threading.Lock()

def compute_lookat_quat(cam_pos, target_pos):
    forward = np.array(target_pos) - np.array(cam_pos)
    dist = np.linalg.norm(forward)
    forward /= max(dist, 1e-5)
    up = np.array([0.0, 0.0, 1.0])
    left = np.cross(up, forward)
    left /= max(np.linalg.norm(left), 1e-5)
    real_up = np.cross(forward, left)
    R = np.eye(3)
    R[:, 0] = forward
    R[:, 1] = left
    R[:, 2] = real_up
    quat = spt.Rotation.from_matrix(R).as_quat()
    return [quat[3], quat[0], quat[1], quat[2]]

def yaw_to_quat(yaw_deg):
    r = spt.Rotation.from_euler('z', yaw_deg, degrees=True)
    q = r.as_quat()
    return [q[3], q[0], q[1], q[2]]

def simulate_lidar(scene, pos, yaw_deg, robot_shapes, num_rays=48, max_dist=10.0):
    yaw_rad = np.radians(yaw_deg)
    angles = np.linspace(-np.pi, np.pi, num_rays, endpoint=False)
    ranges = np.full(num_rays, max_dist, dtype=np.float32)
    rel_angles_deg = []
    
    for i, a in enumerate(angles):
        ray_angle = yaw_rad + a
        dx, dy = np.cos(ray_angle), np.sin(ray_angle)
        rel_angles_deg.append(np.degrees(a))
        
        hit = scene.physx_system.raycast(
            np.array([pos[0], pos[1], 0.25], dtype=np.float32),
            np.array([dx, dy, 0.0], dtype=np.float32),
            float(max_dist)
        )
        if hit is not None and hit.shape not in robot_shapes:
            ranges[i] = min(hit.distance, max_dist)
            
    return ranges, np.array(rel_angles_deg)

class MJPEGStreamHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>🤖 R2 人形机器人 3D 仿真自主导航实时监控台</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #0b0f19; color: #f8fafc; margin: 0; padding: 20px; text-align: center; }
        h1 { color: #38bdf8; margin-bottom: 5px; font-size: 22px; }
        .badge { display: inline-block; background: #059669; color: white; padding: 4px 14px; border-radius: 20px; font-weight: bold; font-size: 13px; animation: pulse 2s infinite; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.6; } 100% { opacity: 1; } }
        .container { max-width: 1280px; margin: 0 auto; background: #1e293b; padding: 15px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.6); }
        .stream-box { border: 2px solid #334155; border-radius: 8px; overflow: hidden; background: #000; display: inline-block; }
        img { width: 100%; max-width: 1200px; height: auto; display: block; }
        .downloads { margin: 15px 0 10px; display: flex; justify-content: center; gap: 15px; }
        .btn-download { display: inline-flex; align-items: center; background: #0284c7; color: white; padding: 8px 18px; border-radius: 6px; text-decoration: none; font-weight: bold; font-size: 14px; transition: background 0.2s; }
        .btn-download:hover { background: #0369a1; }
        .btn-download.gif { background: #7c3aed; }
        .btn-download.gif:hover { background: #6d28d9; }
        .footer { margin-top: 15px; color: #94a3b8; font-size: 13px; line-height: 1.6; }
    </style>
</head>
<body>
    <div class="container">
        <div style="margin-bottom: 12px;">
            <h1>🤖 R2 人形机器人 3D 仿真自主导航实时监控台</h1>
            <span class="badge">● SAPIEN 3D VULKAN LIVE (15 FPS)</span>
        </div>
        <div class="downloads">
            <a href="/r2_robot_3d_navigation_demo.mp4" class="btn-download" download>📥 下载 3D MP4 视频 (1.8 MB)</a>
            <a href="/r2_robot_3d_navigation_demo.gif" class="btn-download gif" download>🎞️ 下载 3D 动态 GIF (2.5 MB)</a>
            <a href="/r2_robot_3d_appearance_hd.png" class="btn-download" download style="background:#059669;">📸 下载 3D 超清写真图</a>
        </div>
        <div class="stream-box">
            <img src="/stream.mjpeg" alt="R2 3D Real-Time Stream" />
        </div>
        <div class="footer">
            <p><strong>左侧主视口</strong>：SAPIEN 3.0.3 第三人称 3D 动态追随漫游视口（实时 3D 机器人本体、轮组旋转与家具环境）</p>
            <p><strong>右侧上视口</strong>：2D 语义拓扑户型与 48 维 LiDAR 实时动态测距扫描扇面 | <strong>右侧下视口</strong>：动力学遥测 HUD 与阶段状态</p>
        </div>
    </div>
</body>
</html>"""
            self.wfile.write(html.encode('utf-8'))
            
        elif self.path == '/stream.mjpeg':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with lock:
                        frame = global_frame_jpeg
                    if frame is not None:
                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                    time.sleep(0.04)
            except Exception:
                pass
                
        elif self.path.startswith('/download/') or self.path.endswith('.mp4') or self.path.endswith('.gif') or self.path.endswith('.png'):
            filename = os.path.basename(self.path.split('?')[0])
            filepath = os.path.join('/home/xujinlong/test/output', filename)
            if os.path.exists(filepath):
                self.send_response(200)
                if filename.endswith('.mp4'):
                    self.send_header('Content-Type', 'video/mp4')
                elif filename.endswith('.gif'):
                    self.send_header('Content-Type', 'image/gif')
                elif filename.endswith('.png'):
                    self.send_header('Content-Type', 'image/png')
                else:
                    self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
                self.send_header('Content-Length', str(os.path.getsize(filepath)))
                self.end_headers()
                with open(filepath, 'rb') as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, f"File {filename} not found")
        else:
            self.send_error(404)

def run_simulation_loop():
    global global_frame_jpeg
    
    usd_path = "/home/xujinlong/test/GRScenes/scenes/GRScenes-100/home_scenes/scenes/MWAX5JYKTKJZ2AABAAAAAEI8_usd/start_result_navigation.usd"
    targets_info, walls_info = extract_grscene_layout(usd_path)
    graph = build_scene_semantic_graph(usd_path)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ppo_agent = PPONavAgent(obs_dim=52, act_dim=2).to(device)
    ppo_model_path = "/home/xujinlong/test/maniskill_r2/models/ppo_r2_nav_69scenes_full.pt"
    if os.path.exists(ppo_model_path):
        ppo_agent.load_state_dict(torch.load(ppo_model_path, map_location=device))
    ppo_agent.eval()
    
    instruction = "去厨房里的冰箱"
    start_pos = [3.5, 0.0]
    plan = graph.plan_topological_route(start_pos, instruction)
    waypoints = plan["waypoints"]
    target_obj = plan["target_object"]
    target_room = plan["target_room"]
    final_goal = plan["final_goal"]
    
    while True:
        scene = sapien.Scene()
        scene.set_timestep(0.01)
        scene.set_ambient_light([0.48, 0.48, 0.52])
        scene.add_directional_light([2.0, -1.5, -2.5], [1.2, 1.15, 1.1])
        scene.add_directional_light([-2.0, 1.5, -2.0], [0.6, 0.65, 0.7])
        scene.add_point_light([0.0, 0.0, 3.5], [1.2, 1.2, 1.2])
        scene.add_point_light([3.5, 0.0, 3.5], [1.0, 1.0, 1.0])
        scene.add_ground(altitude=0, render=True)
        
        mat_wall = sapien.render.RenderMaterial(base_color=[0.88, 0.90, 0.92, 1.0], roughness=0.65)
        mat_sofa = sapien.render.RenderMaterial(base_color=[0.18, 0.35, 0.58, 1.0], roughness=0.55)
        mat_bed = sapien.render.RenderMaterial(base_color=[0.28, 0.52, 0.42, 1.0], roughness=0.5)
        mat_table = sapien.render.RenderMaterial(base_color=[0.72, 0.48, 0.28, 1.0], roughness=0.4)
        mat_fridge = sapien.render.RenderMaterial(base_color=[0.85, 0.88, 0.92, 1.0], metallic=0.75, roughness=0.25)
        mat_target = sapien.render.RenderMaterial(base_color=[0.95, 0.20, 0.20, 1.0], metallic=0.5, roughness=0.3)
        mat_other = sapien.render.RenderMaterial(base_color=[0.58, 0.60, 0.65, 1.0], roughness=0.6)
        
        for name, info in targets_info.items():
            b = scene.create_actor_builder()
            sx, sy, sz = info["size"]
            is_target = (name == target_obj.name)
            n_low = name.lower()
            if is_target: m = mat_target
            elif "bed" in n_low: m = mat_bed
            elif "sofa" in n_low or "couch" in n_low: m = mat_sofa
            elif "table" in n_low or "desk" in n_low: m = mat_table
            elif "fridge" in n_low or "refrigerator" in n_low: m = mat_fridge
            else: m = mat_other
            
            b.add_box_collision(half_size=[sx/2, sy/2, sz/2])
            b.add_box_visual(half_size=[sx/2, sy/2, sz/2], material=m)
            actor = b.build_static(name=name)
            actor.set_pose(sapien.Pose(p=info["pos"]))
            
        for i, w in enumerate(walls_info):
            b = scene.create_actor_builder()
            sx, sy, sz = w["size"]
            b.add_box_collision(half_size=[sx/2, sy/2, sz/2])
            b.add_box_visual(half_size=[sx/2, sy/2, sz/2], material=mat_wall)
            actor = b.build_static(name=f"wall_{i+1}")
            actor.set_pose(sapien.Pose(p=w["pos"]))
            
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        robot = loader.load("/tmp/r2_3d_vis.urdf")
        
        robot_shapes = []
        for l in robot.get_links():
            robot_shapes.extend(l.get_collision_shapes())
            
        cam_3d = scene.add_camera(name="chase_3d", width=720, height=540, fovy=np.radians(52), near=0.1, far=50.0)
        
        robot_pos = np.array([start_pos[0], start_pos[1]], dtype=np.float32)
        robot_yaw = 180.0
        robot.set_pose(sapien.Pose(p=[robot_pos[0], robot_pos[1], 0.0], q=yaw_to_quat(robot_yaw)))
        
        current_stage_idx = 0
        trajectory = [(robot_pos[0], robot_pos[1])]
        dt = 0.04
        act_v, act_w = 0.0, 0.0
        
        for step in range(500):
            ranges, rel_angles = simulate_lidar(scene, robot_pos, robot_yaw, robot_shapes)
            min_clearance = float(np.min(ranges))
            
            curr_wp = waypoints[current_stage_idx]
            curr_target = curr_wp["pos"]
            dist_to_curr = np.sqrt((robot_pos[0] - curr_target[0])**2 + (robot_pos[1] - curr_target[1])**2)
            dist_to_final = np.sqrt((robot_pos[0] - final_goal[0])**2 + (robot_pos[1] - final_goal[1])**2)
            
            is_final_stage = (current_stage_idx == len(waypoints) - 1)
            if dist_to_final < 0.75:
                time.sleep(2.0)
                break
            elif dist_to_curr < (0.45 if not is_final_stage else 0.75):
                if is_final_stage:
                    time.sleep(2.0)
                    break
                else:
                    current_stage_idx += 1
                    curr_target = waypoints[current_stage_idx]["pos"]
                    dist_to_curr = np.sqrt((robot_pos[0] - curr_target[0])**2 + (robot_pos[1] - curr_target[1])**2)
                    
            dx = curr_target[0] - robot_pos[0]
            dy = curr_target[1] - robot_pos[1]
            f_att = np.array([dx / max(dist_to_curr, 0.05), dy / max(dist_to_curr, 0.05)])
            
            f_rep = np.zeros(2, dtype=np.float32)
            for r_val, rel_a in zip(ranges, rel_angles):
                if 0.05 < r_val < 0.42:
                    beam_rad = np.radians(robot_yaw + rel_a)
                    beam_dir = np.array([np.cos(beam_rad), np.sin(beam_rad)])
                    f_rep -= 0.15 * (1.0/r_val - 1.0/0.42) / (r_val**2) * beam_dir
                    
            f_total = f_att + f_rep
            target_angle_deg = np.degrees(np.arctan2(f_total[1], f_total[0]))
            heading_err = (target_angle_deg - robot_yaw + 180) % 360 - 180
            
            obs = np.zeros(52, dtype=np.float32)
            obs[:48] = np.clip(ranges / 10.0, 0.0, 1.0)
            obs[48] = dist_to_curr / 10.0
            obs[49] = np.radians(heading_err) / np.pi
            obs[50] = np.clip(act_v / 0.55, 0.0, 1.0)
            obs[51] = np.clip(act_w / 1.5, -1.0, 1.0)
            
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            ppo_action = ppo_agent.act_deterministic(obs_t).squeeze(0).cpu().numpy()
            
            ppo_v = float(np.clip((ppo_action[0] + 1.0) * 0.5 * 0.50, 0.0, 0.50))
            ppo_w = float(np.clip(ppo_action[1] * 1.5, -1.5, 1.5))
            exp_w = float(np.clip(2.8 * np.radians(heading_err), -1.5, 1.5))
            exp_align = max(0.0, float(np.cos(np.radians(heading_err))))
            exp_v = float(0.50 * (exp_align ** 2) * np.clip(dist_to_curr / 0.5, 0.35, 1.0))
            
            if abs(heading_err) > 25.0:
                v_cmd = 0.0
                omega_cmd = exp_w
            else:
                alpha = 0.5 if abs(heading_err) < 12.0 else 0.1
                v_cmd = alpha * ppo_v + (1 - alpha) * exp_v
                if exp_align > 0.6:
                    v_cmd = max(v_cmd, 0.32 * exp_align)
                omega_cmd = alpha * ppo_w + (1 - alpha) * exp_w
                
            if min_clearance < 0.35 and abs(heading_err) < 20.0:
                v_cmd = max(v_cmd, 0.28)
                
            act_v = v_cmd
            act_w = omega_cmd
            
            robot_yaw = (robot_yaw + np.degrees(act_w * dt) + 180) % 360 - 180
            yaw_rad = np.radians(robot_yaw)
            robot_pos[0] += act_v * np.cos(yaw_rad) * dt
            robot_pos[1] += act_v * np.sin(yaw_rad) * dt
            trajectory.append((robot_pos[0], robot_pos[1]))
            
            robot.set_pose(sapien.Pose(p=[robot_pos[0], robot_pos[1], 0.0], q=yaw_to_quat(robot_yaw)))
            
            chase_dist = 3.2
            cam_angle = yaw_rad - np.radians(35)
            cam_x = robot_pos[0] + chase_dist * np.cos(cam_angle)
            cam_y = robot_pos[1] + chase_dist * np.sin(cam_angle)
            cam_z = 1.85
            
            target_pt = [robot_pos[0], robot_pos[1], 0.70]
            cam_quat = compute_lookat_quat([cam_x, cam_y, cam_z], target_pt)
            cam_3d.set_pose(sapien.Pose(p=[cam_x, cam_y, cam_z], q=cam_quat))
            
            scene.update_render()
            cam_3d.take_picture()
            rgba = cam_3d.get_picture('Color')
            frame_3d = (np.clip(rgba[:, :, :3], 0, 1) * 255).astype(np.uint8)
            
            side_panel = np.full((540, 480, 3), 24, dtype=np.uint8)
            map_box = side_panel[10:270, 10:470]
            map_box[:] = 16
            scale = 32.0
            cx, cy = 230, 130
            
            for w in walls_info:
                wx, wy = int(cx + w["pos"][0] * scale), int(cy - w["pos"][1] * scale)
                wsx, wsy = int(w["size"][0] * scale / 2), int(w["size"][1] * scale / 2)
                cv2.rectangle(map_box, (wx - wsx, wy - wsy), (wx + wsx, wy + wsy), (100, 100, 110), -1)
                
            for name, info in targets_info.items():
                fx, fy = int(cx + info["pos"][0] * scale), int(cy - info["pos"][1] * scale)
                fsx, fsy = int(info["size"][0] * scale / 2), int(info["size"][1] * scale / 2)
                col = (0, 0, 220) if name == target_obj.name else (140, 100, 60)
                cv2.rectangle(map_box, (fx - fsx, fy - fsy), (fx + fsx, fy + fsy), col, -1)
                
            rx, ry = int(cx + robot_pos[0] * scale), int(cy - robot_pos[1] * scale)
            for r_val, rel_a in zip(ranges[::2], rel_angles[::2]):
                beam_rad = np.radians(robot_yaw + rel_a)
                bx = int(rx + r_val * scale * np.cos(beam_rad))
                by = int(ry - r_val * scale * np.sin(beam_rad))
                cv2.line(map_box, (rx, ry), (bx, by), (40, 180, 40), 1)
                
            if len(trajectory) > 1:
                pts = np.array([[int(cx + p[0]*scale), int(cy - p[1]*scale)] for p in trajectory], np.int32)
                cv2.polylines(map_box, [pts], False, (0, 230, 255), 2)
                
            cv2.circle(map_box, (rx, ry), 7, (0, 255, 255), -1)
            hx = int(rx + 14 * np.cos(yaw_rad))
            hy = int(ry - 14 * np.sin(yaw_rad))
            cv2.arrowedLine(map_box, (rx, ry), (hx, hy), (0, 0, 255), 2, tipLength=0.4)
            cv2.putText(map_box, "2D Semantic Map & 48D LiDAR", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            hud_box = side_panel[280:530, 10:470]
            hud_box[:] = 30
            cv2.putText(hud_box, "Dynamic Telemetry HUD", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 1)
            cv2.putText(hud_box, f"Goal: {target_obj.zh_name} ({target_room.zh_name})", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(hud_box, f"Stage: {current_stage_idx+1}/{len(waypoints)} -> {waypoints[current_stage_idx]['desc']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1)
            cv2.putText(hud_box, f"Linear Vel (v): {act_v:.2f} m/s", (10, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(hud_box, f"Angular Vel (w): {act_w:+.2f} rad/s", (10, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(hud_box, f"Goal Distance: {dist_to_final:.2f} m", (10, 186), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(hud_box, f"LiDAR Clearance: {min_clearance:.2f} m", (10, 218), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0) if min_clearance > 0.25 else (0, 165, 255), 1)
            cv2.putText(hud_box, "Posture: Upright | Control: PPO+APF", (10, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)
            
            frame_combined = np.hstack([frame_3d, side_panel])
            frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
            _, jpeg = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            
            with lock:
                global_frame_jpeg = jpeg.tobytes()
                
            time.sleep(0.04)

def main():
    port = 8088
    fuser_cmd = "fuser -k 8088/tcp > /dev/null 2>&1 || true"
    os.system(fuser_cmd)
    
    sim_thread = threading.Thread(target=run_simulation_loop, daemon=True)
    sim_thread.start()
    
    server = ThreadingHTTPServer(("0.0.0.0", port), MJPEGStreamHandler)
    print(f"==================================================================")
    print(f"🌐 [3D Live Streamer] 3D 实时流媒体服务器已就绪并监听: 0.0.0.0:{port}")
    print(f"   局域网内网访问: http://10.214.131.225:{port}/")
    print(f"   本地转发访问:   http://127.0.0.1:{port}/")
    print(f"==================================================================")
    server.serve_forever()

if __name__ == "__main__":
    main()
