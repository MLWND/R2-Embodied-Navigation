#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Fidelity 3D Robot Navigation Simulator & Video Generator
使用 SAPIEN 3D Vulkan 引擎 + 平面差速动力学刚体约束，生成 100% 挺拔稳定、无倾倒、无抖动的 3D 机器人自主导航演示视频与动画
"""

import os
import sys
import time
import imageio
import cv2
import numpy as np
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

def main():
    print("=================================================================")
    print("🚀 启动 3D 高保真实体仿真：平稳差速运动 + 3D 动态追随视角渲染")
    print("=================================================================\n")

    usd_path = "/home/xujinlong/test/GRScenes/scenes/GRScenes-100/home_scenes/scenes/MWAX5JYKTKJZ2AABAAAAAEI8_usd/start_result_navigation.usd"
    targets_info, walls_info = extract_grscene_layout(usd_path)
    graph = build_scene_semantic_graph(usd_path)
    
    # 1. 创建 3D 物理与高画质光影渲染场景
    scene = sapien.Scene()
    scene.set_timestep(0.01)
    scene.set_ambient_light([0.48, 0.48, 0.52])
    scene.add_directional_light([2.0, -1.5, -2.5], [1.2, 1.15, 1.1])
    scene.add_directional_light([-2.0, 1.5, -2.0], [0.6, 0.65, 0.7])
    scene.add_point_light([0.0, 0.0, 3.5], [1.2, 1.2, 1.2])
    scene.add_point_light([3.5, 0.0, 3.5], [1.0, 1.0, 1.0])
    scene.add_ground(altitude=0, render=True)
    
    # 材质定义
    mat_wall = sapien.render.RenderMaterial(base_color=[0.88, 0.90, 0.92, 1.0], roughness=0.65)
    mat_sofa = sapien.render.RenderMaterial(base_color=[0.18, 0.35, 0.58, 1.0], roughness=0.55)
    mat_bed = sapien.render.RenderMaterial(base_color=[0.28, 0.52, 0.42, 1.0], roughness=0.5)
    mat_table = sapien.render.RenderMaterial(base_color=[0.72, 0.48, 0.28, 1.0], roughness=0.4)
    mat_fridge = sapien.render.RenderMaterial(base_color=[0.85, 0.88, 0.92, 1.0], metallic=0.75, roughness=0.25)
    mat_target = sapien.render.RenderMaterial(base_color=[0.95, 0.20, 0.20, 1.0], metallic=0.5, roughness=0.3)
    mat_other = sapien.render.RenderMaterial(base_color=[0.58, 0.60, 0.65, 1.0], roughness=0.6)
    
    instruction = "去厨房里的冰箱"
    start_pos = [3.5, 0.0]
    plan = graph.plan_topological_route(start_pos, instruction)
    waypoints = plan["waypoints"]
    target_obj = plan["target_object"]
    target_room = plan["target_room"]
    final_goal = plan["final_goal"]
    
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
        
    # 加载 3D 机器人模型 (使用 Kinematic Root 锁定地面与直立姿态)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load("/tmp/r2_3d_vis.urdf")
    
    robot_shapes = []
    for l in robot.get_links():
        robot_shapes.extend(l.get_collision_shapes())
        
    # 3D 追随相机
    cam_3d = scene.add_camera(name="chase_3d", width=720, height=540, fovy=np.radians(52), near=0.1, far=50.0)
    
    # 策略与控制器
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ppo_agent = PPONavAgent(obs_dim=52, act_dim=2).to(device)
    ppo_model_path = "/home/xujinlong/test/maniskill_r2/models/ppo_r2_nav_69scenes_full.pt"
    if os.path.exists(ppo_model_path):
        ppo_agent.load_state_dict(torch.load(ppo_model_path, map_location=device))
    ppo_agent.eval()
    
    # 初始状态
    robot_pos = np.array([start_pos[0], start_pos[1]], dtype=np.float32)
    robot_yaw = 180.0 # 面向走廊方向
    
    robot.set_pose(sapien.Pose(p=[robot_pos[0], robot_pos[1], 0.0], q=yaw_to_quat(robot_yaw)))
    
    current_stage_idx = 0
    trajectory = [(robot_pos[0], robot_pos[1])]
    frames_combined = []
    
    dt = 0.04
    max_steps = 450
    act_v, act_w = 0.0, 0.0
    
    print("🎬 正在执行 3D 机器人平稳移动与多重视图合成录制...")
    
    for step in range(max_steps):
        ranges, rel_angles = simulate_lidar(scene, robot_pos, robot_yaw, robot_shapes)
        min_clearance = float(np.min(ranges))
        
        curr_wp = waypoints[current_stage_idx]
        curr_target = curr_wp["pos"]
        dist_to_curr = np.sqrt((robot_pos[0] - curr_target[0])**2 + (robot_pos[1] - curr_target[1])**2)
        dist_to_final = np.sqrt((robot_pos[0] - final_goal[0])**2 + (robot_pos[1] - final_goal[1])**2)
        
        is_final_stage = (current_stage_idx == len(waypoints) - 1)
        if dist_to_final < 0.75:
            print(f"  🎯 成功到达目标家具前停靠 (Step {step})!")
            break
        elif dist_to_curr < (0.45 if not is_final_stage else 0.75):
            if is_final_stage:
                break
            else:
                current_stage_idx += 1
                curr_target = waypoints[current_stage_idx]["pos"]
                dist_to_curr = np.sqrt((robot_pos[0] - curr_target[0])**2 + (robot_pos[1] - curr_target[1])**2)
                
        # 目标引力
        dx = curr_target[0] - robot_pos[0]
        dy = curr_target[1] - robot_pos[1]
        f_att = np.array([dx / max(dist_to_curr, 0.05), dy / max(dist_to_curr, 0.05)])
        
        # 48D 激光雷达 APF 斥力
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
            
        # 狭窄门洞 APF 前进保护
        if min_clearance < 0.35 and abs(heading_err) < 20.0:
            v_cmd = max(v_cmd, 0.28)
            
        # 平稳差速动力学前向积分 (0 抖动，0 倾翻)
        act_v = v_cmd
        act_w = omega_cmd
        
        robot_yaw = (robot_yaw + np.degrees(act_w * dt) + 180) % 360 - 180
        yaw_rad = np.radians(robot_yaw)
        robot_pos[0] += act_v * np.cos(yaw_rad) * dt
        robot_pos[1] += act_v * np.sin(yaw_rad) * dt
        trajectory.append((robot_pos[0], robot_pos[1]))
        
        # 更新 SAPIEN 中机器人的真实 3D 姿态
        robot.set_pose(sapien.Pose(p=[robot_pos[0], robot_pos[1], 0.0], q=yaw_to_quat(robot_yaw)))
        
        # 3D 追随相机：位于机器人斜后方 3.2m，视线锁定机器人胸腔与头部中心
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
        
        # 构造右侧 2D 户型小地图 + 48D LiDAR 动态雷达扫描图 + HUD
        side_panel = np.full((540, 480, 3), 24, dtype=np.uint8)
        
        # 顶部 2D 户型小地图 (260px)
        map_box = side_panel[10:270, 10:470]
        map_box[:] = 16
        scale = 32.0
        cx, cy = 230, 130
        
        # 绘制墙体
        for w in walls_info:
            wx, wy = int(cx + w["pos"][0] * scale), int(cy - w["pos"][1] * scale)
            wsx, wsy = int(w["size"][0] * scale / 2), int(w["size"][1] * scale / 2)
            cv2.rectangle(map_box, (wx - wsx, wy - wsy), (wx + wsx, wy + wsy), (100, 100, 110), -1)
            
        # 绘制家具
        for name, info in targets_info.items():
            fx, fy = int(cx + info["pos"][0] * scale), int(cy - info["pos"][1] * scale)
            fsx, fsy = int(info["size"][0] * scale / 2), int(info["size"][1] * scale / 2)
            col = (0, 0, 220) if name == target_obj.name else (140, 100, 60)
            cv2.rectangle(map_box, (fx - fsx, fy - fsy), (fx + fsx, fy + fsy), col, -1)
            
        # 绘制 48D LiDAR 动态扫描光线
        rx, ry = int(cx + robot_pos[0] * scale), int(cy - robot_pos[1] * scale)
        for r_val, rel_a in zip(ranges[::2], rel_angles[::2]):
            beam_rad = np.radians(robot_yaw + rel_a)
            bx = int(rx + r_val * scale * np.cos(beam_rad))
            by = int(ry - r_val * scale * np.sin(beam_rad))
            cv2.line(map_box, (rx, ry), (bx, by), (40, 180, 40), 1)
            
        # 绘制历史轨迹
        if len(trajectory) > 1:
            pts = np.array([[int(cx + p[0]*scale), int(cy - p[1]*scale)] for p in trajectory], np.int32)
            cv2.polylines(map_box, [pts], False, (0, 230, 255), 2)
            
        # 绘制机器人本体与朝向
        cv2.circle(map_box, (rx, ry), 7, (0, 255, 255), -1)
        hx = int(rx + 14 * np.cos(yaw_rad))
        hy = int(ry - 14 * np.sin(yaw_rad))
        cv2.arrowedLine(map_box, (rx, ry), (hx, hy), (0, 0, 255), 2, tipLength=0.4)
        cv2.putText(map_box, "2D Semantic Map & 48D LiDAR", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # 底部 HUD 遥测数据 (250px)
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
        
        # 左右拼接 (720 + 480 = 1200x540)
        frame_combined = np.hstack([frame_3d, side_panel])
        frames_combined.append(frame_combined)
        
    out_dir = "/home/xujinlong/test/output"
    os.makedirs(out_dir, exist_ok=True)
    
    # 导出 MP4 视频
    mp4_path = os.path.join(out_dir, "r2_robot_3d_navigation_demo.mp4")
    imageio.mimsave(mp4_path, frames_combined, fps=15, quality=8)
    print(f"🎬 3D MP4 视频已保存至: {mp4_path}")
    
    # 导出轻量级 GIF 动画
    gif_path = os.path.join(out_dir, "r2_robot_3d_navigation_demo.gif")
    gif_frames = [cv2.resize(f, (600, 270)) for f in frames_combined[::3]]
    imageio.mimsave(gif_path, gif_frames, fps=10, loop=0)
    print(f"🎞️ 3D 动态 GIF 动图已保存至: {gif_path}")
    
    # 同步至 brain artifacts
    brain_dir = "/home/xujinlong/.gemini/antigravity-cli/brain/dad4514f-eaa8-4f43-a94c-b474d94ccab9"
    if os.path.exists(brain_dir):
        import shutil
        shutil.copy(gif_path, os.path.join(brain_dir, "r2_robot_3d_navigation_demo.gif"))
        print("📦 3D 成果已同步至 Artifacts！")

if __name__ == "__main__":
    main()
