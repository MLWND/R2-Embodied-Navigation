#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Photorealistic Robot Appearance & Dynamic Navigation Demonstration
锁定人形上身与机械双臂关节刚度，呈现优雅挺拔的 3D 机器人外观与真实 3D 漫游姿态
"""

import os
import sys
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

def euler(q):
    w, x, y, z = q
    roll = np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y-x*z), -1, 1)))
    yaw = np.degrees(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    return roll, pitch, yaw

def compute_lookat_quat(cam_pos, target_pos):
    forward = np.array(target_pos) - np.array(cam_pos)
    dist = np.linalg.norm(forward)
    forward /= dist
    up = np.array([0.0, 0.0, 1.0])
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)
    real_up = np.cross(forward, left)
    R = np.eye(3)
    R[:, 0] = forward
    R[:, 1] = left
    R[:, 2] = real_up
    quat = spt.Rotation.from_matrix(R).as_quat()
    return [quat[3], quat[0], quat[1], quat[2]]

def diff_drive(v, omega, wheel_r=0.085, track=0.458):
    v_l = (v - omega * track / 2.0) / wheel_r
    v_r = -(v + omega * track / 2.0) / wheel_r
    return v_l, v_r

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
    print("🎨 启动 R2 人形机器人 3D 姿态锁定与真实 3D 漫游导航高保真渲染")
    print("=================================================================\n")

    usd_path = "/home/xujinlong/test/GRScenes/scenes/GRScenes-100/home_scenes/scenes/MWAX5JYKTKJZ2AABAAAAAEI8_usd/start_result_navigation.usd"
    targets_info, walls_info = extract_grscene_layout(usd_path)
    graph = build_scene_semantic_graph(usd_path)
    
    # 1. 创建 3D 场景与高品质摄影棚级布光
    scene = sapien.Scene()
    scene.set_timestep(0.005)
    scene.set_ambient_light([0.42, 0.42, 0.46])
    scene.add_directional_light([1.8, -1.2, -2.2], [1.15, 1.1, 1.05])
    scene.add_directional_light([-1.5, 1.5, -1.5], [0.55, 0.6, 0.65])
    scene.add_point_light([0.0, 0.0, 3.2], [1.1, 1.1, 1.1])
    scene.add_point_light([3.5, 0.0, 3.2], [1.0, 1.0, 1.0])
    scene.add_ground(altitude=0, render=True)
    
    # 材质定义
    mat_wall = sapien.render.RenderMaterial(base_color=[0.92, 0.92, 0.94, 1.0], roughness=0.65)
    mat_sofa = sapien.render.RenderMaterial(base_color=[0.20, 0.35, 0.55, 1.0], roughness=0.6)
    mat_bed = sapien.render.RenderMaterial(base_color=[0.30, 0.50, 0.40, 1.0], roughness=0.5)
    mat_table = sapien.render.RenderMaterial(base_color=[0.70, 0.48, 0.28, 1.0], roughness=0.4)
    mat_fridge = sapien.render.RenderMaterial(base_color=[0.85, 0.88, 0.90, 1.0], metallic=0.75, roughness=0.25)
    mat_target = sapien.render.RenderMaterial(base_color=[0.95, 0.22, 0.22, 1.0], metallic=0.5, roughness=0.3)
    mat_other = sapien.render.RenderMaterial(base_color=[0.60, 0.60, 0.65, 1.0], roughness=0.6)
    
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
        
    # 加载 3D 机器人模型
    loader = scene.create_urdf_loader()
    loader.fix_root_link = False
    robot = loader.load("/tmp/r2_3d_vis.urdf")
    
    robot_shapes = []
    for l in robot.get_links():
        robot_shapes.extend(l.get_collision_shapes())
        
    # 锁定上身、腰部与双臂所有非移动关节，保持自然优美挺拔姿态
    for j in robot.get_active_joints():
        jname = j.get_name()
        if jname in ["wheel_left_joint", "wheel_right_joint"]:
            j.set_drive_properties(0.0, 5000.0, 20000.0, "force")
        elif "caster" in jname:
            j.set_drive_properties(0.0, 10.0, 50.0, "force")
        else:
            # 锁定腰部、升降柱、头部与机械双臂
            j.set_drive_properties(10000.0, 500.0, 50000.0, "force")
            j.set_drive_target(0.0)
            
    wl_j = [j for j in robot.get_joints() if j.get_name() == "wheel_left_joint"][0]
    wr_j = [j for j in robot.get_joints() if j.get_name() == "wheel_right_joint"][0]
    
    # 放置机器人并微步让轮组贴地
    robot.set_pose(sapien.Pose(p=[start_pos[0], start_pos[1], 0.0], q=[1.0, 0, 0, 0]))
    for _ in range(20): scene.step()
    
    out_dir = "/home/xujinlong/test/output"
    os.makedirs(out_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # 一、生成 4 视角 3D 外观超清写真展图 (r2_robot_3d_appearance_hd.png)
    # -------------------------------------------------------------
    print("📸 正在渲染 R2 机器人 3D 超清全景外观写真展...")
    cam_gallery = scene.add_camera(name="cam_gal", width=960, height=540, fovy=np.radians(45), near=0.1, far=50.0)
    
    gallery_views = [
        ("1. 3D Full-Body Hero View (正面立体姿态)", [start_pos[0] + 1.8, start_pos[1] - 1.8, 1.3], [start_pos[0], start_pos[1], 0.75]),
        ("2. Upper Body Humanoid Torso (胸腔、头部与机械双臂)", [start_pos[0] + 1.1, start_pos[1] - 1.1, 1.25], [start_pos[0], start_pos[1], 1.1]),
        ("3. Chassis & Wheels Close-Up (底盘双轮与万向轮)", [start_pos[0] + 1.1, start_pos[1] - 0.9, 0.45], [start_pos[0], start_pos[1], 0.18]),
        ("4. 3D Scene Overview (3D 室内场景与家具布局)", [start_pos[0] + 3.2, start_pos[1] - 2.8, 2.5], [start_pos[0] - 1.0, start_pos[1] + 1.0, 0.5])
    ]
    
    g_imgs = []
    for title, cpos, tpos in gallery_views:
        cam_gallery.set_pose(sapien.Pose(p=cpos, q=compute_lookat_quat(cpos, tpos)))
        scene.update_render()
        cam_gallery.take_picture()
        rgba = cam_gallery.get_picture('Color')
        rgb = (np.clip(rgba[:, :, :3], 0, 1) * 255).astype(np.uint8)
        
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        draw.rectangle([(0, 0), (pil.width, 35)], fill=(15, 23, 42, 210))
        draw.text((15, 8), f"R2 3D Robot: {title}", fill=(56, 189, 248, 255))
        g_imgs.append(np.array(pil))
        
    top_row = np.hstack([g_imgs[0], g_imgs[1]])
    bot_row = np.hstack([g_imgs[2], g_imgs[3]])
    gallery_hd = np.vstack([top_row, bot_row])
    
    hd_gallery_path = os.path.join(out_dir, "r2_robot_3d_appearance_hd.png")
    Image.fromarray(gallery_hd).save(hd_gallery_path)
    print(f"✅ 3D 超清外观写真展图已保存至: {hd_gallery_path}")
    
    # -------------------------------------------------------------
    # 二、生成 3D 动态跟随漫游视频与 GIF
    # -------------------------------------------------------------
    print("\n🎥 正在执行 3D 动态跟随漫游导航录制...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ppo_agent = PPONavAgent(obs_dim=52, act_dim=2).to(device)
    ppo_model_path = "/home/xujinlong/test/maniskill_r2/models/ppo_r2_nav_69scenes_full.pt"
    if os.path.exists(ppo_model_path):
        ppo_agent.load_state_dict(torch.load(ppo_model_path, map_location=device))
    ppo_agent.eval()
    
    cam_chase = scene.add_camera(name="chase_cam", width=960, height=540, fovy=np.radians(52), near=0.1, far=50.0)
    
    frames_3d = []
    trajectory = []
    current_stage_idx = 0
    max_steps = 1500
    prev_pos = None
    prev_yaw = 0.0
    dt = 0.02
    
    for step in range(max_steps):
        pos = robot.get_pose().p[:2]
        yaw = euler(robot.get_pose().q)[2]
        trajectory.append((pos[0], pos[1]))
        
        if prev_pos is None:
            act_v, act_w = 0.0, 0.0
        else:
            dx = pos[0] - prev_pos[0]
            dy = pos[1] - prev_pos[1]
            act_v = np.sqrt(dx**2 + dy**2) / dt
            dyaw = (yaw - prev_yaw + 180) % 360 - 180
            act_w = np.radians(dyaw) / dt
        prev_pos = pos
        prev_yaw = yaw
        
        ranges, rel_angles = simulate_lidar(scene, pos, yaw, robot_shapes)
        min_clearance = float(np.min(ranges))
        
        curr_wp = waypoints[current_stage_idx]
        curr_target = curr_wp["pos"]
        dist_to_curr = np.sqrt((pos[0] - curr_target[0])**2 + (pos[1] - curr_target[1])**2)
        dist_to_final = np.sqrt((pos[0] - final_goal[0])**2 + (pos[1] - final_goal[1])**2)
        
        is_final_stage = (current_stage_idx == len(waypoints) - 1)
        if dist_to_final < 0.80:
            print(f"  🎯 成功到达目标家具前 (Step {step})!")
            break
        elif dist_to_curr < (0.45 if not is_final_stage else 0.80):
            if is_final_stage:
                break
            else:
                current_stage_idx += 1
                curr_target = waypoints[current_stage_idx]["pos"]
                dist_to_curr = np.sqrt((pos[0] - curr_target[0])**2 + (pos[1] - curr_target[1])**2)
                
        dx = curr_target[0] - pos[0]
        dy = curr_target[1] - pos[1]
        f_att = np.array([dx / max(dist_to_curr, 0.05), dy / max(dist_to_curr, 0.05)])
        
        f_rep = np.zeros(2, dtype=np.float32)
        for r_val, rel_a in zip(ranges, rel_angles):
            if 0.05 < r_val < 0.40:
                beam_angle_rad = np.radians(yaw + rel_a)
                beam_dir = np.array([np.cos(beam_angle_rad), np.sin(beam_angle_rad)])
                f_rep -= 0.12 * (1.0/r_val - 1.0/0.40) / (r_val**2) * beam_dir
                
        f_total = f_att + f_rep
        target_angle_deg = np.degrees(np.arctan2(f_total[1], f_total[0]))
        heading_err = (target_angle_deg - yaw + 180) % 360 - 180
        
        obs = np.zeros(52, dtype=np.float32)
        obs[:48] = np.clip(ranges / 10.0, 0.0, 1.0)
        obs[48] = dist_to_curr / 10.0
        obs[49] = np.radians(heading_err) / np.pi
        obs[50] = np.clip(act_v / 0.55, 0.0, 1.0)
        obs[51] = np.clip(act_w / 1.5, -1.0, 1.0)
        
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        ppo_action = ppo_agent.act_deterministic(obs_t).squeeze(0).cpu().numpy()
        
        ppo_v = float(np.clip((ppo_action[0] + 1.0) * 0.5 * 0.55, 0.0, 0.55))
        ppo_w = float(np.clip(ppo_action[1] * 1.5, -1.5, 1.5))
        exp_w = float(np.clip(2.5 * np.radians(heading_err), -1.5, 1.5))
        exp_align = max(0.0, float(np.cos(np.radians(heading_err))))
        exp_v = float(0.55 * (exp_align ** 3) * np.clip(dist_to_curr / 0.5, 0.40, 1.0))
        
        if abs(heading_err) > 15.0:
            v_cmd = 0.0
            omega_cmd = exp_w
        else:
            alpha = 0.5 if abs(heading_err) < 10.0 else 0.1
            v_cmd = alpha * ppo_v + (1 - alpha) * exp_v
            if exp_align > 0.5:
                v_cmd = max(v_cmd, 0.35 * (exp_align ** 2))
            omega_cmd = alpha * ppo_w + (1 - alpha) * exp_w
            
        v_l, v_r = diff_drive(v_cmd, omega_cmd)
        wl_j.set_drive_velocity_target(v_l)
        wr_j.set_drive_velocity_target(v_r)
        for _ in range(4): scene.step()
        
        if step % 7 == 0:
            # 3D 侧后方追随漫游相机：对准机器人头部与胸腔 (Z=0.75m)
            yaw_rad = np.radians(yaw)
            chase_dist = 2.3
            chase_angle = yaw_rad - np.radians(45) # 位于机器人侧前方透视
            cam_x = pos[0] + chase_dist * np.cos(chase_angle)
            cam_y = pos[1] + chase_dist * np.sin(chase_angle)
            cam_z = 1.35
            
            target_pt = [pos[0], pos[1], 0.70]
            cam_quat = compute_lookat_quat([cam_x, cam_y, cam_z], target_pt)
            cam_chase.set_pose(sapien.Pose(p=[cam_x, cam_y, cam_z], q=cam_quat))
            
            scene.update_render()
            cam_chase.take_picture()
            rgba = cam_chase.get_picture('Color')
            frame_rgb = (np.clip(rgba[:, :, :3], 0, 1) * 255).astype(np.uint8)
            
            img_pil = Image.fromarray(frame_rgb)
            overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            draw.rectangle([(0, 0), (img_pil.width, 42)], fill=(15, 23, 42, 200))
            draw.text((15, 11), f"🤖 R2 3D Robot Live Navigation | Target: [{target_obj.zh_name}] ({target_room.zh_name})", fill=(56, 189, 248, 255))
            
            draw.rectangle([(0, img_pil.height - 45), (img_pil.width, img_pil.height)], fill=(15, 23, 42, 210))
            status_text = f"Stage: {current_stage_idx+1}/{len(waypoints)} -> {waypoints[current_stage_idx]['desc']} | v={act_v:.2f}m/s | w={act_w:+.2f}rad/s | dist={dist_to_final:.2f}m | clearance={min_clearance:.2f}m"
            draw.text((15, img_pil.height - 30), status_text, fill=(241, 245, 249, 255))
            
            img_final = Image.alpha_composite(img_pil.convert('RGBA'), overlay).convert('RGB')
            frames_3d.append(np.array(img_final))
            
    # 导出 3D MP4 视频与 GIF 动画
    mp4_3d_path = os.path.join(out_dir, "r2_robot_3d_navigation_demo.mp4")
    imageio.mimsave(mp4_3d_path, frames_3d, fps=12, quality=8)
    print(f"🎬 3D MP4 漫游视频已保存至: {mp4_3d_path}")
    
    gif_3d_path = os.path.join(out_dir, "r2_robot_3d_navigation_demo.gif")
    gif_3d_frames = [cv2.resize(f, (640, 360)) for f in frames_3d[::2]]
    imageio.mimsave(gif_3d_path, gif_3d_frames, fps=8, loop=0)
    print(f"🎞️ 3D 动态 GIF 动图已保存至: {gif_3d_path}")
    
    brain_dir = "/home/xujinlong/.gemini/antigravity-cli/brain/dad4514f-eaa8-4f43-a94c-b474d94ccab9"
    if os.path.exists(brain_dir):
        import shutil
        shutil.copy(hd_gallery_path, os.path.join(brain_dir, "r2_robot_3d_appearance_hd.png"))
        shutil.copy(gif_3d_path, os.path.join(brain_dir, "r2_robot_3d_navigation_demo.gif"))
        print("📦 3D 成果已同步至 Artifacts 目录！")

if __name__ == "__main__":
    main()
