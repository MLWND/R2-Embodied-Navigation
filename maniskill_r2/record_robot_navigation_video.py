#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot Real-Time Navigation Visualizer & Video/GIF Generator
生成机器人真实场景跨房间长距离导航全景动态视频 (MP4)、动图 (GIF) 与多阶段关键帧拼接图
"""

import os
import sys
import glob
import time
import imageio
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from grscenes_importer import extract_grscene_layout
from scene_semantic_graph import build_scene_semantic_graph
from depth_projector import compute_standoff_waypoint
from ppo_nav_agent import PPONavAgent
import torch
import sapien

def euler(q):
    w, x, y, z = q
    roll = np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y-x*z), -1, 1)))
    yaw = np.degrees(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    return roll, pitch, yaw

def diff_drive(v, omega, wheel_r=0.085, track=0.458):
    v_l = (v - omega * track / 2.0) / wheel_r
    v_r = -(v + omega * track / 2.0) / wheel_r
    return v_l, v_r

def simulate_lidar(scene, pos, yaw_deg, robot_shapes, num_rays=48, max_dist=10.0):
    yaw_rad = np.radians(yaw_deg)
    angles = np.linspace(-np.pi, np.pi, num_rays, endpoint=False)
    ranges = np.full(num_rays, max_dist, dtype=np.float32)
    ray_dirs = []
    rel_angles_deg = []
    
    for i, a in enumerate(angles):
        ray_angle = yaw_rad + a
        dx, dy = np.cos(ray_angle), np.sin(ray_angle)
        ray_dirs.append([dx, dy])
        rel_angles_deg.append(np.degrees(a))
        
        hit = scene.physx_system.raycast(
            np.array([pos[0], pos[1], 0.25], dtype=np.float32),
            np.array([dx, dy, 0.0], dtype=np.float32),
            float(max_dist)
        )
        if hit is not None and hit.shape not in robot_shapes:
            ranges[i] = min(hit.distance, max_dist)
            
    return ranges, np.array(ray_dirs), np.array(rel_angles_deg)

def record_navigation_demonstration(
    usd_path="/home/xujinlong/test/GRScenes/scenes/GRScenes-100/home_scenes/scenes/MWAX5JYKTKJZ2AABAAAAAEI8_usd/start_result_navigation.usd",
    instruction="去厨房里的冰箱",
    out_prefix="robot_navigation_demo"
):
    print("=================================================================")
    print(f"🎬 启动机器人真实场景长距离移动动态渲染录制器")
    print(f"   场景 USD: {os.path.basename(os.path.dirname(usd_path))}")
    print(f"   指令: 「{instruction}」")
    print("=================================================================\n")

    # 1. 解析场景与构建拓扑图
    targets_info, walls_info = extract_grscene_layout(usd_path)
    graph = build_scene_semantic_graph(usd_path)
    
    # 2. 规划拓扑航线 (跨房间长距离长达 7.5 米寻路)
    plan = graph.plan_topological_route([3.5, 0.0], instruction)
    waypoints = plan["waypoints"]
    target_obj = plan["target_object"]
    target_room = plan["target_room"]
    start_pos = [3.5, 0.0]
    final_goal = plan["final_goal"]
    theta_goal = plan["theta_goal"]
    
    # 3. 构建无头物理仿真环境
    import re
    URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_swivel.urdf"
    MESH = "/home/xujinlong/test/robot_example1_EN_demo/robot_example1_EN_demo/robot_model/r2_urdf_v01/meshes"
    with open(URDF) as f: urdf_str = f.read()
    urdf_str = urdf_str.replace("package://r2_urdf_v01/meshes/", MESH + "/")
    urdf_no_vis = re.sub(r'<visual>.*?</visual>', '', urdf_str, flags=re.DOTALL)
    with open("/tmp/r2_rec.urdf", "w") as f: f.write(urdf_no_vis)
    
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    scene.set_timestep(0.005)
    scene.add_ground(altitude=0, render=False)
    
    for name, info in targets_info.items():
        b = scene.create_actor_builder()
        sx, sy, sz = info["size"]
        b.add_box_collision(half_size=[sx/2, sy/2, sz/2])
        actor = b.build_static(name=name)
        actor.set_pose(sapien.Pose(p=info["pos"]))
        
    for i, w in enumerate(walls_info):
        b = scene.create_actor_builder()
        sx, sy, sz = w["size"]
        b.add_box_collision(half_size=[sx/2, sy/2, sz/2])
        actor = b.build_static(name=f"wall_{i+1}")
        actor.set_pose(sapien.Pose(p=w["pos"]))
        
    loader = scene.create_urdf_loader()
    loader.fix_root_link = False
    robot = loader.load("/tmp/r2_rec.urdf")
    
    robot_shapes = []
    for l in robot.get_links():
        robot_shapes.extend(l.get_collision_shapes())
        
    wl_j = [j for j in robot.get_joints() if j.get_name() == "wheel_left_joint"][0]
    wr_j = [j for j in robot.get_joints() if j.get_name() == "wheel_right_joint"][0]
    wl_j.set_drive_properties(0.0, 5000.0, 20000.0, "force")
    wr_j.set_drive_properties(0.0, 5000.0, 20000.0, "force")
    
    # 放置机器人
    robot.set_pose(sapien.Pose(p=[start_pos[0], start_pos[1], 0.0], q=[1.0, 0, 0, 0]))
    for _ in range(50): scene.step()
    
    # 加载 PPO
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ppo_agent = PPONavAgent(obs_dim=52, act_dim=2).to(device)
    ppo_model_path = "/home/xujinlong/test/maniskill_r2/models/ppo_r2_nav_69scenes_full.pt"
    if os.path.exists(ppo_model_path):
        ppo_agent.load_state_dict(torch.load(ppo_model_path, map_location=device))
    ppo_agent.eval()
    
    # 地图绘制范围计算
    all_x = [t["pos"][0] for t in targets_info.values()]
    all_y = [t["pos"][1] for t in targets_info.values()]
    margin = 1.8
    map_xlim = (min(all_x) - margin, max(all_x) + margin)
    map_ylim = (min(all_y) - margin, max(all_y) + margin)
    
    frames = []
    trajectory = []
    current_stage_idx = 0
    max_steps = 1500
    prev_pos = None
    prev_yaw = 0.0
    dt = 0.02
    
    # 记录关键帧
    key_snapshots = {}
    
    print(f"🎥 开始执行多阶段导航并逐帧捕获三重视图画面...")
    
    for step in range(max_steps):
        pos = robot.get_pose().p[:2]
        yaw = euler(robot.get_pose().q)[2]
        trajectory.append((pos[0], pos[1]))
        
        # 动力学测速
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
        
        # 激光雷达
        ranges, ray_dirs, rel_angles = simulate_lidar(scene, pos, yaw, robot_shapes)
        min_clearance = float(np.min(ranges))
        
        curr_wp = waypoints[current_stage_idx]
        curr_target = curr_wp["pos"]
        dist_to_curr = np.sqrt((pos[0] - curr_target[0])**2 + (pos[1] - curr_target[1])**2)
        dist_to_final = np.sqrt((pos[0] - final_goal[0])**2 + (pos[1] - final_goal[1])**2)
        
        is_final_stage = (current_stage_idx == len(waypoints) - 1)
        
        # 阶段推进判定
        if dist_to_final < 0.85:
            print(f"  🎯 精准到达终点停靠区 (Step {step})!")
            break
        elif dist_to_curr < (0.45 if not is_final_stage else 0.80):
            if is_final_stage:
                break
            else:
                current_stage_idx += 1
                curr_target = waypoints[current_stage_idx]["pos"]
                dist_to_curr = np.sqrt((pos[0] - curr_target[0])**2 + (pos[1] - curr_target[1])**2)
                
        # 导引计算 (APF + PPO)
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
        
        # 每隔 6 步渲染记录 1 帧画面 (保证 15fps 动画极其流畅细腻)
        if step % 6 == 0:
            fig = plt.figure(figsize=(15, 8.5), dpi=100)
            gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.0])
            
            # 1. 左侧大面板: 全局顶视 2D 户型图 + 实时激光雷达扫描 + 轨迹
            ax_map = fig.add_subplot(gs[:, 0])
            ax_map.set_facecolor("#f8f9fa")
            ax_map.set_title(f"🌍 Global Floorplan & Real-Time LiDAR Trajectory\nScene: {graph.scene_id} | Query: '{instruction}'", fontsize=12, fontweight="bold")
            
            # 绘制房间语义背景
            room_colors = {"kitchen": "#d4efdf", "living_room": "#ebdef0", "master_bedroom": "#d6eaf8", "corridor": "#fcf3cf", "dining_room": "#fae5d3"}
            for r_id, r_info in graph.rooms.items():
                c = room_colors.get(r_info.room_type, "#eaeded")
                rx, ry, rw, rh = r_info.bounds[0], r_info.bounds[1], r_info.bounds[2]-r_info.bounds[0], r_info.bounds[3]-r_info.bounds[1]
                ax_map.add_patch(patches.Rectangle((rx, ry), rw, rh, facecolor=c, alpha=0.35, edgecolor="#7f8c8d", linestyle="--", linewidth=1.5))
                ax_map.text(rx + rw/2, ry + rh/2, f"[{r_info.zh_name}]", fontsize=10, color="#2c3e50", ha="center", va="center", fontweight="bold", alpha=0.6)
                
            # 绘制家具
            for name, info in targets_info.items():
                fx, fy = info["pos"][0], info["pos"][1]
                fsx, fsy = info["size"][0], info["size"][1]
                is_target = (name == target_obj.name)
                fc = "#e74c3c" if is_target else "#95a5a6"
                ec = "red" if is_target else "black"
                lw = 2.5 if is_target else 1.0
                ax_map.add_patch(patches.Rectangle((fx - fsx/2, fy - fsy/2), fsx, fsy, facecolor=fc, edgecolor=ec, alpha=0.85 if is_target else 0.45, linewidth=lw))
                if is_target or "Bed" in name or "Sofa" in name or "Couch" in name or "Fridge" in name or "Table" in name:
                    zh = info.get("zh_name", name.split("_")[0])
                    ax_map.text(fx, fy, zh, fontsize=8, ha="center", va="center", fontweight="bold", color="white" if is_target else "black")
                    
            # 绘制拓扑路径
            wp_xs = [w["pos"][0] for w in waypoints]
            wp_ys = [w["pos"][1] for w in waypoints]
            ax_map.plot(wp_xs, wp_ys, 'b--', linewidth=1.8, alpha=0.7, label="Topological Plan")
            ax_map.scatter(wp_xs, wp_ys, color="blue", s=40, zorder=4)
            for idx, w in enumerate(waypoints):
                ax_map.text(w["pos"][0]+0.15, w["pos"][1]+0.15, f"W{idx+1}", fontsize=8, color="blue", fontweight="bold")
                
            # 绘制历史轨迹
            traj_arr = np.array(trajectory)
            ax_map.plot(traj_arr[:, 0], traj_arr[:, 1], color="#2980b9", linewidth=3.0, label="Robot Path")
            
            # 绘制 48 维激光雷达光束
            yaw_rad = np.radians(yaw)
            for r_val, rel_a in zip(ranges, rel_angles):
                b_rad = yaw_rad + np.radians(rel_a)
                bx = pos[0] + r_val * np.cos(b_rad)
                by = pos[1] + r_val * np.sin(b_rad)
                ray_color = "#e74c3c" if r_val < 0.40 else ("#f39c12" if r_val < 0.80 else "#2ecc71")
                ax_map.plot([pos[0], bx], [pos[1], by], color=ray_color, alpha=0.22, linewidth=1.0)
                
            # 绘制机器人本体与朝向箭头
            ax_map.add_patch(patches.Circle((pos[0], pos[1]), radius=0.22, facecolor="#e67e22", edgecolor="black", linewidth=2.0, zorder=5))
            head_dx = 0.45 * np.cos(yaw_rad)
            head_dy = 0.45 * np.sin(yaw_rad)
            ax_map.arrow(pos[0], pos[1], head_dx, head_dy, head_width=0.18, head_length=0.15, fc="black", ec="black", zorder=6)
            
            ax_map.set_xlim(map_xlim)
            ax_map.set_ylim(map_ylim)
            ax_map.set_aspect("equal")
            ax_map.legend(loc="upper right", fontsize=8)
            ax_map.grid(True, linestyle=":", alpha=0.5)
            
            # 2. 右上面板: 车载第一视角相机 (First-Person View) 动态视口
            ax_fpv = fig.add_subplot(gs[0, 1])
            ax_fpv.set_facecolor("#2c3e50")
            ax_fpv.set_title(f"📷 Onboard First-Person View (FPV)\nRobot Yaw: {yaw:+.1f}° | Clearance: {min_clearance:.2f}m", fontsize=11, fontweight="bold")
            
            # 合成第一人称视角的雷达极坐标投影雷达屏
            fwd_ranges = ranges[(rel_angles >= -60) & (rel_angles <= 60)]
            fwd_angles = rel_angles[(rel_angles >= -60) & (rel_angles <= 60)]
            fwd_rads = np.radians(fwd_angles)
            fwd_xs = fwd_ranges * np.sin(fwd_rads)
            fwd_ys = fwd_ranges * np.cos(fwd_rads)
            
            ax_fpv.fill_between([-4, 4], [0, 0], [6, 6], color="#34495e", alpha=0.6)
            ax_fpv.plot([0, -3.5], [0, 5], 'g--', alpha=0.5)
            ax_fpv.plot([0, 3.5], [0, 5], 'g--', alpha=0.5)
            ax_fpv.scatter(fwd_xs, fwd_ys, c=fwd_ranges, cmap="RdYlGn", s=70, edgecolors="white", linewidth=1.0, zorder=3)
            ax_fpv.plot([0], [0], 'yo', markersize=12, label="Robot Camera")
            
            # 若前方有目标家具，投影显示目标提示
            tgt_dx = final_goal[0] - pos[0]
            tgt_dy = final_goal[1] - pos[1]
            tgt_dist = np.sqrt(tgt_dx**2 + tgt_dy**2)
            tgt_rel_deg = (np.degrees(np.arctan2(tgt_dy, tgt_dx)) - yaw + 180) % 360 - 180
            if abs(tgt_rel_deg) < 60:
                rad_t = np.radians(tgt_rel_deg)
                ax_fpv.scatter([tgt_dist * np.sin(rad_t)], [tgt_dist * np.cos(rad_t)], color="red", s=180, marker="*", edgecolors="white", label="TARGET")
                ax_fpv.text(tgt_dist * np.sin(rad_t), tgt_dist * np.cos(rad_t)+0.3, f"TARGET\n[{target_obj.zh_name}]", color="yellow", fontsize=9, fontweight="bold", ha="center")
                
            ax_fpv.set_xlim(-3.5, 3.5)
            ax_fpv.set_ylim(-0.5, 6.0)
            ax_fpv.grid(True, linestyle="--", alpha=0.3)
            ax_fpv.legend(loc="upper right", fontsize=8)
            
            # 3. 右下面板: 实时遥测仪表盘与动力学 HUD
            ax_hud = fig.add_subplot(gs[1, 1])
            ax_hud.set_facecolor("#1a1a24")
            ax_hud.set_title("📊 Dynamic Telemetry & State HUD", fontsize=11, fontweight="bold", color="white")
            ax_hud.axis("off")
            
            hud_text = [
                f"• Instruction: 「{instruction}」",
                f"• Target Room: [{target_room.zh_name}]  |  Object: [{target_obj.zh_name}]",
                f"• Current Stage: {current_stage_idx+1} / {len(waypoints)} -> {waypoints[current_stage_idx]['desc']}",
                f"• Linear Velocity (v): {act_v:.2f} m/s  |  Command: {v_cmd:.2f} m/s",
                f"• Angular Velocity (ω): {act_w:+.2f} rad/s  |  Command: {omega_cmd:+.2f} rad/s",
                f"• Heading Error: {heading_err:+.1f}°  |  Distance to Sub-goal: {dist_to_curr:.2f} m",
                f"• Distance to Target Docking: {dist_to_final:.2f} m",
                f"• Min LiDAR Clearance: {min_clearance:.2f} m  ({'✅ SAFE' if min_clearance >= 0.22 else '⚠️ CAUTION'})",
                f"• Navigation State: PPO+APF Autonomous Cruising"
            ]
            for l_idx, line in enumerate(hud_text):
                color = "#2ecc71" if "SAFE" in line else ("#f1c40f" if "Stage" in line or "Instruction" in line else "white")
                ax_hud.text(0.05, 0.90 - l_idx * 0.095, line, fontsize=10.5, color=color, fontweight="bold", transform=ax_hud.transAxes)
                
            fig.canvas.draw()
            frame_rgba = np.asarray(fig.canvas.buffer_rgba())
            frame_rgb = frame_rgba[:, :, :3]
            frames.append(frame_rgb)
            
            # 记录关键进度截图 (0%, 25%, 50%, 75%, 100%)
            if len(key_snapshots) == 0: key_snapshots["t0"] = frame_rgb
            elif len(key_snapshots) == 1 and dist_to_final < 2.2: key_snapshots["t25"] = frame_rgb
            elif len(key_snapshots) == 2 and dist_to_final < 1.6: key_snapshots["t50"] = frame_rgb
            elif len(key_snapshots) == 3 and dist_to_final < 1.0: key_snapshots["t75"] = frame_rgb
            
            plt.close(fig)

    if "t100" not in key_snapshots and len(frames) > 0:
        key_snapshots["t100"] = frames[-1]
        
    print(f"\n✅ 录制完成！共捕获 {len(frames)} 帧高清画面。正在导出多媒体产物...")
    
    out_dir = "/home/xujinlong/test/output"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. 导出 MP4 高清视频
    mp4_path = os.path.join(out_dir, f"{out_prefix}.mp4")
    imageio.mimsave(mp4_path, frames, fps=12, quality=8)
    print(f"🎬 MP4 高清视频已保存至: {mp4_path}")
    
    # 2. 导出 GIF 动画 (降采样生成轻量级 GIF，适合嵌入文档)
    gif_path = os.path.join(out_dir, f"{out_prefix}.gif")
    gif_frames = [cv2.resize(f, (750, 425)) for f in frames[::2]]
    imageio.mimsave(gif_path, gif_frames, fps=8, loop=0)
    print(f"🎞️ 动态 GIF 动图已保存至: {gif_path}")
    
    # 3. 导出多阶段关键帧拼接图 (Montage)
    montage_keys = ["t0", "t25", "t50", "t75", "t100"]
    valid_keys = [k for k in montage_keys if k in key_snapshots]
    if len(valid_keys) >= 3:
        fig_m, axes_m = plt.subplots(1, len(valid_keys), figsize=(4.5 * len(valid_keys), 4))
        for idx, k in enumerate(valid_keys):
            axes_m[idx].imshow(key_snapshots[k])
            axes_m[idx].set_title(f"Progression: {idx * 25}%", fontsize=11, fontweight="bold")
            axes_m[idx].axis("off")
        plt.tight_layout()
        montage_path = os.path.join(out_dir, "robot_nav_montage.png")
        plt.savefig(montage_path, dpi=150)
        plt.close()
        print(f"🖼️ 多阶段关键帧拼接图已保存至: {montage_path}")
        
    # 复制产物到当前会话的 artifacts 目录，以便直接在界面嵌入
    brain_dir = "/home/xujinlong/.gemini/antigravity-cli/brain/dad4514f-eaa8-4f43-a94c-b474d94ccab9"
    if os.path.exists(brain_dir):
        import shutil
        shutil.copy(gif_path, os.path.join(brain_dir, "robot_navigation_demo.gif"))
        if os.path.exists(os.path.join(out_dir, "robot_nav_montage.png")):
            shutil.copy(os.path.join(out_dir, "robot_nav_montage.png"), os.path.join(brain_dir, "robot_nav_montage.png"))
        print(f"📦 产物已同步至 Artifacts 目录，支持富文本直接渲染！")
        
    return mp4_path, gif_path

if __name__ == "__main__":
    record_navigation_demonstration()
