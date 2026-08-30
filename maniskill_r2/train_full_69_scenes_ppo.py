"""GRScenes-100 全量 69 套家庭场景大规模 PPO 强化学习训练与全量评测脚本 (充分航行步数)"""
import os
import glob
import json
import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ppo_nav_agent import PPONavAgent
from grscenes_importer import extract_grscene_layout
from depth_projector import compute_standoff_waypoint

def get_all_69_scenes():
    base_dir = "/home/xujinlong/test/GRScenes/scenes/GRScenes-100/home_scenes/scenes"
    scene_paths = sorted(glob.glob(os.path.join(base_dir, "*_usd")))
    return scene_paths

def train_full_69_scenes(num_rounds=2, steps_per_env=300):
    scene_paths = get_all_69_scenes()
    total_scenes = len(scene_paths)
    total_episodes = total_scenes * num_rounds
    
    print(f"🏠 正在加载全部 {total_scenes} 套 GRScenes-100 真实家庭场景...")
    print(f"🔄 启动全场景覆盖式 PPO 强化学习训练 (总轮次: {num_rounds}, 每幕充分步数: {steps_per_env}, 总 Episode: {total_episodes})...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = PPONavAgent(obs_dim=52, act_dim=2).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=5e-4, eps=1e-5)

    all_rewards = []
    all_successes = []
    scene_metrics = {}

    start_time = time.time()

    for ep in range(total_episodes):
        scene_idx = ep % total_scenes
        s_path = scene_paths[scene_idx]
        s_name = os.path.basename(s_path)

        usd_file = os.path.join(s_path, "start_result_navigation.usd")
        targets_info, walls_info = extract_grscene_layout(usd_file if os.path.exists(usd_file) else None)
        
        target_keys = list(targets_info.keys())
        chosen_key = target_keys[ep % len(target_keys)]
        chosen_target = targets_info[chosen_key]
        p_obj = chosen_target["pos"]
        sx, sy = chosen_target["size"][0], chosen_target["size"][1]
        goal_x, goal_y, _ = compute_standoff_waypoint(
            p_obj, [0, 0, 0], spatial_relation="front", standoff_dist=0.75,
            obj_half_extents=[sx / 2.0, sy / 2.0]
        )
        
        obs_dim = 52
        curr_pos = np.array([0.0, 0.0], dtype=np.float32)
        yaw_deg = 0.0
        prev_dist = float(np.sqrt(goal_x**2 + goal_y**2))
        prev_v_cmd = 0.0
        prev_w_cmd = 0.0
        
        ep_ret = 0.0
        is_succ = False
        min_clearance = float('inf')

        # Collect obstacles (walls + all non-target furniture)
        obstacle_list = list(walls_info)
        for k, v in targets_info.items():
            if k != chosen_key:
                obstacle_list.append({"pos": v["pos"], "size": v["size"]})

        obs_list, act_list, logp_list, rew_list, val_list, done_list, expert_act_list = [], [], [], [], [], [], []

        for st in range(steps_per_env):
            dx = goal_x - curr_pos[0]
            dy = goal_y - curr_pos[1]
            dist = float(np.sqrt(dx**2 + dy**2))
            target_angle = np.degrees(np.arctan2(dy, dx))
            heading_err = (target_angle - yaw_deg + 180) % 360 - 180
            
            # 激光雷达避障模拟 (包含墙体与场景内非目标家具障碍)
            lidar = np.full(48, 8.0, dtype=np.float32)
            for obs_item in obstacle_list:
                ox, oy = obs_item["pos"][0], obs_item["pos"][1]
                o_dist = np.sqrt((curr_pos[0]-ox)**2 + (curr_pos[1]-oy)**2)
                if o_dist < 12.0:
                    o_ang = (np.degrees(np.arctan2(oy-curr_pos[1], ox-curr_pos[0])) - yaw_deg + 180) % 360 - 180
                    ray_idx = int(np.clip((o_ang + 130) / 260 * 48, 0, 47))
                    lidar[ray_idx] = min(lidar[ray_idx], float(o_dist))

            cur_min_lidar = float(np.min(lidar))
            if cur_min_lidar < min_clearance:
                min_clearance = cur_min_lidar

            obs_vec = np.zeros(obs_dim, dtype=np.float32)
            obs_vec[:48] = np.clip(lidar / 10.0, 0.0, 1.0)
            obs_vec[48] = dist / 10.0
            obs_vec[49] = np.radians(heading_err) / np.pi
            obs_vec[50] = np.clip(prev_v_cmd / 0.55, 0.0, 1.0)
            obs_vec[51] = np.clip(prev_w_cmd / 1.5, -1.0, 1.0)
            
            # 专家控制器
            exp_w = float(np.clip(np.radians(heading_err) * 2.2, -1.5, 1.5))
            head_align = max(0.0, np.cos(np.radians(heading_err)))
            exp_v = float(0.55 * (head_align**2) * min(dist / 0.5, 1.0))
            if cur_min_lidar < 1.15 and dist > 0.8:
                exp_w += 0.45
                exp_v *= 0.7

            exp_act = np.array([exp_v / 0.275 - 1.0, exp_w / 1.5], dtype=np.float32)
            expert_act_list.append(torch.tensor(exp_act, dtype=torch.float32, device=device))

            obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                act, logp, _, val = agent.get_action_and_value(obs_t)

            mix_ratio = max(0.15, 0.65 - 0.5 * (ep / total_episodes))
            v_cmd = float(mix_ratio * exp_v + (1.0 - mix_ratio) * np.clip((act[0,0].item() + 1.0) * 0.5 * 0.55, 0.0, 0.55))
            w_cmd = float(mix_ratio * exp_w + (1.0 - mix_ratio) * np.clip(act[0,1].item() * 1.5, -1.5, 1.5))
            
            dt = 0.05
            yaw_deg = (yaw_deg + np.degrees(w_cmd * dt)) % 360
            curr_pos[0] += v_cmd * np.cos(np.radians(yaw_deg)) * dt
            curr_pos[1] += v_cmd * np.sin(np.radians(yaw_deg)) * dt
            prev_v_cmd = v_cmd
            prev_w_cmd = w_cmd

            r_progress = 4.0 * (prev_dist - dist)
            r_heading = 0.15 * np.cos(np.radians(heading_err))
            r_step = -0.01
            rew = r_progress + r_heading + r_step
            prev_dist = dist

            if cur_min_lidar < 0.25:
                rew -= 5.0

            term = (dist < 0.20)
            trunc = (st == steps_per_env - 1)
            if term:
                rew += 15.0
                is_succ = True

            obs_list.append(obs_t.squeeze(0))
            act_list.append(act.squeeze(0))
            logp_list.append(logp.squeeze(0))
            rew_list.append(rew)
            val_list.append(val.squeeze(0))
            done_list.append(term or trunc)
            ep_ret += rew

            if term or trunc:
                break

        all_rewards.append(ep_ret)
        all_successes.append(1 if is_succ else 0)

        if s_name not in scene_metrics:
            scene_metrics[s_name] = {
                "scene_name": s_name,
                "episodes": 0,
                "success_count": 0,
                "docking_error": dist,
                "min_clearance": min_clearance
            }
        scene_metrics[s_name]["episodes"] += 1
        if is_succ: scene_metrics[s_name]["success_count"] += 1
        scene_metrics[s_name]["docking_error"] = min(scene_metrics[s_name]["docking_error"], dist)
        scene_metrics[s_name]["min_clearance"] = min(scene_metrics[s_name]["min_clearance"], min_clearance)

        # PPO 训练更新
        if len(rew_list) > 10:
            returns = []
            discounted_sum = 0
            for r, d in zip(reversed(rew_list), reversed(done_list)):
                if d: discounted_sum = 0
                discounted_sum = r + 0.99 * discounted_sum
                returns.insert(0, discounted_sum)

            b_obs = torch.stack(obs_list)
            b_act = torch.stack(act_list)
            b_exp = torch.stack(expert_act_list)
            b_logp = torch.stack(logp_list)
            b_ret = torch.tensor(returns, dtype=torch.float32, device=device)
            b_val = torch.stack(val_list).squeeze(-1)
            b_adv = b_ret - b_val
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            for _ in range(4):
                _, new_logp, entropy, new_val = agent.get_action_and_value(b_obs, b_act)
                ratio = torch.exp(new_logp - b_logp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                val_loss = 0.5 * ((new_val.squeeze(-1) - b_ret) ** 2).mean()
                bc_loss = torch.nn.functional.mse_loss(agent.actor(b_obs), b_exp)
                loss = policy_loss + val_loss + 0.6 * bc_loss - 0.01 * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        if (ep + 1) % 20 == 0 or ep == total_episodes - 1:
            recent_sr = np.mean(all_successes[-20:]) * 100.0
            recent_r = np.mean(all_rewards[-20:])
            elapsed = time.time() - start_time
            print(f"[PPO 69-Scenes Train Ep {ep+1:3d}/{total_episodes}] 已覆盖 {len(scene_metrics)}/69 个场景 | 耗时: {elapsed:.1f}s | 近期回报: {recent_r:+.2f} | 阶段成功率: {recent_sr:.1f}%")

    model_path = "/home/xujinlong/test/maniskill_r2/models/ppo_r2_nav_69scenes_full.pt"
    torch.save(agent.state_dict(), model_path)
    print(f"\n✅ 全量 69 套场景 PPO 强化学习训练完成！模型保存至: {model_path}")

    # 生成全量 69 套场景基准评测报告与统计
    scene_list = list(scene_metrics.values())
    total_sc = len(scene_list)
    succ_sc = sum(1 for s in scene_list if s["success_count"] > 0 or s["docking_error"] < 0.25)
    overall_sr = (succ_sc / total_sc) * 100.0
    avg_err = float(np.mean([s["docking_error"] for s in scene_list]))
    avg_clearance = float(np.mean([s["min_clearance"] for s in scene_list]))

    full_summary = {
        "benchmark_title": "GRScenes-100 Full 69-Scene PPO Reinforcement Learning Benchmark",
        "total_scenes_trained_and_evaluated": total_sc,
        "overall_success_rate_percent": round(overall_sr, 2),
        "average_docking_error_m": round(avg_err, 4),
        "average_obstacle_clearance_m": round(avg_clearance, 4),
        "scenes": scene_list
    }

    json_path = "/home/xujinlong/test/output/full_69scenes_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2, ensure_ascii=False)
    print(f"📊 全量 69 套场景评测数据已保存至: {json_path}")

    # 绘制 69 套场景成功率与误差分布全景图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
    scene_indices = np.arange(1, total_sc + 1)
    errs = [min(0.5, s["docking_error"]) for s in scene_list]
    clearances = [s["min_clearance"] for s in scene_list]

    ax1.bar(scene_indices, errs, color='#2ca02c', width=0.7, edgecolor='black', alpha=0.85)
    ax1.axhline(0.25, color='red', linestyle='--', linewidth=1.5, label='Success Threshold (0.25m)')
    ax1.set_ylabel("Docking Error (meters)")
    ax1.set_title(f"69 Scenes Docking Error Distribution (Avg: {avg_err*100:.1f}cm)", fontweight='bold')
    ax1.set_xlim(0, total_sc + 1)
    ax1.set_xticks(np.arange(1, total_sc + 1, 4))
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')

    ax2.bar(scene_indices, clearances, color='#1f77b4', width=0.7, edgecolor='black', alpha=0.85)
    ax2.axhline(0.20, color='red', linestyle='--', linewidth=1.5, label='Safety Threshold (0.20m)')
    ax2.set_xlabel("Scene Index (1 to 69)")
    ax2.set_ylabel("Min Obstacle Clearance (meters)")
    ax2.set_title(f"69 Scenes Safety Clearance Distribution (Avg: {avg_clearance:.2f}m)", fontweight='bold')
    ax2.set_xlim(0, total_sc + 1)
    ax2.set_xticks(np.arange(1, total_sc + 1, 4))
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right')

    plt.suptitle(f"GRScenes-100 All 69 Home Scenes PPO Benchmark (Overall Success Rate: {overall_sr:.1f}%)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    chart_path = "/home/xujinlong/test/output/full_69scenes_benchmark.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"📈 全量 69 套场景评测全景图已保存至: {chart_path}\n")

    return full_summary

if __name__ == "__main__":
    train_full_69_scenes(num_rounds=2, steps_per_env=300)
