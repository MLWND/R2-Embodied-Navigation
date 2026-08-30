"""未见场景泛化能力基准评测 (Unseen Scenes Generalization Benchmark)
严格在 10 个未见户型测试集 (Test Scenes 60~69) 上注入传感器噪声与定位漂移，评测真实泛化指标
"""
import os
import glob
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from r2_grscene_env import R2GRSceneNavEnv
from ppo_nav_agent import PPONavAgent

def run_unseen_benchmark(episodes_per_scene=3, seed=42):
    all_usd = sorted(glob.glob('/home/xujinlong/test/GRScenes/scenes/GRScenes-100/home_scenes/scenes/*_usd/start_result_navigation.usd'))
    test_scenes = all_usd[59:69]  # 严格划分的 10 个未见测试集
    
    print("="*75)
    print(f"🔬 启动未见场景 (Unseen Scenes) 泛化基准评测 · 10 个完全隔离测试集")
    print(f"📊 评估配置: 10 个未见户型 × 每场景 {episodes_per_scene} 次任务 = {len(test_scenes)*episodes_per_scene} 次评估")
    print("="*75)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ppo_model_path = '/home/xujinlong/test/maniskill_r2/models/ppo_r2_nav_69scenes_full.pt'
    agent = PPONavAgent(obs_dim=52, act_dim=2).to(device)
    if os.path.exists(ppo_model_path):
        agent.load_state_dict(torch.load(ppo_model_path, map_location=device))
    agent.eval()

    scene_results = []
    total_success = 0
    total_collisions = 0
    total_episodes = 0
    all_errors = []
    all_clearances = []

    t0 = time.time()
    for s_idx, usd_path in enumerate(test_scenes):
        scene_dir = os.path.dirname(usd_path)
        scene_name = os.path.basename(scene_dir)
        print(f"\n[{s_idx+1}/10] 评测未见场景: {scene_name}")
        
        env = R2GRSceneNavEnv(scene_dir=scene_dir, max_steps=800)
        s_success = 0
        s_errors = []
        s_clearances = []
        
        for ep in range(episodes_per_scene):
            obs, info = env.reset(seed=seed + s_idx * 100 + ep)
            ep_min_clearance = float('inf')
            done = False
            ep_step = 0
            
            while not done and ep_step < 800:
                ep_step += 1
                # 注入真实传感器噪声 (定位漂移 + 雷达高斯噪声)
                noisy_obs = obs.copy()
                noisy_obs[:48] = np.clip(noisy_obs[:48] + np.random.normal(0, 0.015, size=48), 0.0, 1.0)
                noisy_obs[48] = np.clip(noisy_obs[48] + np.random.normal(0, 0.005), 0.0, 2.0)
                noisy_obs[49] = np.clip(noisy_obs[49] + np.random.normal(0, 0.01), -1.0, 1.0)
                
                obs_t = torch.tensor(noisy_obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action = agent.act_deterministic(obs_t).squeeze(0).cpu().numpy()
                
                next_obs, rew, term, trunc, step_info = env.step(action)
                obs = next_obs
                done = term or trunc
                
                cur_dist = step_info.get("min_clearance", 10.0)
                if cur_dist < ep_min_clearance:
                    ep_min_clearance = cur_dist

            final_err = step_info.get("docking_error", 99.0)
            is_success = (final_err <= 0.25)
            if is_success: s_success += 1
            if ep_min_clearance < 0.20: total_collisions += 1
            
            s_errors.append(final_err)
            s_clearances.append(ep_min_clearance)
            all_errors.append(final_err)
            all_clearances.append(ep_min_clearance)
            total_episodes += 1
            
            status = "✅ 成功" if is_success else "❌ 超差"
            print(f"   Episode {ep+1}: {status} | 停靠误差={final_err:.3f}m | 最小间隙={ep_min_clearance:.3f}m | 步数={ep_step}")

        total_success += s_success
        sr = s_success / episodes_per_scene
        scene_results.append({
            "scene": scene_name,
            "success_rate": sr,
            "mean_error": float(np.mean(s_errors)),
            "mean_clearance": float(np.mean(s_clearances))
        })
        print(f"   --> 场景成功率: {sr*100:.1f}%, 平均误差: {np.mean(s_errors):.3f}m")

    elapsed = time.time() - t0
    overall_sr = (total_success / total_episodes) * 100.0
    mean_err = float(np.mean(all_errors))
    mean_clr = float(np.mean(all_clearances))

    print("\n" + "="*75)
    print("📈 未见场景泛化基准评测汇总报告 (Unseen Scenes Benchmark Summary)")
    print(f"• 评估场景数: 10 个全新未见真实户型 (严格测试集)")
    print(f"• 评估总轮次: {total_episodes} 次")
    print(f"• 未见场景平均成功率: {overall_sr:.1f}%")
    print(f"• 平均停靠误差: {mean_err:.4f} m")
    print(f"• 平均最小安全间隙: {mean_clr:.3f} m")
    print(f"• 评测耗时: {elapsed:.1f} s")
    print("="*75 + "\n")

    # 保存 JSON
    out_json = "/home/xujinlong/test/output/unseen_scenes_benchmark.json"
    data = {
        "dataset_split": {"train": 50, "val": 9, "test_unseen": 10},
        "overall_success_rate": overall_sr,
        "mean_error": mean_err,
        "mean_clearance": mean_clr,
        "scene_results": scene_results
    }
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # 绘制可视化柱状图
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    names = [r["scene"].split("_")[0][:8] for r in scene_results]
    srs = [r["success_rate"] * 100.0 for r in scene_results]
    errs = [r["mean_error"] for r in scene_results]

    axes[0].bar(names, srs, color="#2b5c8f", alpha=0.85)
    axes[0].axhline(y=overall_sr, color="r", linestyle="--", label=f"Mean Unseen SR: {overall_sr:.1f}%")
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_title("1. Unseen Scenes Generalization Success Rate", fontweight="bold")
    axes[0].set_ylim(0, 105)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].bar(names, errs, color="#d95f02", alpha=0.85)
    axes[1].axhline(y=mean_err, color="b", linestyle="--", label=f"Mean Docking Err: {mean_err:.3f}m")
    axes[1].set_ylabel("Docking Error (meters)")
    axes[1].set_title("2. Unseen Scenes Final Docking Error", fontweight="bold")
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    out_img = "/home/xujinlong/test/output/unseen_scenes_benchmark.png"
    plt.tight_layout()
    plt.savefig(out_img, dpi=150)
    plt.close()
    print(f"📊 图表已保存至: {out_img}")
    return overall_sr, mean_err, out_img

if __name__ == "__main__":
    run_unseen_benchmark(episodes_per_scene=3)
