#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hierarchical Scene Semantic Graph + PPO Policy Unseen Scenes Benchmark
在 10 个完全未见（Unseen Test Set）GRScenes 户型上全面评测层次化语义拓扑导航系统的跨房间导航成功率、泛化能力与避障安全性
"""

import os
import sys
import glob
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 确保 maniskill_r2 目录在 Python 模块搜索路径中
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from scene_semantic_graph import build_scene_semantic_graph
from grscene_vlm_nav import run_hierarchical_semantic_navigation

def run_hierarchical_unseen_benchmark():
    all_scenes = sorted(glob.glob('/home/xujinlong/test/GRScenes/scenes/GRScenes-100/home_scenes/scenes/*_usd/start_result_navigation.usd'))
    assert len(all_scenes) == 69, f"Expected 69 scenes, found {len(all_scenes)}"
    
    # 严格取未见测试集 10 个场景 (Zero Leakage)
    test_scenes = all_scenes[59:69]
    print(f"=================================================================")
    print(f"🚀 启动 10 个未见场景 (Unseen Test Set) 层次化拓扑语义导航全量评测")
    print(f"   测试场景数量: {len(test_scenes)}")
    print(f"   评测架构: SceneSemanticGraph (拓扑路由) + PPO (局部避障动力学)")
    print(f"=================================================================\n")

    results = []
    
    # 典型测试指令集合
    test_instructions = [
        "去厨房里的冰箱",
        "去客厅沙发前",
        "去餐厅餐桌旁边",
        "去主卧的大床旁边"
    ]

    for s_idx, usd_path in enumerate(test_scenes):
        scene_id = os.path.basename(os.path.dirname(usd_path)).replace("_usd", "")
        print(f"\n[{s_idx+1}/10] ---------------- 正在评测未见场景: {scene_id} ----------------")
        
        # 1. 构建该场景的语义拓扑图
        graph = build_scene_semantic_graph(usd_path)
        print(f"  🏢 识别出 {len(graph.rooms)} 个功能房间: {[r.zh_name for r in graph.rooms.values()]}")
        
        scene_episodes = []
        for q_idx, instruction in enumerate(test_instructions):
            matches = graph.find_objects(instruction)
            if not matches:
                continue
            
            target_obj, target_room = matches[0]
            print(f"  🎯 任务 [{q_idx+1}]: 「{instruction}」 -> 目标: {target_obj.zh_name} ({target_room.zh_name})")
            
            t0 = time.time()
            final_err, min_clearance, out_img = run_hierarchical_semantic_navigation(
                instruction=instruction,
                usd_path=usd_path,
                start_pos=None,
                max_steps=2200
            )
            elapsed = time.time() - t0
            
            # 成功判定标准 (Embodied Navigation 评测标准): 停靠误差 <= 0.85m 且 避障余量 >= 0.22m
            is_success = (final_err <= 0.85) and (min_clearance >= 0.22)
            
            res_item = {
                "scene_id": scene_id,
                "instruction": instruction,
                "target_room": target_room.room_id,
                "target_object": target_obj.name,
                "final_err": float(final_err),
                "min_clearance": float(min_clearance),
                "success": bool(is_success),
                "elapsed_sec": float(elapsed)
            }
            scene_episodes.append(res_item)
            results.append(res_item)
            
            status_emoji = "✅ 成功" if is_success else "❌ 失败"
            print(f"     {status_emoji} | 停靠误差={final_err:.3f}m | 最小安全余量={min_clearance:.3f}m | 耗时={elapsed:.1f}s")

    # 汇总统计
    total_episodes = len(results)
    successful_episodes = sum(1 for r in results if r["success"])
    success_rate = (successful_episodes / total_episodes) * 100 if total_episodes > 0 else 0.0
    avg_err = np.mean([r["final_err"] for r in results])
    avg_clearance = np.mean([r["min_clearance"] for r in results])
    
    print("\n==================== 10 个未见场景评测全量汇总 ====================")
    print(f"📊 总测试轮次: {total_episodes}")
    print(f"🏆 层次化导航整体成功率: {success_rate:.1f}% ({successful_episodes}/{total_episodes})")
    print(f"📏 平均终点停靠误差: {avg_err:.3f} m")
    print(f"🛡️ 平均最小避障间隙: {avg_clearance:.3f} m")
    print(f"🔥 对比 Pure PPO 基线 (0.0% 跨房间成功率): 提升 +{success_rate:.1f}%")
    print("===================================================================\n")

    # 保存 JSON 评估数据
    out_dir = "/home/xujinlong/test/output"
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "hierarchical_unseen_benchmark.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_episodes": total_episodes,
            "success_rate": success_rate,
            "avg_error_m": float(avg_err),
            "avg_clearance_m": float(avg_clearance),
            "pure_ppo_baseline_success_rate": 0.0,
            "episodes": results
        }, f, indent=2, ensure_ascii=False)

    # 绘制对比柱状图与误差分布图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 子图 1: Pure PPO vs Hierarchical System 成功率对比
    methods = ["Pure PPO (End-to-End)", "Hierarchical (Graph + PPO)"]
    success_rates = [0.0, success_rate]
    colors = ["#e74c3c", "#2ecc71"]
    
    bars = axes[0].bar(methods, success_rates, color=colors, width=0.45, edgecolor="black", linewidth=1.2)
    axes[0].set_ylabel("Navigation Success Rate (%)", fontsize=12)
    axes[0].set_ylim(0, 110)
    axes[0].set_title("Unseen Test Scenes Navigation Success Rate", fontsize=13, fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, height + 2.5, f"{height:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    # 子图 2: 停靠误差与安全余量分布
    scene_labels = [r["scene_id"][:6] + f"-Q{i%4+1}" for i, r in enumerate(results)]
    errors = [r["final_err"] for r in results]
    clearances = [r["min_clearance"] for r in results]
    
    x = np.arange(len(results))
    width = 0.35
    axes[1].bar(x - width/2, errors, width, label="Docking Error (m)", color="#3498db", alpha=0.85)
    axes[1].bar(x + width/2, clearances, width, label="Min Clearance (m)", color="#f39c12", alpha=0.85)
    axes[1].axhline(y=0.25, color="red", linestyle="--", label="Target Standoff Threshold (0.25m)")
    axes[1].set_xlabel("Unseen Episode ID", fontsize=11)
    axes[1].set_ylabel("Meters", fontsize=11)
    axes[1].set_title("Per-Episode Docking Error & Obstacle Clearance", fontsize=13, fontweight="bold")
    axes[1].set_xticks(x[::2])
    axes[1].set_xticklabels(scene_labels[::2], rotation=45, ha="right", fontsize=8)
    axes[1].legend(loc="upper right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart_path = os.path.join(out_dir, "hierarchical_unseen_benchmark.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"📊 评测对比图表已保存至: {chart_path}")
    print(f"📄 评测原始数据已保存至: {json_path}")
    return success_rate

if __name__ == "__main__":
    run_hierarchical_unseen_benchmark()
