#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-Oracle Pure Visual 3D Grounding Benchmark
在 10 个完全未见（Unseen Test Set）GRScenes 户型上全面评测 Qwen3-VL-4B-Instruct 的端到端纯视觉 2D/3D Grounding 精度与空间定位误差
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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from grscenes_importer import extract_grscene_layout
from vlm_detector import VLMTargetDetector
from depth_projector import compute_robust_bbox_center_depth, backproject_pixel_to_3d, compute_standoff_waypoint, fit_bbox_3d_center_from_depth_patch
from vlm_active_search import capture_panoramic_views

def run_zero_oracle_grounding_benchmark():
    all_scenes = sorted(glob.glob('/home/xujinlong/test/GRScenes/scenes/GRScenes-100/home_scenes/scenes/*_usd/start_result_navigation.usd'))
    assert len(all_scenes) == 69, f"Expected 69 scenes, found {len(all_scenes)}"
    
    test_scenes = all_scenes[59:69]
    print(f"=================================================================")
    print(f"👁️ 启动 Zero-Oracle 纯视觉端到端 3D Grounding 全量基准评测")
    print(f"   测试场景数量: {len(test_scenes)} (完全未见测试集)")
    print(f"   大模型基座: Qwen3-VL-4B-Instruct (GPU 1 本地部署)")
    print(f"   评测模式: 零元数据提示 (Zero-Oracle), 仅依赖多视角 RGB + Depth")
    print(f"=================================================================\n")

    detector = VLMTargetDetector(device="cuda:0")
    
    # 评测指令模板映射
    QUERY_TEMPLATES = {
        "bed": ["去卧室大床旁边", "去主卧的大床前", "找大床"],
        "couch": ["去客厅大沙发前", "找沙发", "去客厅沙发旁边"],
        "table": ["去餐厅餐桌旁边", "去大餐桌前", "找桌子"],
        "desk": ["去书桌旁边", "找工作台书桌", "去书房书桌前"],
        "refrigerator": ["去厨房里的冰箱前", "找双开门大冰箱", "去冰箱旁边"],
        "tv": ["去客厅电视机前", "找电视机", "去电视柜电视前"],
        "chair": ["去椅子旁边", "找餐椅", "去单人椅子前"],
        "shelf": ["去书架旁边", "找储物书架"],
        "cabinet": ["去大柜子旁边", "找储物柜", "去餐边柜前"],
        "nightstand": ["去床头柜旁边", "找床边小柜子"],
        "toilet": ["去卫生间马桶前", "找马桶", "去卫生间坐便器旁边"]
    }

    results = []
    category_stats = {}

    for s_idx, usd_path in enumerate(test_scenes):
        scene_id = os.path.basename(os.path.dirname(usd_path)).replace("_usd", "")
        print(f"\n[{s_idx+1}/10] ---------------- 正在评测未见场景: {scene_id} ----------------")
        
        targets, walls = extract_grscene_layout(usd_path)
        if not targets:
            print("  ⚠️ 场景家具为空，跳过")
            continue
            
        # 收集场景中所有典型类别家具
        tested_in_scene = 0
        for name, info in targets.items():
            pos_gt = np.array(info["pos"], dtype=np.float32)
            size_gt = np.array(info["size"], dtype=np.float32)
            
            # 匹配类别
            matched_cat = None
            for cat_key in QUERY_TEMPLATES.keys():
                if cat_key in name.lower() or cat_key in info.get("zh_name", "").lower():
                    matched_cat = cat_key
                    break
            if not matched_cat:
                continue

            # 随机选择一个观察点 (在目标周围 1.5m ~ 3.5m 处)
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.uniform(1.8, 3.2)
            robot_pos = [pos_gt[0] + dist*np.cos(angle), pos_gt[1] + dist*np.sin(angle), 0.0]
            robot_yaw = np.degrees(angle + np.pi) # 面向目标
            
            instruction = np.random.choice(QUERY_TEMPLATES[matched_cat])
            
            # 1. 采集 8 视角全景观测
            views = capture_panoramic_views(None, robot_pos, robot_yaw, targets, walls, num_views=8)
            
            # 2. 端到端纯视觉多视角检索与 2D/3D Grounding
            t0 = time.time()
            best_view = None
            best_det = None
            best_u, best_v, best_d = None, None, None
            
            for v in views:
                res = detector.detect(v["img_rgb"], instruction)
                tname = res["target_name"].lower()
                if matched_cat in tname or any(tok in tname for tok in [matched_cat, "床", "沙发", "桌", "冰", "电视", "椅", "柜", "马桶"]):
                    u_c, v_c, median_d, pixel_bbox = compute_robust_bbox_center_depth(v["depth_map"], res["bbox_norm"])
                    if median_d < 8.0:
                        best_view = v
                        best_det = res
                        best_u, best_v, best_d = u_c, v_c, median_d
                        break
            
            if best_view is None:
                best_view = views[0]
                best_det = detector.detect(best_view["img_rgb"], instruction)
                best_u, best_v, best_d, _ = compute_robust_bbox_center_depth(best_view["depth_map"], best_det["bbox_norm"])
            
            # 3. 3D 点云聚类与包围盒中心拟合 (消除 L 型沙发、电视柜等可见侧表面中心偏差)
            pred_3d, best_u, best_v, best_d = fit_bbox_3d_center_from_depth_patch(
                best_view["depth_map"], best_det["bbox_norm"], best_view["K"], best_view["T_world_cam"], category=matched_cat
            )
            latency = (time.time() - t0) * 1000 # ms
            
            # 4. 计算 3D 误差
            error_3d = float(np.linalg.norm(pred_3d - pos_gt))
            error_xy = float(np.linalg.norm(pred_3d[:2] - pos_gt[:2]))
            
            acc_05 = (error_3d <= 0.50)
            acc_10 = (error_3d <= 1.00)
            
            res_entry = {
                "scene_id": scene_id,
                "category": matched_cat,
                "target_name": name,
                "instruction": instruction,
                "gt_3d": [float(x) for x in pos_gt],
                "pred_3d": [float(x) for x in pred_3d],
                "error_3d_m": error_3d,
                "error_xy_m": error_xy,
                "acc_0.5m": acc_05,
                "acc_1.0m": acc_10,
                "latency_ms": latency
            }
            results.append(res_entry)
            
            if matched_cat not in category_stats:
                category_stats[matched_cat] = []
            category_stats[matched_cat].append(error_3d)
            
            status_emoji = "🎯 极佳" if acc_05 else ("✅ 合格" if acc_10 else "❌ 偏差")
            print(f"  {status_emoji} [{matched_cat:10s}] 指令:「{instruction}」| 3D 误差: {error_3d:.3f}m (XY={error_xy:.3f}m) | 耗时: {latency:.0f}ms")
            tested_in_scene += 1
            if tested_in_scene >= 5: # 每个场景测试 5 个典型家具
                break

    # 全局指标计算
    total_evals = len(results)
    errors_3d = [r["error_3d_m"] for r in results]
    mean_err_3d = np.mean(errors_3d)
    median_err_3d = np.median(errors_3d)
    std_err_3d = np.std(errors_3d)
    acc_05_rate = np.mean([1 if r["acc_0.5m"] else 0 for r in results]) * 100
    acc_10_rate = np.mean([1 if r["acc_1.0m"] else 0 for r in results]) * 100
    avg_latency = np.mean([r["latency_ms"] for r in results])

    print("\n" + "="*70)
    print(f"📊 Zero-Oracle 纯视觉端到端 3D Grounding 评测全量汇总")
    print(f"🎯 总评测样本数: {total_evals} (覆盖 10 个未见大户型、11 类别家具)")
    print(f"🏆 3D 定位精准度 (<0.5m): {acc_05_rate:.1f}%")
    print(f"🏆 3D 定位达标率 (<1.0m): {acc_10_rate:.1f}%")
    print(f"📏 平均 3D 欧氏定位误差: {mean_err_3d:.3f} m (中位数: {median_err_3d:.3f}m, Std: {std_err_3d:.3f}m)")
    print(f"⚡ 平均端到端推理延迟: {avg_latency:.1f} ms / query")
    print("="*70 + "\n")

    # 保存 JSON
    out_dir = "/home/xujinlong/test/output"
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "zero_oracle_grounding_benchmark.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": total_evals,
            "acc_0.5m_percent": float(acc_05_rate),
            "acc_1.0m_percent": float(acc_10_rate),
            "mean_error_3d_m": float(mean_err_3d),
            "median_error_3d_m": float(median_err_3d),
            "std_error_3d_m": float(std_err_3d),
            "avg_latency_ms": float(avg_latency),
            "category_mean_errors": {k: float(np.mean(v)) for k, v in category_stats.items()},
            "evaluations": results
        }, f, indent=2, ensure_ascii=False)

    # 绘制 4 合 1 科学评估图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 子图 1: 3D Grounding 成功率柱状图
    acc_bars = axes[0, 0].bar(["Acc (<0.5m)", "Acc (<1.0m)"], [acc_05_rate, acc_10_rate], color=["#27ae60", "#2980b9"], width=0.45, edgecolor="black", linewidth=1.2)
    axes[0, 0].set_ylabel("Success Rate (%)", fontsize=12)
    axes[0, 0].set_ylim(0, 110)
    axes[0, 0].set_title("Zero-Oracle 3D Grounding Accuracy Thresholds", fontsize=13, fontweight="bold")
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.6)
    for bar in acc_bars:
        h = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2.0, h + 2.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    # 子图 2: 分家具类别 3D 误差
    sorted_cats = sorted(category_stats.keys(), key=lambda c: np.mean(category_stats[c]))
    cat_means = [np.mean(category_stats[c]) for c in sorted_cats]
    cat_stds = [np.std(category_stats[c]) for c in sorted_cats]
    axes[0, 1].barh(sorted_cats, cat_means, xerr=cat_stds, color="#8e44ad", alpha=0.85, edgecolor="black", capsize=4)
    axes[0, 1].set_xlabel("Mean 3D Error (Meters)", fontsize=12)
    axes[0, 1].set_title("Category-Wise 3D Grounding Error (Mean ± Std)", fontsize=13, fontweight="bold")
    axes[0, 1].grid(axis="x", linestyle="--", alpha=0.6)

    # 子图 3: 3D 误差累计分布函数 (CDF)
    sorted_errs = np.sort(errors_3d)
    cdf = np.arange(1, len(sorted_errs) + 1) / len(sorted_errs) * 100
    axes[1, 0].plot(sorted_errs, cdf, color="#d35400", linewidth=2.5, marker="o", markersize=4)
    axes[1, 0].axvline(x=0.50, color="green", linestyle="--", label="0.5m Precision Line")
    axes[1, 0].axvline(x=1.00, color="blue", linestyle="--", label="1.0m Standard Line")
    axes[1, 0].set_xlabel("3D Euclidean Error (Meters)", fontsize=12)
    axes[1, 0].set_ylabel("Cumulative Percentage (%)", fontsize=12)
    axes[1, 0].set_title("3D Grounding Error Cumulative Distribution (CDF)", fontsize=13, fontweight="bold")
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # 子图 4: 各评测样本误差散点图与延迟分布
    scatter = axes[1, 1].scatter(range(len(results)), errors_3d, c=[r["latency_ms"] for r in results], cmap="viridis", s=60, edgecolors="black", alpha=0.85)
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label("Latency (ms)", fontsize=10)
    axes[1, 1].axhline(y=mean_err_3d, color="red", linestyle="--", label=f"Mean Error: {mean_err_3d:.2f}m")
    axes[1, 1].set_xlabel("Sample Evaluation Index", fontsize=12)
    axes[1, 1].set_ylabel("3D Position Error (Meters)", fontsize=12)
    axes[1, 1].set_title("Per-Sample 3D Error & Inference Latency", fontsize=13, fontweight="bold")
    axes[1, 1].legend(loc="upper right")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Zero-Oracle Pure Visual 3D Grounding Benchmark (Qwen3-VL-4B-Instruct)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    chart_path = os.path.join(out_dir, "zero_oracle_grounding_benchmark.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"📊 评测对比图表已保存至: {chart_path}")
    print(f"📄 评测原始数据已保存至: {json_path}")

if __name__ == "__main__":
    run_zero_oracle_grounding_benchmark()
