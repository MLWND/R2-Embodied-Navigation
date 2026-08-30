"""GRScenes-100 VLM 自然语言导航与避障量化基准评测脚本 (Benchmark Evaluator)

评测指标体系 (对齐国际具身智能标准 Habitat / ManiSkill / VLN-CE):
1. 成功判定标准 (Success Criteria):
   - 停靠误差 d_err <= 0.25 m
   - 全程无碰撞 (Min Obstacle Distance >= 0.25 m)
2. 准确率指标 (Accuracy Metrics):
   - 最终停靠点几何误差 (Final Docking Error, 米)
   - 最小安全避障余量 (Min Obstacle Clearance, 米)
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from grscenes_importer import extract_grscene_layout
from grscene_vlm_nav import run_grscene_vlm_navigation

def run_benchmark():
    test_cases = [
        {"instruction": "去餐厅大餐桌旁边", "target_key": "DiningRoom_Table (餐厅大餐桌)"},
        {"instruction": "走到客厅大沙发附近", "target_key": "LivingRoom_Couch (客厅大沙发)"},
        {"instruction": "去客厅电视柜前面", "target_key": "LivingRoom_TV (客厅电视柜)"},
        {"instruction": "去玄关书架前面", "target_key": "Hallway_Shelf (玄关书架)"},
    ]

    print("="*75)
    print("🚀 启动 GRScenes-100 VLM 具身导航量化评测基准测试 (4 组典型家居指令)")
    print("="*75)

    results = []
    targets_info, _ = extract_grscene_layout()

    for idx, tc in enumerate(test_cases):
        cmd = tc["instruction"]
        print(f"\n[{idx+1}/{len(test_cases)}] 正在评测指令: 「{cmd}」...")
        try:
            final_err, min_obs_dist, _ = run_grscene_vlm_navigation(instruction=cmd)
            is_success = (final_err <= 0.25 and min_obs_dist >= 0.25)
            results.append({
                "case_id": idx + 1,
                "instruction": cmd,
                "target": tc["target_key"].split()[0],
                "docking_error_m": round(float(final_err), 4),
                "min_obstacle_dist_m": round(float(min_obs_dist), 4),
                "success": bool(is_success)
            })
        except Exception as e:
            print(f"❌ 评测异常: {e}")
            results.append({
                "case_id": idx + 1,
                "instruction": cmd,
                "target": tc["target_key"].split()[0],
                "docking_error_m": 99.0,
                "min_obstacle_dist_m": 0.0,
                "success": False
            })

    # 计算汇总统计
    total = len(results)
    success_count = sum(1 for r in results if r["success"])
    sr = (success_count / total) * 100.0
    valid_errs = [r["docking_error_m"] for r in results if r["docking_error_m"] < 10.0]
    avg_err = float(np.mean(valid_errs)) if valid_errs else 0.0
    min_clearances = [r["min_obstacle_dist_m"] for r in results if r["min_obstacle_dist_m"] > 0]
    avg_clearance = float(np.mean(min_clearances)) if min_clearances else 0.0

    summary = {
        "total_test_cases": total,
        "success_rate_percent": sr,
        "avg_docking_error_m": round(avg_err, 4),
        "avg_obstacle_clearance_m": round(avg_clearance, 4),
        "cases": results
    }

    # 保存 JSON
    json_path = "/home/xujinlong/test/output/benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n📊 量化评测结果 JSON 已保存至: {json_path}")

    # 绘制量化柱状图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    case_names = [f"Case {r['case_id']}\n{r['target']}" for r in results]
    errs = [r["docking_error_m"] for r in results]
    colors = ['#2ca02c' if r['success'] else '#d62728' for r in results]

    ax1.bar(case_names, errs, color=colors, width=0.5, edgecolor='black')
    ax1.axhline(0.25, color='red', linestyle='--', label='Success Threshold (0.25m)')
    ax1.set_ylabel("Docking Error (meters)")
    ax1.set_title(f"Docking Precision (Avg: {avg_err:.3f}m)")
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    clearances = [r["min_obstacle_dist_m"] for r in results]
    ax2.bar(case_names, clearances, color='#1f77b4', width=0.5, edgecolor='black')
    ax2.axhline(0.20, color='red', linestyle='--', label='Safety Threshold (0.20m)')
    ax2.set_ylabel("Min Obstacle Clearance (meters)")
    ax2.set_title(f"Safety Clearance (Avg: {avg_clearance:.3f}m)")
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    plt.suptitle(f"GRScenes-100 VLM Navigation Benchmark (Success Rate: {sr:.1f}%)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    chart_path = "/home/xujinlong/test/output/benchmark_evaluation.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"📈 评测图表已保存至: {chart_path}")

    print("\n" + "="*70)
    print(f"📊 量化评测汇总:")
    print(f"   - 导航任务成功率 (Success Rate): {sr:.1f}% ({success_count}/{total})")
    print(f"   - 平均最终停靠误差 (Avg Docking Error): {avg_err*100:.1f} cm")
    print(f"   - 平均安全避障余量 (Avg Obstacle Clearance): {avg_clearance:.2f} m")
    print("="*70 + "\n")
    return summary

if __name__ == "__main__":
    run_benchmark()
