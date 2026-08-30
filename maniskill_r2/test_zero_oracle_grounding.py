#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit and Integration Tests for Zero-Oracle Pure Visual 3D Grounding
验证 Qwen3-VL-4B-Instruct 在零元数据辅助 (Zero-Oracle) 下的 2D 检测与 3D 深度反投影定位精度
"""

import os
import sys
import unittest
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from vlm_detector import VLMTargetDetector
from depth_projector import compute_robust_bbox_center_depth, backproject_pixel_to_3d, compute_standoff_waypoint, fit_bbox_3d_center_from_depth_patch
from vlm_active_search import capture_panoramic_views, run_active_360_search

class TestZeroOracleVisualGrounding(unittest.TestCase):

    def setUp(self):
        self.targets_info = {
            "Refrigerator (冰箱)": {"pos": [2.8, -1.3, 0.80], "size": [0.65, 0.65, 1.60], "color": [0.85, 0.85, 0.9, 1.0]},
            "Sofa (客厅大沙发)": {"pos": [4.2, -0.2, 0.40], "size": [0.9, 1.8, 0.80], "color": [0.3, 0.5, 0.7, 1.0]},
            "Bed (卧室大床)": {"pos": [-2.5, 2.8, 0.45], "size": [1.8, 2.0, 0.80], "color": [0.4, 0.6, 0.5, 1.0]},
            "DiningTable (餐桌)": {"pos": [3.2, 0.9, 0.38], "size": [0.8, 1.2, 0.75], "color": [0.7, 0.45, 0.2, 1.0]}
        }
        self.obstacles_info = [
            {"pos": [1.4, 0.0, 0.50], "size": [0.4, 0.4, 1.0], "color": [0.8, 0.2, 0.2, 1.0]}
        ]

    def test_01_vlm_detector_loading_and_2d_grounding(self):
        """测试 1: 验证 Qwen3-VL-4B-Instruct 模型加载与 2D 目标边界框预测"""
        print("\n[Test 1] 测试 Qwen3-VL 视觉大模型 2D Grounding 与意图提取...")
        detector = VLMTargetDetector(device="cuda:0")
        
        views = capture_panoramic_views(None, [0,0,0], 0.0, self.targets_info, self.obstacles_info, num_views=8)
        self.assertEqual(len(views), 8, "全景视角数量不等于 8")
        
        # 针对包含沙发的视角 (视角 1) 进行检测
        view_sofa = views[0]
        res = detector.detect(view_sofa["img_rgb"], "去客厅大沙发前")
        
        self.assertIn("target_name", res)
        self.assertIn("bbox_norm", res)
        self.assertEqual(len(res["bbox_norm"]), 4)
        print(f"  ✅ 2D 目标检测成功: target={res['target_name']}, bbox={res['bbox_norm']}, spatial={res['spatial_relation']}")

    def test_02_zero_oracle_depth_backprojection_accuracy(self):
        """测试 2: 验证无元数据辅助 (Zero-Oracle) 下由 2D 像素与深度图反投影得到 3D 坐标的精度"""
        print("\n[Test 2] 验证 Zero-Oracle 纯视觉 3D 坐标重建精度...")
        views = capture_panoramic_views(None, [0,0,0], 0.0, self.targets_info, self.obstacles_info, num_views=8)
        
        # 测试沙发真值与反投影预测值
        gt_pos = np.array(self.targets_info["Sofa (客厅大沙发)"]["pos"])
        view = views[0]
        
        pred_3d, u_c, v_c, depth = fit_bbox_3d_center_from_depth_patch(
            view["depth_map"], [347, 393, 504, 664], view["K"], view["T_world_cam"], category="sofa"
        )
        
        error_3d = np.linalg.norm(pred_3d - gt_pos)
        print(f"  🎯 真值坐标: {gt_pos} | 纯视觉预测: {pred_3d} | 3D 误差: {error_3d:.3f}m")
        self.assertLessEqual(error_3d, 0.60, f"3D 反投影定位误差超标: {error_3d:.3f}m > 0.60m")

    def test_03_active_360_search_e2e_grounding(self):
        """测试 3: 验证 360° 主动自旋视觉搜索端到端目标锁定与航路点生成"""
        print("\n[Test 3] 验证 360° 主动全景自旋搜索与最终停靠点生成...")
        goal_x, goal_y, theta_goal, out_img = run_active_360_search(
            instruction="去卧室大床旁边",
            targets_info=self.targets_info,
            obstacles_info=self.obstacles_info,
            robot_pos=[0,0,0],
            robot_yaw=0.0
        )
        
        self.assertTrue(os.path.exists(out_img), "360° 搜索合成图未生成")
        gt_bed = np.array(self.targets_info["Bed (卧室大床)"]["pos"][:2])
        dist_to_gt = np.linalg.norm(np.array([goal_x, goal_y]) - gt_bed)
        print(f"  ✅ 成功锁定背后盲区目标「卧室大床」, 停靠点=({goal_x:.2f}, {goal_y:.2f}), 距床中心={dist_to_gt:.2f}m")
        self.assertLessEqual(dist_to_gt, 1.50, f"停靠点偏离目标过大: {dist_to_gt:.2f}m > 1.50m")

if __name__ == "__main__":
    unittest.main()
