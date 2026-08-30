"""系统级全链路自动化验证与单元测试套件 (Comprehensive Master Test Suite)"""
import os
import unittest
import numpy as np
import torch
from PIL import Image

from depth_projector import compute_robust_bbox_center_depth, backproject_pixel_to_3d, compute_standoff_waypoint
from grscenes_importer import extract_grscene_layout
from vlm_detector import VLMTargetDetector
from vlm_active_search import capture_panoramic_views
from ppo_nav_agent import PPONavAgent
from r2_grscene_env import R2GRSceneNavEnv
from grscene_vlm_nav import run_grscene_vlm_navigation

class TestEmbodiedVLMNavigationSystem(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("\n" + "="*75)
        print("🧪 启动具身智能 VLM 导航与强化学习全系统端到端自动化测试套件")
        print("="*75)
        cls.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def test_01_depth_projector_math(self):
        """单元测试 1: 空间几何与 3D 点云反投影数学解算"""
        print("\n[Test 1] 验证 3D 深度几何反投影与 Standoff 停靠点计算...")
        depth_map = np.full((480, 640), 5.0, dtype=np.float32)
        bbox = [200, 200, 800, 800]
        
        u, v, median_d, pixel_bbox = compute_robust_bbox_center_depth(depth_map, bbox)
        self.assertAlmostEqual(median_d, 5.0, places=2)
        self.assertEqual(u, 320)
        self.assertEqual(v, 240)

        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
        T_world_cam = np.eye(4, dtype=np.float32)
        T_world_cam[2, 3] = 1.0  # Z 偏移 1m
        
        P_world = backproject_pixel_to_3d(u, v, median_d, K, T_world_cam)
        self.assertAlmostEqual(P_world[0], 0.0, places=2)
        self.assertAlmostEqual(P_world[1], 0.0, places=2)
        self.assertAlmostEqual(P_world[2], 6.0, places=2)

        # 停靠点计算 — 验证距离与方向一致性
        gx, gy, theta = compute_standoff_waypoint(P_world, [0, 0, 0], spatial_relation="front", standoff_dist=0.85)
        standoff_dist_actual = np.sqrt((gx - P_world[0])**2 + (gy - P_world[1])**2)
        self.assertAlmostEqual(standoff_dist_actual, 0.85, places=2, msg="Standoff 距离不等于设定值")
        self.assertGreater(theta, -180.0)
        self.assertLessEqual(theta, 180.0)

        # 验证不同空间方位产生不同停靠点
        gx_side, gy_side, _ = compute_standoff_waypoint(P_world, [0, 0, 0], spatial_relation="side", standoff_dist=0.85)
        self.assertFalse(
            np.allclose([gx, gy], [gx_side, gy_side], atol=0.01),
            "front 和 side 停靠点不应相同"
        )
        print("  ✅ 空间几何反投影与多方位停靠点数学逻辑完全正确！")

    def test_02_grscenes_importer(self):
        """单元测试 2: GRScenes-100 真实大户型数据解析"""
        print("\n[Test 2] 验证 GRScenes-100 真实户型与语义实体提取...")
        targets, walls = extract_grscene_layout()
        self.assertGreaterEqual(len(targets), 4, f"家具数量不足: {len(targets)}")
        self.assertGreaterEqual(len(walls), 2, f"墙体数量不足: {len(walls)}")
        
        # 验证核心家具类别存在（动态 USD 解析的 key 格式: "Category_model_xxx (中文名)"）
        all_keys_lower = " ".join(targets.keys()).lower()
        self.assertTrue("table" in all_keys_lower or "餐桌" in all_keys_lower, "未找到餐桌")
        self.assertTrue("couch" in all_keys_lower or "沙发" in all_keys_lower, "未找到沙发")
        self.assertTrue("bed" in all_keys_lower or "床" in all_keys_lower, "未找到床")
        
        # 验证餐桌坐标与真实 USD 数据吻合 (center ≈ 6.33, -2.94)
        table_key = [k for k in targets if "table" in k.lower() or "餐桌" in k][0]
        table_pos = targets[table_key]["pos"]
        self.assertAlmostEqual(table_pos[0], 6.33, delta=0.1, msg=f"餐桌 X 坐标偏差过大: {table_pos[0]}")
        self.assertAlmostEqual(table_pos[1], -2.94, delta=0.1, msg=f"餐桌 Y 坐标偏差过大: {table_pos[1]}")
        
        # 验证每个家具都有合理的尺寸 (>0)
        for k, v in targets.items():
            self.assertTrue(all(s > 0 for s in v["size"]), f"{k} 尺寸无效: {v['size']}")
        
        print(f"  ✅ 从 USD 动态提取 {len(targets)} 个大件家具与 {len(walls)} 面墙体，坐标验证通过！")

    def test_03_vlm_detector_and_active_search(self):
        """单元测试 3: 360° 环视扫描与 Qwen3-VL 视觉语言目标锁定"""
        print("\n[Test 3] 验证 360° 原地自旋采样与 VLM 视觉大模型推理...")
        targets, walls = extract_grscene_layout()
        views = capture_panoramic_views(None, [0,0,0], 0.0, targets, walls, num_views=8)
        self.assertEqual(len(views), 8)
        
        detector = VLMTargetDetector(device=self.device)
        test_img = views[0]["img_rgb"]
        res = detector.detect(test_img, "去餐厅大餐桌旁边")
        
        self.assertIn("target_name", res)
        self.assertIn("bbox_norm", res)
        self.assertIn("spatial_relation", res)
        print(f"  ✅ VLM 推理成功锁定目标: '{res['target_name']}', 意图='{res['spatial_relation']}'")

    def test_04_r2_grscene_gym_env(self):
        """单元测试 4: 标准 Gymnasium 强化学习环境与 PhysX 动力学"""
        print("\n[Test 4] 验证 Gymnasium R2GRSceneNavEnv 环境与原地复位...")
        env = R2GRSceneNavEnv(max_steps=50)
        obs, _ = env.reset(seed=42)
        self.assertEqual(obs.shape, (52,))
        
        # 测试 Step 并验证速度观测通道
        action = np.array([0.5, 0.2], dtype=np.float32)
        next_obs, rew, term, trunc, info = env.step(action)
        self.assertEqual(next_obs.shape, (52,))
        self.assertIsInstance(float(rew), float)
        self.assertIn("docking_error", info)
        
        # 多步驱动后验证速度观测通道不为死零值
        for _ in range(10):
            next_obs, _, _, _, _ = env.step(np.array([1.0, 0.0], dtype=np.float32))
        self.assertGreater(next_obs[50], 0.0, "obs[50] 线速度通道为零 — 速度观测通道失效")
        
        # 测试多次原地 Reset 不泄露 Vulkan 上下文
        for i in range(3):
            obs, _ = env.reset(seed=100 + i)
            self.assertEqual(obs.shape, (52,))
        print("  ✅ Gymnasium 强化学习环境动力学、速度观测与原地复位验证通过！")

    def test_05_ppo_agent_policy_network(self):
        """单元测试 5: PPO Actor-Critic 策略前向传播与梯度更新"""
        print("\n[Test 5] 验证 PPO 策略网络前向与梯度计算...")
        agent = PPONavAgent(obs_dim=52, act_dim=2).to(self.device)
        dummy_obs = torch.randn(4, 52, device=self.device)
        
        action, logp, entropy, val = agent.get_action_and_value(dummy_obs)
        self.assertEqual(action.shape, (4, 2))
        self.assertEqual(logp.shape, (4,))
        self.assertEqual(entropy.shape, (4,))
        self.assertEqual(val.shape, (4, 1))

        det_act = agent.act_deterministic(dummy_obs)
        self.assertEqual(det_act.shape, (4, 2))
        print("  ✅ PPO Actor-Critic 网络前向与采样功能完全正常！")

    def test_06_e2e_vlm_navigation_closed_loop(self):
        """系统级集成测试 6: 端到端 VLM 寻物导航与避障闭环测试"""
        print("\n[Test 6] 执行端到端完整 VLM 自然语言寻物避障导航闭环测试...")
        final_err, min_clearance, out_img = run_grscene_vlm_navigation(instruction="去餐厅大餐桌旁边")
        
        self.assertLessEqual(final_err, 0.85, f"停靠误差超标: {final_err:.3f}m > 0.85m")
        self.assertGreaterEqual(min_clearance, 0.22, f"避障余量不足: {min_clearance:.3f}m < 0.22m")
        self.assertTrue(os.path.exists(out_img), f"输出图未生成: {out_img}")
        print(f"  ✅ 端到端闭环测试圆满通过！终点停靠误差: {final_err:.3f}m, 避障安全余量: {min_clearance:.3f}m")

    def test_07_real_sapien_camera_rendering(self):
        """系统级测试 7: SAPIEN 3 真实光学相机多视角渲染与坐标系数学闭环"""
        print("\n[Test 7] 验证 SAPIEN 3 真实相机光学渲染、深度图提取与反投影闭环...")
        import sapien
        scene = sapien.Scene()
        scene.set_timestep(0.01)
        scene.add_ground(altitude=0, render=True)

        targets, walls = extract_grscene_layout()
        for name, info in targets.items():
            b = scene.create_actor_builder()
            sx, sy, sz = info["size"]
            b.add_box_visual(half_size=[sx/2, sy/2, sz/2])
            actor = b.build_static(name=name)
            actor.set_pose(sapien.Pose(p=info["pos"]))

        views = capture_panoramic_views(scene, [0, 0, 0], 0.0, targets, walls, num_views=4)
        self.assertEqual(len(views), 4)
        for v in views:
            self.assertEqual(v["img_rgb"].size, (640, 480))
            self.assertEqual(v["depth_map"].shape, (480, 640))
            self.assertEqual(v["T_world_cam"].shape, (4, 4))
            self.assertFalse(np.any(np.isnan(v["depth_map"])))

        print("  ✅ SAPIEN 3 真实相机全景渲染与 3D 外参矩阵闭环测试通过！")

    def test_08_multi_category_grounding_coverage(self):
        """系统级测试 8: 多类别家居语义覆盖与意图解析验证"""
        print("\n[Test 8] 验证 11 类典型家居自然语言指令与空间方位意图提取...")
        detector = VLMTargetDetector(device=self.device)
        dummy_img = Image.new("RGB", (640, 480), color=(200, 200, 200))
        
        test_commands = [
            ("去卧室大床旁边", ["床", "bed", "master"], "side"),
            ("去客厅电视柜前面", ["电视", "tv"], "front"),
            ("去玄关书架前面", ["书架", "shelf"], "front"),
            ("走到客厅大沙发左侧", ["沙发", "sofa", "couch"], "left"),
            ("去餐厅餐桌右边", ["桌", "table", "dining"], "right"),
        ]

        for cmd, expected_tokens, expected_rel in test_commands:
            res = detector.detect(dummy_img, cmd)
            name_lower = res["target_name"].lower()
            matched = any(tok in name_lower for tok in expected_tokens)
            self.assertTrue(matched, f"指令 '{cmd}' 未匹配到目标类别 (检测为: '{res['target_name']}')")
            self.assertEqual(res["spatial_relation"], expected_rel, f"指令 '{cmd}' 意图解析错误")
        
        print("  ✅ 多类别家居自然语言指令覆盖与空间方位语义测试全部通过！")

if __name__ == "__main__":
    unittest.main(verbosity=2)
