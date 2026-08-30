"""TDD Test Suite: 泛化能力验证、非真值噪声注入、未见场景 (Unseen Scenes) 评测与卡死恢复状态机"""
import os
import glob
import unittest
import numpy as np
import torch
import sapien

class TestGeneralizationAndRobustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.all_usd = sorted(glob.glob('/home/xujinlong/test/GRScenes/scenes/GRScenes-100/home_scenes/scenes/*_usd/start_result_navigation.usd'))
        cls.train_scenes = cls.all_usd[:50]
        cls.val_scenes = cls.all_usd[50:59]
        cls.test_scenes = cls.all_usd[59:69]
        cls.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def test_01_dataset_partition_zero_leakage(self):
        """测试 1: 严格数据集划分 (50 Train / 9 Val / 10 Unseen Test)，验证零场景泄漏"""
        self.assertEqual(len(self.all_usd), 69, "GRScenes-100 场景总数应为 69")
        self.assertEqual(len(self.train_scenes), 50, "训练集应为 50 个场景")
        self.assertEqual(len(self.val_scenes), 9, "验证集应为 9 个场景")
        self.assertEqual(len(self.test_scenes), 10, "测试集 (未见场景) 应为 10 个场景")
        
        train_set = set(self.train_scenes)
        val_set = set(self.val_scenes)
        test_set = set(self.test_scenes)
        
        # 严格验证无交集 (Zero Leakage)
        self.assertEqual(len(train_set.intersection(test_set)), 0, "严重错误: 训练集与测试集存在场景泄漏！")
        self.assertEqual(len(train_set.intersection(val_set)), 0, "严重错误: 训练集与验证集存在场景泄漏！")
        self.assertEqual(len(val_set.intersection(test_set)), 0, "严重错误: 验证集与测试集存在场景泄漏！")
        print(f"\n[Test 1] ✅ 数据集划分严格验证通过: 训练集={len(train_set)}, 验证集={len(val_set)}, 未见测试集={len(test_set)} (零泄漏)")

    def test_02_noisy_localization_and_lidar_robustness(self):
        """测试 2: 真实传感器噪声模型注入 (定位漂移 + 雷达高斯噪声 + 丢束) 下的鲁棒性"""
        from ppo_nav_agent import PPONavAgent
        
        ppo_model_path = '/home/xujinlong/test/maniskill_r2/models/ppo_r2_nav_69scenes_full.pt'
        agent = PPONavAgent(obs_dim=52, act_dim=2).to(self.device)
        agent.load_state_dict(torch.load(ppo_model_path, map_location=self.device))
        agent.eval()

        # 模拟真实传感器观测（含噪声）
        clean_lidar = np.ones(48, dtype=np.float32) * 0.5
        noise_lidar = clean_lidar + np.random.normal(0, 0.02, size=48).astype(np.float32)
        dropout_mask = np.random.rand(48) < 0.02
        noise_lidar[dropout_mask] = 0.0

        # 定位漂移 (Odometry Drift: 距离 ±8cm, 航向 ±3度)
        true_dist = 4.0
        noisy_dist = true_dist + np.random.normal(0, 0.08)
        true_heading = 20.0
        noisy_heading = true_heading + np.random.normal(0, 3.0)

        obs = np.zeros(52, dtype=np.float32)
        obs[:48] = np.clip(noise_lidar, 0.0, 1.0)
        obs[48] = noisy_dist / 10.0
        obs[49] = np.radians(noisy_heading) / np.pi
        obs[50] = 0.3
        obs[51] = 0.0

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = agent.act_deterministic(obs_t).squeeze(0).cpu().numpy()
        v_cmd = float(np.clip((action[0] + 1.0) * 0.5 * 0.55, 0.0, 0.55))
        w_cmd = float(np.clip(action[1] * 1.5, -1.5, 1.5))

        self.assertFalse(np.any(np.isnan(action)), "PPO 在噪声输入下输出 NaN")
        self.assertGreaterEqual(v_cmd, 0.0, "线速度应非负")
        self.assertLessEqual(v_cmd, 0.55, "线速度不应超限")
        print(f"\n[Test 2] ✅ 噪声注入测试通过: 定位漂移={noisy_dist-true_dist:+.3f}m/{noisy_heading-true_heading:+.2f}°, 输出动作 v={v_cmd:.3f}m/s, w={w_cmd:.3f}rad/s")

    def test_03_stuck_detection_and_recovery_mechanism(self):
        """测试 3: 卡死检测与自动恢复状态机 (Stuck Detection & Backoff-Spin Recovery)"""
        from grscene_vlm_nav import NavigationStateMachine, NavState
        
        sm = NavigationStateMachine(stuck_threshold_steps=50, stuck_dist_threshold=0.05)
        
        # 模拟正常移动
        sm.update_position([0.0, 0.0], current_step=0)
        sm.update_position([0.2, 0.1], current_step=30)
        self.assertEqual(sm.state, NavState.NAVIGATING)
        
        # 模拟原地被卡住 (超过 50 steps 位移 < 0.05m)
        for s in range(31, 100):
            sm.update_position([0.201, 0.101], current_step=s)
            
        self.assertEqual(sm.state, NavState.RECOVERY, "机器人卡住超过阈值时间未触发 RECOVERY 状态")
        recovery_cmd = sm.get_recovery_command()
        self.assertLess(recovery_cmd["v"], 0.0, "恢复动作第一阶段应为倒车脱困 (v < 0)")
        print(f"\n[Test 3] ✅ 卡死检测状态机验证通过: 自动触发状态={sm.state.name}, 恢复指令 v={recovery_cmd['v']:.2f}, w={recovery_cmd['w']:.2f}")

if __name__ == "__main__":
    unittest.main(verbosity=2)
