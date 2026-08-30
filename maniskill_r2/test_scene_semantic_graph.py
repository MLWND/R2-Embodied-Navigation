"""TDD Step 1: 场景语义图 (SceneSemanticGraph) 单元测试"""
import os
import unittest
import numpy as np

class TestSceneSemanticGraph(unittest.TestCase):
    def test_01_build_graph_from_scene(self):
        """测试 1: 从场景数据自动构建层次化语义拓扑图 (Home -> Rooms -> Objects)"""
        from scene_semantic_graph import build_scene_semantic_graph
        
        usd_path = "/home/xujinlong/test/GRScenes/scenes/GRScenes-100/home_scenes/scenes/MV7J6NIKTKJZ2AABAAAAADA8_usd/start_result_navigation.usd"
        graph = build_scene_semantic_graph(usd_path)
        
        self.assertIsNotNone(graph)
        self.assertGreaterEqual(len(graph.rooms), 4, "至少应识别出 4 个语义房间")
        
        room_types = [r.room_type for r in graph.rooms.values()]
        print("识别到的语义房间:", room_types)
        
        # 验证核心房间类型存在
        self.assertTrue(any("kitchen" in t for t in room_types), "未识别出 Kitchen")
        self.assertTrue(any("living" in t for t in room_types), "未识别出 LivingRoom")
        self.assertTrue(any("bedroom" in t for t in room_types), "未识别出 Bedroom")
        self.assertTrue(any("dining" in t for t in room_types), "未识别出 DiningRoom")
        
        # 验证对象被正确归属到对应房间
        kitchen = [r for r in graph.rooms.values() if "kitchen" in r.room_type][0]
        kitchen_obj_cats = [obj.category for obj in kitchen.objects]
        print("厨房内的物体类别:", kitchen_obj_cats)
        self.assertTrue(
            any(k in kitchen_obj_cats for k in ["refrigerator", "microwave", "oven", "hearth", "dishwasher"]),
            "厨房内未包含典型厨电"
        )
        
        living_room = [r for r in graph.rooms.values() if "living" in r.room_type][0]
        living_obj_cats = [obj.category for obj in living_room.objects]
        print("客厅内的物体类别:", living_obj_cats)
        self.assertTrue(
            any(k in living_obj_cats for k in ["couch", "sofa", "tv", "teatable", "tvstand"]),
            "客厅内未包含沙发/电视"
        )

    def test_02_topological_path_planning(self):
        """测试 2: 层次化拓扑路径规划 (Current Room -> Corridor -> Doorway -> Room Center -> Target Standoff)"""
        from scene_semantic_graph import build_scene_semantic_graph
        
        usd_path = "/home/xujinlong/test/GRScenes/scenes/GRScenes-100/home_scenes/scenes/MV7J6NIKTKJZ2AABAAAAADA8_usd/start_result_navigation.usd"
        graph = build_scene_semantic_graph(usd_path)

        # 规划指令: 去厨房里的冰箱
        instruction = "去厨房里的冰箱"
        start_pos = [0.0, 0.0]
        
        plan = graph.plan_topological_route(start_pos, instruction)
        self.assertIsNotNone(plan)
        self.assertEqual(plan["target_room"].room_type, "kitchen")
        self.assertEqual(plan["target_object"].category, "refrigerator")
        
        waypoints = plan["waypoints"]
        print("\n规划出的层次化拓扑航路点:")
        for idx, wp in enumerate(waypoints):
            print(f"  Stage {idx+1}: {wp['desc']} -> ({wp['pos'][0]:.2f}, {wp['pos'][1]:.2f})")
            
        self.assertGreaterEqual(len(waypoints), 3, "跨房间路径至少需要 3 个阶段航路点")
        
        # 验证最终航路点为冰箱外部的安全停靠点
        final_wp = waypoints[-1]["pos"]
        fridge_pos = plan["target_object"].pos
        dist_to_fridge = np.sqrt((final_wp[0] - fridge_pos[0])**2 + (final_wp[1] - fridge_pos[1])**2)
        self.assertGreaterEqual(dist_to_fridge, 0.5, "停靠点不能在冰箱内部")
        self.assertLessEqual(dist_to_fridge, 1.2, "停靠点距冰箱应在有效停靠距离内")

    def test_03_hierarchical_topological_e2e_navigation(self):
        """测试 3: 层次化拓扑图驱动的跨房间端到端自主长距离导航与避障闭环"""
        from grscene_vlm_nav import run_hierarchical_semantic_navigation
        
        instruction = "去厨房里的冰箱"
        final_err, min_clearance, out_img = run_hierarchical_semantic_navigation(instruction=instruction)
        
        self.assertLessEqual(final_err, 0.85, f"停靠误差超标: {final_err:.3f}m > 0.85m")
        self.assertGreaterEqual(min_clearance, 0.22, f"避障余量不足: {min_clearance:.3f}m < 0.22m")
        self.assertTrue(os.path.exists(out_img), f"报告图表未生成: {out_img}")
        print(f"  ✅ 跨房间长距离语义导航闭环成功！最终误差: {final_err:.3f}m, 避障余量: {min_clearance:.3f}m")

if __name__ == "__main__":
    unittest.main(verbosity=2)
