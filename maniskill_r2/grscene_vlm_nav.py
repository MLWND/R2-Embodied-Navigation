"""GRScenes-100 真实大户型长距离 VLM 自然语言寻物导航与多障碍物避障系统 (GRScene Long-Horizon VLM Nav)"""
import os
import argparse
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sapien

from grscenes_importer import extract_grscene_layout
from depth_projector import compute_robust_bbox_center_depth, backproject_pixel_to_3d, compute_standoff_waypoint, fit_bbox_3d_center_from_depth_patch
from vlm_detector import VLMTargetDetector
from vlm_active_search import capture_panoramic_views

URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_swivel.urdf"
MESH = "/home/xujinlong/test/robot_example1_EN_demo/robot_example1_EN_demo/robot_model/r2_urdf_v01/meshes"
TMP = "/tmp/r2_swivel_tmp.urdf"

# 差速动力学参数
WHEEL_R = 0.085
TRACK = 0.458
MAX_V = 0.55
MAX_W = 1.5

import enum
from collections import deque

class NavState(enum.Enum):
    SEARCH = 1              # 360° 全景主动视觉搜索
    LOCALIZE_TARGET = 2     # 2D/3D Grounding 与深度反投影锁定
    TOPOLOGICAL_PLAN = 3    # 场景语义拓扑路径规划
    NAVIGATING = 4          # 动态巡航与 PPO 实时避障
    RECOVERY = 5            # 发生卡死或局部陷阱时的脱困倒车/旋转
    ARRIVED = 6             # 成功精准停靠并正对目标
    FAILED = 7              # 任务超时或不可达

class NavigationStateMachine:
    """具身智能长距离导航状态机与卡死自动恢复控制器"""
    def __init__(self, stuck_threshold_steps=150, stuck_dist_threshold=0.05):
        self.state = NavState.NAVIGATING
        self.stuck_threshold_steps = stuck_threshold_steps
        self.stuck_dist_threshold = stuck_dist_threshold
        self.history = deque(maxlen=stuck_threshold_steps + 1)
        self.recovery_step = 0
        self.recovery_max_steps = 80

    def update_position(self, pos, current_step):
        px, py = float(pos[0]), float(pos[1])
        self.history.append((current_step, px, py))
        
        if self.state == NavState.NAVIGATING:
            if len(self.history) >= self.stuck_threshold_steps:
                oldest = self.history[0]
                latest = self.history[-1]
                dist_moved = np.sqrt((latest[1] - oldest[1])**2 + (latest[2] - oldest[2])**2)
                if dist_moved < self.stuck_dist_threshold:
                    self.state = NavState.RECOVERY
                    self.recovery_step = 0

    def get_recovery_command(self):
        """执行两阶段安全脱困：阶段 1 (0~40步) 倒车后退；阶段 2 (40~80步) 旋转重新建立雷达视场"""
        if self.state != NavState.RECOVERY:
            return {"v": 0.0, "w": 0.0}
            
        self.recovery_step += 1
        if self.recovery_step <= 40:
            # 阶段 1: 倒车后退
            cmd = {"v": -0.25, "w": 0.0}
        elif self.recovery_step <= self.recovery_max_steps:
            # 阶段 2: 原地自旋调整
            cmd = {"v": 0.0, "w": 0.9}
        else:
            # 恢复完成，重置历史并重返导航
            self.state = NavState.NAVIGATING
            self.history.clear()
            cmd = {"v": 0.0, "w": 0.0}
            
        return cmd

def euler(q):
    w, x, y, z = q
    roll = np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y-x*z), -1, 1)))
    yaw = np.degrees(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    return roll, pitch, yaw

def build_grscene_simulation(targets_info, walls_info):
    import re
    with open(URDF) as f:
        urdf = f.read()
    urdf = urdf.replace("package://r2_urdf_v01/meshes/", MESH + "/")
    urdf_no_vis = re.sub(r'<visual>.*?</visual>', '', urdf, flags=re.DOTALL)
    with open(TMP, "w") as f:
        f.write(urdf_no_vis)

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
    loader.set_material(1.5, 1.5, 0.0)
    robot = loader.load(TMP)
    # 设定标准巡航站姿 (收起机械臂，抬高腰部，避免双手拖地摩擦)
    qpos = np.zeros(robot.dof, dtype=np.float32)
    for idx, j in enumerate(robot.get_active_joints()):
        name = j.get_name()
        if 'J2_left' in name: qpos[idx] = 0.6
        elif 'J2_right' in name: qpos[idx] = -0.6
        elif 'J4_left' in name: qpos[idx] = 0.8
        elif 'J4_right' in name: qpos[idx] = 0.8
        elif 'lift' in name: qpos[idx] = 0.25
        elif 'waist' in name: qpos[idx] = 0.0

    robot.set_qpos(qpos)

    robot_shapes = set()
    for l in robot.get_links():
        for sh in l.get_collision_shapes():
            robot_shapes.add(sh)
            sh.set_collision_groups([1, 1, 1<<29, 0])

    for j in robot.get_joints():
        if 'swivel' in j.get_name():
            j.friction = 0.0
    for l in robot.get_links():
        if 'caster' in l.get_name() and 'wheel' in l.get_name():
            for sh in l.get_collision_shapes():
                sh.set_physical_material(sapien.physx.PhysxMaterial(0.01, 0.01, 0.0))

    wl_j = [j for j in robot.get_joints() if j.get_name() == "wheel_left_joint"][0]
    wr_j = [j for j in robot.get_joints() if j.get_name() == "wheel_right_joint"][0]
    base = [l for l in robot.get_links() if l.get_name() == "base_link"][0]

    wl_j.set_drive_properties(0.0, 5000.0, 20000.0, "force")
    wr_j.set_drive_properties(0.0, 5000.0, 20000.0, "force")

    for _ in range(50): scene.step()
    return scene, robot, wl_j, wr_j, base, robot_shapes

def simulate_lidar(scene, base_pos, base_yaw_deg, robot_shapes, num_rays=48, fov_deg=260.0, max_range=15.0):
    angles = np.linspace(-fov_deg/2, fov_deg/2, num_rays)
    ranges = []
    ray_dirs = []
    px = scene.physx_system
    origin = np.array([base_pos[0], base_pos[1], 0.30], dtype=np.float32)

    for deg in angles:
        world_deg = base_yaw_deg + deg
        rad = np.radians(world_deg)
        d = np.array([np.cos(rad), np.sin(rad), 0.0], dtype=np.float32)

        curr_origin = origin.copy()
        total_dist = 0.0
        hit_external = False

        while total_dist < max_range:
            hit = px.raycast(curr_origin.astype(np.float32), d, max_range - total_dist)
            if not hit:
                total_dist = max_range
                break
            total_dist += hit.distance
            if hit.shape in robot_shapes:
                curr_origin = np.array(hit.position) + d * 0.02
                total_dist += 0.02
            else:
                hit_external = True
                break

        ranges.append(float(total_dist if hit_external else max_range))
        ray_dirs.append(d[:2])

    return np.array(ranges), np.array(ray_dirs), angles

def diff_drive(v, omega):
    v_l = (v - omega * TRACK / 2) / WHEEL_R
    v_r = -(v + omega * TRACK / 2) / WHEEL_R
    return v_l, v_r

def run_grscene_vlm_navigation(instruction="去餐厅大餐桌旁边", usd_path=None):
    """GRScenes-100 端到端自然语言语义导航入口 (统一路由至 SceneSemanticGraph 层次拓扑规划器)"""
    return run_hierarchical_semantic_navigation(instruction=instruction, usd_path=usd_path)

def _legacy_single_room_nav(instruction="去餐厅大餐桌旁边", usd_path=None):
    from ppo_nav_agent import PPONavAgent
    import torch

    ppo_model_path = '/home/xujinlong/test/maniskill_r2/models/ppo_r2_nav_69scenes_full.pt'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ppo_agent = PPONavAgent(obs_dim=52, act_dim=2).to(device)
    if os.path.exists(ppo_model_path):
        ppo_agent.load_state_dict(torch.load(ppo_model_path, map_location=device))
    ppo_agent.eval()
    print(f'[PPO] ✅ 强化学习策略网络已加载 ({ppo_model_path}), 启用 PPO+Expert 混合控制')

    print(f"\n" + "="*70)
    print(f"🏠 启动 GRScenes-100 真实大户型长距离 VLM 寻物导航任务")
    print(f"💬 自然语言指令: 「{instruction}」")
    print(f"="*70 + "\n")

    targets_info, walls_info = extract_grscene_layout()
    scene, robot, wl_j, wr_j, base, robot_shapes = build_grscene_simulation(targets_info, walls_info)
    p_init = base.get_entity_pose().p
    yaw_init = euler(base.get_entity_pose().q)[2]

    print("[360 Search] 机器人正在原地分 8 视角扫描真实家庭环境，Qwen3-VL 大模型执行多模态目标锁定...")
    views = capture_panoramic_views(None, p_init, yaw_init, targets_info, walls_info, num_views=8)
    detector = VLMTargetDetector(device="cuda:0")

    query_kw = instruction.lower()
    expected_tokens = []
    if "餐桌" in query_kw or "table" in query_kw or "桌" in query_kw: expected_tokens.extend(["table", "桌", "dining"])
    elif "沙发" in query_kw or "couch" in query_kw or "sofa" in query_kw: expected_tokens.extend(["couch", "sofa", "沙发"])
    elif "床" in query_kw or "bed" in query_kw or "卧室" in query_kw: expected_tokens.extend(["bed", "床", "bedroom"])
    elif "电视" in query_kw or "tv" in query_kw: expected_tokens.extend(["tv", "电视"])
    elif "书架" in query_kw or "shelf" in query_kw: expected_tokens.extend(["shelf", "书架"])
    elif "椅" in query_kw or "chair" in query_kw: expected_tokens.extend(["chair", "椅"])
    elif "冰" in query_kw or "fridge" in query_kw: expected_tokens.extend(["fridge", "冰"])
    elif "柜" in query_kw or "cabinet" in query_kw or "wardrobe" in query_kw: expected_tokens.extend(["cabinet", "wardrobe", "柜"])
    else:
        # Fallback to characters in instruction
        expected_tokens.extend([c for c in query_kw if c not in "去到走前面旁边附近侧边在旁边"])

    candidates = []
    for i, v in enumerate(views):
        res = detector.detect(v["img_rgb"], instruction)
        target_name = res["target_name"].lower()
        u_c, v_c, median_d, pixel_bbox = compute_robust_bbox_center_depth(v["depth_map"], res["bbox_norm"])
        
        is_target_matched = any(tok in target_name for tok in expected_tokens)
        is_visible = any(tok in obj for tok in expected_tokens for obj in v["visible_objects"])

        if is_target_matched and is_visible and median_d < 8.5:
            depth_quality = 1.0 / (1.0 + abs(median_d - 6.5))
            candidates.append({
                "view": v,
                "detection": res,
                "u_c": u_c, "v_c": v_c, "depth": median_d, "pixel_bbox": pixel_bbox,
                "score": depth_quality
            })
            print(f"  - 候选视角 {i+1} (朝向 {v['angle_deg']:5.1f}°): 检测到 '{res['target_name']}', 深度={median_d:.2f}m, 视野质量得分={depth_quality:.2f}")

    if candidates:
        best_cand = max(candidates, key=lambda c: c["score"])
        best_view = best_cand["view"]
        best_detection = best_cand["detection"]
        best_u_c, best_v_c, best_depth, best_pixel_bbox = best_cand["u_c"], best_cand["v_c"], best_cand["depth"], best_cand["pixel_bbox"]
    else:
        best_view = views[0]
        best_detection = detector.detect(best_view["img_rgb"], instruction)
        best_u_c, best_v_c, best_depth, best_pixel_bbox = compute_robust_bbox_center_depth(best_view["depth_map"], best_detection["bbox_norm"])

    print(f"\n[VLM Locked] 🎯 Qwen3-VL 成功锁定最优无遮挡观测视角:")
    print(f"      - 最优视角朝向: {best_view['angle_deg']:.1f}°")
    print(f"      - 识别目标: {best_detection['target_name']}")
    print(f"      - 空间方位意图: {best_detection['spatial_relation']}")
    print(f"      - 测距深度: {best_depth:.2f} m")

    obj_world_pos = backproject_pixel_to_3d(best_u_c, best_v_c, best_depth, best_view["K"], best_view["T_world_cam"])

    # 查找匹配目标的物理尺寸，用于从表面而非中心计算停靠偏移
    obj_half_extents = None
    detected_name_lower = best_detection["target_name"].lower()
    for tname, tinfo in targets_info.items():
        if detected_name_lower in tname.lower() or any(tok in tname.lower() for tok in detected_name_lower.split("_")):
            sx, sy = tinfo["size"][0], tinfo["size"][1]
            obj_half_extents = (sx / 2.0, sy / 2.0)
            break

    goal_x, goal_y, theta_goal = compute_standoff_waypoint(
        obj_world_pos, [p_init[0], p_init[1], 0],
        spatial_relation=best_detection["spatial_relation"],
        standoff_dist=0.85,
        obj_half_extents=obj_half_extents
    )

    print(f"[3D Waypoint] 物体 3D 世界真实坐标: ({obj_world_pos[0]:.2f}, {obj_world_pos[1]:.2f}, {obj_world_pos[2]:.2f}) m")
    print(f"[3D Waypoint] 规划停靠目标点: (x={goal_x:.2f}, y={goal_y:.2f}) m, 期望朝向={theta_goal:+.1f}°\n")

    final_goal = (goal_x, goal_y)
    trajectory = []
    min_obstacle_dist = float('inf')

    prev_pos = None
    prev_yaw = None
    act_v = 0.0
    act_w = 0.0

    print("[Navigation] 启动 260° 激光雷达长距离多房间自主过门与走廊避障航行...")
    for step in range(15000):
        pos = base.get_entity_pose().p
        yaw = euler(base.get_entity_pose().q)[2]
        trajectory.append((pos[0], pos[1]))

        # 速度差分估计
        if prev_pos is not None:
            dt = 0.005 * 4
            dx_step = pos[0] - prev_pos[0]
            dy_step = pos[1] - prev_pos[1]
            dir_dot = dx_step * np.cos(np.radians(prev_yaw)) + dy_step * np.sin(np.radians(prev_yaw))
            act_v = (np.sqrt(dx_step**2 + dy_step**2) / dt) * (1 if dir_dot >= 0 else -1)
            dyaw = (yaw - prev_yaw + 180) % 360 - 180
            act_w = np.radians(dyaw) / dt
        prev_pos = pos
        prev_yaw = yaw

        dist_to_final = np.sqrt((pos[0]-final_goal[0])**2 + (pos[1]-final_goal[1])**2)
        if dist_to_final < 0.18:
            print(f"\n🎯 成功精准到达指定家具停靠点! 总步数={step}, 终点定位误差={dist_to_final:.3f}m")
            break

        dx = final_goal[0] - pos[0]
        dy = final_goal[1] - pos[1]
        target_angle_deg = np.degrees(np.arctan2(dy, dx))
        heading_err = (target_angle_deg - yaw + 180) % 360 - 180

        # Construct 52-dim obs
        obs = np.zeros(52, dtype=np.float32)
        ranges, ray_dirs, rel_angles = simulate_lidar(scene, pos, yaw, robot_shapes)
        if np.min(ranges) < min_obstacle_dist: min_obstacle_dist = np.min(ranges)
        obs[:48] = np.clip(ranges / 10.0, 0.0, 1.0)
        obs[48] = dist_to_final / 10.0
        obs[49] = np.radians(heading_err) / np.pi
        obs[50] = np.clip(act_v / MAX_V, 0.0, 1.0)
        obs[51] = np.clip(act_w / MAX_W, -1.0, 1.0)

        # PPO act
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        ppo_action = ppo_agent.act_deterministic(obs_t).squeeze(0).cpu().numpy()
        
        # PPO action -> velocity commands
        ppo_v = float(np.clip((ppo_action[0] + 1.0) * 0.5 * MAX_V, 0.0, MAX_V))
        ppo_w = float(np.clip(ppo_action[1] * MAX_W, -MAX_W, MAX_W))
        
        # Expert P-controller as safety fallback
        exp_w = float(np.clip(2.5 * np.radians(heading_err), -MAX_W, MAX_W))
        exp_align = max(0.0, float(np.cos(np.radians(heading_err))))
        exp_v = float(MAX_V * (exp_align ** 2) * np.clip(dist_to_final / 0.5, 0.40, 1.0))
        
        # Dynamic Blend: when turning into waypoint, follow expert steering; when aligned, blend with PPO
        alpha = 0.5 if abs(heading_err) < 25.0 else 0.1
        v_cmd = alpha * ppo_v + (1 - alpha) * exp_v
        if exp_align > 0.4:
            v_cmd = max(v_cmd, 0.28 * (exp_align ** 2))
        omega_cmd = alpha * ppo_w + (1 - alpha) * exp_w

        # Emergency stop if about to collide
        fwd_mask = (np.abs(rel_angles) < 30)
        fwd_min_dist = np.min(ranges[fwd_mask])
        if fwd_min_dist < 0.22:
            v_cmd *= 0.1  # emergency slow down
            omega_cmd = MAX_W * (1.0 if heading_err > 0 else -1.0)  # turn away

        v_l, v_r = diff_drive(v_cmd, omega_cmd)
        wl_j.set_drive_velocity_target(v_l)
        wr_j.set_drive_velocity_target(v_r)
        for _ in range(4):
            scene.step()

        if step % 200 == 0:
            print(f"[Step {step:4d}] 坐标=({pos[0]:+.2f}, {pos[1]:+.2f}) 距停靠点={dist_to_final:.2f}m 航向={yaw:+.1f}° v={v_cmd:.2f}m/s")

    print("[Alignment] 正在微调最终朝向，正对目标家具实体中心...")
    for _ in range(150):
        yaw = euler(base.get_entity_pose().q)[2]
        yaw_err = (theta_goal - yaw + 180) % 360 - 180
        if abs(yaw_err) < 2.0:
            break
        w_align = np.clip(2.0 * np.radians(yaw_err), -0.8, 0.8)
        v_l, v_r = diff_drive(0.0, w_align)
        wl_j.set_drive_velocity_target(v_l)
        wr_j.set_drive_velocity_target(v_r)
        scene.step()

    wl_j.set_drive_velocity_target(0.0)
    wr_j.set_drive_velocity_target(0.0)
    for _ in range(50): scene.step()

    final_pos = base.get_entity_pose().p
    final_yaw = euler(base.get_entity_pose().q)[2]
    final_err = np.sqrt((final_pos[0]-final_goal[0])**2 + (final_pos[1]-final_goal[1])**2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    img_show = best_view["img_rgb"].copy()
    d_d = ImageDraw.Draw(img_show)
    bx1, by1, bx2, by2 = best_pixel_bbox
    d_d.rectangle([bx1, by1, bx2, by2], outline="red", width=3)
    d_d.text((bx1+5, by1+5), f"{best_detection['target_name']} (LOCKED)", fill="red")
    axes[0].imshow(img_show)
    target_clean_name = best_detection['target_name'].split()[0]
    axes[0].set_title(f"1. Qwen3-VL 360 Grounding (View {best_view['angle_deg']:.0f} deg)\nTarget: {target_clean_name}")
    axes[0].axis("off")

    depth_vis = np.clip(best_view["depth_map"], 0.0, 8.0)
    im_d = axes[1].imshow(depth_vis, cmap="viridis")
    axes[1].plot(best_u_c, best_v_c, "r*", markersize=14, label=f"Centroid (Z={best_depth:.2f}m)")
    axes[1].legend(loc="lower right")
    axes[1].set_title("2. Aligned Depth Buffer")
    axes[1].axis("off")
    plt.colorbar(im_d, ax=axes[1], fraction=0.046, pad=0.04)

    traj_arr = np.array(trajectory)
    axes[2].plot(traj_arr[:, 0], traj_arr[:, 1], "b-", linewidth=2.5, label="R2 Trajectory")
    axes[2].plot(p_init[0], p_init[1], "go", markersize=9, label="Start (0,0)")
    axes[2].plot(goal_x, goal_y, "r*", markersize=12, label=f"Goal ({goal_x:.2f},{goal_y:.2f})")

    for name, info in targets_info.items():
        ox, oy, _ = info["pos"]
        sx, sy, _ = info["size"]
        rect = plt.Rectangle((ox - sx/2, oy - sy/2), sx, sy, color="blue", alpha=0.3)
        axes[2].add_patch(rect)
        axes[2].text(ox, oy, name.split()[0], ha="center", va="center", fontsize=8, fontweight="bold")

    for i, w in enumerate(walls_info):
        ox, oy, _ = w["pos"]
        sx, sy, _ = w["size"]
        rect = plt.Rectangle((ox - sx/2, oy - sy/2), sx, sy, color="gray", alpha=0.6)
        axes[2].add_patch(rect)
        axes[2].text(ox, oy, f"Wall {i+1}", ha="center", va="center", color="white", fontsize=7)

    axes[2].set_xlabel("X (meters)")
    axes[2].set_ylabel("Y (meters)")
    axes[2].set_title("3. GRScenes-100 Apartment BEV Trajectory")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="upper left")
    axes[2].set_xlim(-1.0, 9.0)
    axes[2].set_ylim(-5.0, 5.0)
    axes[2].set_aspect("equal")

    out_img_path = "/home/xujinlong/test/output/grscene_vlm_nav_result.png"
    plt.tight_layout()
    plt.savefig(out_img_path, dpi=150)
    plt.close()

    print(f"\n==================== GRScenes-100 VLM 导航评测汇总报告 ====================")
    print(f"💬 自然语言指令: 「{instruction}」")
    print(f"🧠 视觉大模型: Qwen3-VL-4B-Instruct (GPU 1 运行)")
    print(f"🎯 锁定目标家具: {best_detection['target_name']} | 意图: {best_detection['spatial_relation']}")
    print(f"📍 家具 3D 真实物理中心: ({obj_world_pos[0]:.3f}, {obj_world_pos[1]:.3f}) m")
    print(f"🚩 规划停靠目标点: ({goal_x:.3f}, {goal_y:.3f}) m | 期望朝向: {theta_goal:+.1f}°")
    print(f"🏁 机器人最终停止位姿: ({final_pos[0]:.3f}, {final_pos[1]:.3f}) m | 实际朝向: {final_yaw:+.1f}°")
    print(f"📏 最终停靠位置误差: {final_err:.4f} m (阈值 < 0.25m: {'✅ 成功达标' if final_err < 0.25 else '❌ 未达标'})")
    print(f"🛡️ 全程最近障碍物距离: {min_obstacle_dist:.3f} m (安全阈值 > 0.30m: {'✅ 安全无碰撞' if min_obstacle_dist > 0.30 else '❌ 发生干涉'})")
    print(f"🖼️ 可视化多模态报告图已保存至: {out_img_path}")
    print(f"===========================================================================\n")
    return final_err, min_obstacle_dist, out_img_path

def find_safe_spawn_point(targets_info):
    """自动在中心走廊区域检索无碰撞的安全机器人出生点 (优先主走廊中轴线，Clearance >= 0.50m)"""
    grid = []
    for y in np.arange(-3.0, 3.1, 0.25):
        grid.append((0.0, float(y)))
    for x in np.arange(-3.0, 3.1, 0.25):
        grid.append((float(x), 0.0))
    for x in np.arange(-3.0, 3.1, 0.25):
        for y in np.arange(-3.0, 3.1, 0.25):
            grid.append((float(x), float(y)))
    grid.sort(key=lambda pt: pt[0]**2 + pt[1]**2)

    seen = set()
    best_p = [0.0, 0.0]
    best_c = -1.0
    for gx, gy in grid:
        key = (round(gx, 2), round(gy, 2))
        if key in seen:
            continue
        seen.add(key)
        min_d = 100.0
        for k, v in targets_info.items():
            p, s = v['pos'], v['size']
            dx = max(0.0, abs(gx - p[0]) - s[0]/2)
            dy = max(0.0, abs(gy - p[1]) - s[1]/2)
            d = np.sqrt(dx**2 + dy**2)
            if d < min_d:
                min_d = d
        if min_d >= 0.50:
            return [float(gx), float(gy)]
        if min_d > best_c:
            best_c = min_d
            best_p = [float(gx), float(gy)]
    return best_p

def run_hierarchical_semantic_navigation(instruction="去厨房里的冰箱", usd_path=None, start_pos=None, max_steps=3500):
    """基于 SceneSemanticGraph 场景语义拓扑图与 PPO 强化学习策略的跨房间层次化长距离自主导航系统"""
    from ppo_nav_agent import PPONavAgent
    from scene_semantic_graph import build_scene_semantic_graph
    import torch

    if usd_path is None:
        usd_path = "/home/xujinlong/test/GRScenes/scenes/GRScenes-100/home_scenes/scenes/MV7J6NIKTKJZ2AABAAAAADA8_usd/start_result_navigation.usd"

    print("\n" + "="*75)
    print("🗺️ 启动基于场景语义拓扑图 (Scene Semantic Graph) 的层次化跨房间自主寻路")
    print(f"💬 自然语言指令: 「{instruction}」")
    print("="*75 + "\n")

    # 1. 构建全屋语义拓扑图并规划层次化拓扑路径
    graph = build_scene_semantic_graph(usd_path)
    graph.print_hierarchy()

    targets_info, walls_info = extract_grscene_layout(usd_path)
    scene, robot, wl_j, wr_j, base, robot_shapes = build_grscene_simulation(targets_info, walls_info)
    
    if start_pos is None:
        p_xy = find_safe_spawn_point(targets_info)
    else:
        p_xy = [float(start_pos[0]), float(start_pos[1])]
    
    p_init = [float(p_xy[0]), float(p_xy[1]), 0.0]
    robot.set_pose(sapien.Pose(p=p_init))
    wl_j.set_drive_velocity_target(0.0)
    wr_j.set_drive_velocity_target(0.0)
    for _ in range(20):
        scene.step()
    yaw_init = euler(robot.get_pose().q)[2]

    plan = graph.plan_topological_route([p_init[0], p_init[1]], instruction)
    target_obj = plan["target_object"]
    target_room = plan["target_room"]
    waypoints = plan["waypoints"]
    final_goal = plan["final_goal"]
    theta_goal = plan["theta_goal"]

    print(f"[Semantic Plan] 🎯 目标定位: [{target_room.zh_name}] -> [{target_obj.zh_name}]")
    print(f"[Semantic Plan] 🛣️ 规划拓扑阶段路径 ({len(waypoints)} 阶段):")
    for idx, wp in enumerate(waypoints):
        print(f"   Stage {idx+1}: {wp['desc']} -> ({wp['pos'][0]:.2f}, {wp['pos'][1]:.2f})")

    # 2. 加载 PPO 强化学习策略网络
    ppo_model_path = '/home/xujinlong/test/maniskill_r2/models/ppo_r2_nav_69scenes_full.pt'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ppo_agent = PPONavAgent(obs_dim=52, act_dim=2).to(device)
    if os.path.exists(ppo_model_path):
        ppo_agent.load_state_dict(torch.load(ppo_model_path, map_location=device))
    ppo_agent.eval()
    print(f'\n[PPO] ✅ 强化学习策略网络已加载 ({ppo_model_path}), 启用 PPO+Expert 混合控制\n')

    # 3. 逐阶段拓扑导航执行
    trajectory = []
    min_obstacle_dist = float('inf')
    prev_pos = None
    prev_yaw = None
    act_v = 0.0
    act_w = 0.0

    current_stage_idx = 1  # 从第 2 个航路点开始 (第 1 个是起点)
    sm = NavigationStateMachine(stuck_threshold_steps=250, stuck_dist_threshold=0.05)
    total_steps = 0

    print("[Navigation] 🚀 启动多阶段跨房间长距离拓扑自主航行...")
    for step in range(max_steps):
        total_steps = step
        pos = robot.get_pose().p
        yaw = euler(robot.get_pose().q)[2]
        trajectory.append((pos[0], pos[1]))

        # 实际速度差分估计
        if prev_pos is not None:
            dt = 0.005 * 4
            dx_step = pos[0] - prev_pos[0]
            dy_step = pos[1] - prev_pos[1]
            dir_dot = dx_step * np.cos(np.radians(prev_yaw)) + dy_step * np.sin(np.radians(prev_yaw))
            act_v = (np.sqrt(dx_step**2 + dy_step**2) / dt) * (1 if dir_dot >= 0 else -1)
            dyaw = (yaw - prev_yaw + 180) % 360 - 180
            act_w = np.radians(dyaw) / dt
        prev_pos = pos
        prev_yaw = yaw

        # 激光雷达避障模拟
        ranges, ray_dirs, rel_angles = simulate_lidar(scene, pos, yaw, robot_shapes)
        cur_min_range = np.min(ranges)
        if cur_min_range < min_obstacle_dist:
            min_obstacle_dist = cur_min_range

        # 当前阶段目标点
        curr_wp = waypoints[current_stage_idx]
        curr_target = curr_wp["pos"]
        dist_to_curr = np.sqrt((pos[0] - curr_target[0])**2 + (pos[1] - curr_target[1])**2)
        dist_to_final = np.sqrt((pos[0] - final_goal[0])**2 + (pos[1] - final_goal[1])**2)

        # 阶段切换判定
        is_final_stage = (current_stage_idx == len(waypoints) - 1)
        stage_switch_dist = 0.45 if not is_final_stage else 0.80

        if dist_to_final < 0.85:
            print(f"\n🎯 成功精准到达终点家具停靠区! 总步数={step}, 终点定位误差={dist_to_final:.3f}m")
            break
        elif dist_to_curr < stage_switch_dist:
            if is_final_stage:
                print(f"\n🎯 成功精准到达终点家具停靠区! 总步数={step}, 终点定位误差={dist_to_final:.3f}m")
                break
            else:
                current_stage_idx += 1
                next_wp = waypoints[current_stage_idx]
                print(f"  🚩 阶段完成！切换到 Stage {current_stage_idx+1}: {next_wp['desc']} -> ({next_wp['pos'][0]:.2f}, {next_wp['pos'][1]:.2f})")
                curr_target = next_wp["pos"]
                dist_to_curr = np.sqrt((pos[0] - curr_target[0])**2 + (pos[1] - curr_target[1])**2)

        # 航向角与人工势场法 (APF) 狭窄通道居中导引
        dx = curr_target[0] - pos[0]
        dy = curr_target[1] - pos[1]
        dist_to_curr = np.sqrt(dx**2 + dy**2)
        f_att = np.array([dx / max(dist_to_curr, 0.05), dy / max(dist_to_curr, 0.05)])

        # 48 维激光雷达实时斥力场 (左右对称障碍物天然形成门洞居中力矩)
        f_rep = np.zeros(2, dtype=np.float32)
        d_influence = 0.40
        for r_val, rel_a in zip(ranges, rel_angles):
            if 0.05 < r_val < d_influence:
                beam_angle_rad = np.radians(yaw + rel_a)
                beam_dir = np.array([np.cos(beam_angle_rad), np.sin(beam_angle_rad)])
                rep_mag = 0.12 * (1.0 / r_val - 1.0 / d_influence) / (r_val**2)
                f_rep -= rep_mag * beam_dir

        f_total = f_att + f_rep
        target_angle_deg = np.degrees(np.arctan2(f_total[1], f_total[0]))
        heading_err = (target_angle_deg - yaw + 180) % 360 - 180

        # 构建 52 维 PPO 观测
        obs = np.zeros(52, dtype=np.float32)
        obs[:48] = np.clip(ranges / 10.0, 0.0, 1.0)
        obs[48] = dist_to_curr / 10.0
        obs[49] = np.radians(heading_err) / np.pi
        obs[50] = np.clip(act_v / MAX_V, 0.0, 1.0)
        obs[51] = np.clip(act_w / MAX_W, -1.0, 1.0)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        ppo_action = ppo_agent.act_deterministic(obs_t).squeeze(0).cpu().numpy()

        ppo_v = float(np.clip((ppo_action[0] + 1.0) * 0.5 * MAX_V, 0.0, MAX_V))
        ppo_w = float(np.clip(ppo_action[1] * MAX_W, -MAX_W, MAX_W))
        exp_w = float(np.clip(2.5 * np.radians(heading_err), -MAX_W, MAX_W))
        exp_align = max(0.0, float(np.cos(np.radians(heading_err))))
        exp_v = float(MAX_V * (exp_align ** 3) * np.clip(dist_to_curr / 0.5, 0.40, 1.0))

        # 动态自适应 PPO+拓扑 Expert 融合控制
        if abs(heading_err) > 15.0:
            v_cmd = 0.0
            omega_cmd = exp_w
        else:
            alpha = 0.5 if abs(heading_err) < 10.0 else 0.1
            v_cmd = alpha * ppo_v + (1 - alpha) * exp_v
            if exp_align > 0.5:
                v_cmd = max(v_cmd, 0.35 * (exp_align ** 2))
            omega_cmd = alpha * ppo_w + (1 - alpha) * exp_w

        # 狭窄门洞通道保持稳定前进动量，防止原地犹豫震荡
        left_mask = (rel_angles >= 30) & (rel_angles <= 90)
        right_mask = (rel_angles >= -90) & (rel_angles <= -30)
        min_left = np.min(ranges[left_mask]) if np.any(left_mask) else 10.0
        min_right = np.min(ranges[right_mask]) if np.any(right_mask) else 10.0
        fwd_mask_wide = (np.abs(rel_angles) < 20)
        fwd_clear = (np.min(ranges[fwd_mask_wide]) > 0.30) if np.any(fwd_mask_wide) else True
        if min_left < 0.42 and min_right < 0.42 and fwd_clear and abs(heading_err) < 25.0:
            v_cmd = max(v_cmd, 0.32)

        # 智能卡死检测与两阶段脱困状态机 (仅在前进巡航状态下监控卡死，原地自旋时不误判)
        if abs(heading_err) <= 15.0 and v_cmd > 0.15:
            sm.update_position([pos[0], pos[1]], step)
        else:
            sm.history.clear()

        if sm.state == NavState.RECOVERY:
            rec_cmd = sm.get_recovery_command()
            v_cmd, omega_cmd = rec_cmd["v"], rec_cmd["w"]
        else:
            # 最小化防碰撞安全制动 (正前方窄视角防追尾，侧方交由 PPO 连续避障)
            fwd_mask = (np.abs(rel_angles) < 15)
            fwd_min_dist = np.min(ranges[fwd_mask])
            if fwd_min_dist < 0.18:
                v_cmd *= 0.1
                omega_cmd = MAX_W * (1.0 if heading_err > 0 else -1.0)

        v_l, v_r = diff_drive(v_cmd, omega_cmd)
        wl_j.set_drive_velocity_target(v_l)
        wr_j.set_drive_velocity_target(v_r)
        for _ in range(4):
            scene.step()

        if step % 200 == 0:
            print(f"[Step {step:4d}] 坐标=({pos[0]:+.2f}, {pos[1]:+.2f}) 距阶段点={dist_to_curr:.2f}m 距终点={dist_to_final:.2f}m 航向={yaw:+.1f}°")

    # 4. 终点对准目标物体中心朝向
    print("[Alignment] 正在微调最终朝向，对准目标物体实体...")
    for _ in range(150):
        yaw = euler(robot.get_pose().q)[2]
        yaw_err = (theta_goal - yaw + 180) % 360 - 180
        if abs(yaw_err) < 2.0:
            break
        w_align = np.clip(2.0 * np.radians(yaw_err), -0.8, 0.8)
        v_l, v_r = diff_drive(0.0, w_align)
        wl_j.set_drive_velocity_target(v_l)
        wr_j.set_drive_velocity_target(v_r)
        scene.step()

    wl_j.set_drive_velocity_target(0.0)
    wr_j.set_drive_velocity_target(0.0)
    for _ in range(50): scene.step()

    final_pos = robot.get_pose().p
    final_yaw = euler(robot.get_pose().q)[2]
    final_err = float(np.sqrt((final_pos[0] - final_goal[0])**2 + (final_pos[1] - final_goal[1])**2))

    # 5. 生成可视化多模态拓扑导航报告图
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

    # Panel 1: 语义图层次结构示意
    axes[0].text(0.05, 0.95, f"Scene: {graph.scene_id}", fontsize=12, fontweight='bold', transform=axes[0].transAxes)
    axes[0].text(0.05, 0.88, f"Query: \"{instruction}\"", fontsize=11, color='blue', transform=axes[0].transAxes)
    axes[0].text(0.05, 0.82, f"Target: [{target_room.zh_name}] -> [{target_obj.zh_name}]", fontsize=11, color='darkgreen', fontweight='bold', transform=axes[0].transAxes)
    
    y_pos = 0.72
    axes[0].text(0.05, y_pos, "Topological Plan:", fontsize=11, fontweight='bold', transform=axes[0].transAxes)
    y_pos -= 0.06
    for idx, wp in enumerate(waypoints):
        symbol = "🟢" if idx == 0 else ("🔴" if idx == len(waypoints)-1 else "➡️")
        axes[0].text(0.08, y_pos, f"{symbol} Stage {idx+1}: {wp['desc']}", fontsize=10, transform=axes[0].transAxes)
        axes[0].text(0.70, y_pos, f"({wp['pos'][0]:.2f}, {wp['pos'][1]:.2f})m", fontsize=9, color='gray', transform=axes[0].transAxes)
        y_pos -= 0.05

    # 统计数据
    y_pos -= 0.04
    axes[0].text(0.05, y_pos, "Execution Metrics:", fontsize=11, fontweight='bold', transform=axes[0].transAxes)
    y_pos -= 0.06
    axes[0].text(0.08, y_pos, f"• Total Steps: {total_steps}", fontsize=10, transform=axes[0].transAxes)
    y_pos -= 0.05
    axes[0].text(0.08, y_pos, f"• Docking Error: {final_err:.4f} m ({'SUCCESS' if final_err <= 0.25 else 'FAIL'})", fontsize=10, color='green' if final_err <= 0.25 else 'red', fontweight='bold', transform=axes[0].transAxes)
    y_pos -= 0.05
    axes[0].text(0.08, y_pos, f"• Min Clearance: {min_obstacle_dist:.3f} m ({'SAFE' if min_obstacle_dist >= 0.25 else 'WARN'})", fontsize=10, color='green' if min_obstacle_dist >= 0.25 else 'red', fontweight='bold', transform=axes[0].transAxes)
    axes[0].axis("off")
    axes[0].set_title("1. Scene Semantic Graph & Topological Route", fontweight='bold', fontsize=12)

    # Panel 2: 全户型鸟瞰轨迹与语义房间包络
    traj_arr = np.array(trajectory)
    axes[1].plot(traj_arr[:, 0], traj_arr[:, 1], "b-", linewidth=2.5, label="R2 Trajectory")
    axes[1].plot(p_init[0], p_init[1], "go", markersize=10, label="Start (0,0)")

    # 绘制房间包络
    room_colors = {
        "kitchen": "#ff9999", "dining_room": "#ffcc99", "living_room": "#99ccff",
        "master_bedroom": "#cc99ff", "bedroom": "#ffff99", "bathroom": "#99ffcc", "corridor": "#e6e6e6"
    }
    for r_id, room in graph.rooms.items():
        if r_id == "corridor": continue
        b = room.bounds
        w_rect = b[2] - b[0]
        h_rect = b[3] - b[1]
        c = room_colors.get(room.room_type, "#e0e0e0")
        rect = plt.Rectangle((b[0], b[1]), w_rect, h_rect, color=c, alpha=0.25, edgecolor="black", linestyle=":")
        axes[1].add_patch(rect)
        axes[1].text(room.center[0], room.center[1], room.zh_name, ha="center", va="center", fontsize=9, fontweight="bold", color="#333333")

    # 绘制拓扑航路点
    wp_arr = np.array([wp["pos"] for wp in waypoints])
    axes[1].plot(wp_arr[:, 0], wp_arr[:, 1], "ro--", markersize=6, linewidth=1.2, alpha=0.7, label="Topological Route")
    axes[1].plot(final_goal[0], final_goal[1], "r*", markersize=14, label=f"Goal ({final_goal[0]:.2f},{final_goal[1]:.2f})")

    # 绘制大件家具
    for name, info in targets_info.items():
        ox, oy, _ = info["pos"]
        sx, sy, _ = info["size"]
        rect = plt.Rectangle((ox - sx/2, oy - sy/2), sx, sy, color="navy", alpha=0.35)
        axes[1].add_patch(rect)
        axes[1].text(ox, oy, name.split()[0][:8], ha="center", va="center", fontsize=7)

    axes[1].set_xlabel("X (meters)")
    axes[1].set_ylabel("Y (meters)")
    axes[1].set_title(f"2. Multi-Room Topological BEV Trajectory: '{instruction}'", fontweight='bold', fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_xlim(-1.0, 18.5)
    axes[1].set_ylim(-6.5, 6.5)
    axes[1].set_aspect("equal")

    out_img_path = "/home/xujinlong/test/output/grscene_semantic_graph_nav.png"
    plt.tight_layout()
    plt.savefig(out_img_path, dpi=150)
    plt.close()

    print(f"\n==================== GRScenes-100 拓扑语义导航汇总报告 ====================")
    print(f"💬 自然语言指令: 「{instruction}」")
    print(f"🚪 目标语义房间: {target_room.zh_name} [{target_room.room_id}]")
    print(f"🎯 锁定目标家具: {target_obj.zh_name} ({target_obj.name})")
    print(f"📍 家具 3D 真实物理中心: ({target_obj.pos[0]:.3f}, {target_obj.pos[1]:.3f}) m")
    print(f"🚩 规划停靠目标点: ({final_goal[0]:.3f}, {final_goal[1]:.3f}) m | 期望朝向: {theta_goal:+.1f}°")
    print(f"🏁 机器人最终停止位姿: ({final_pos[0]:.3f}, {final_pos[1]:.3f}) m | 实际朝向: {final_yaw:+.1f}°")
    print(f"📏 最终停靠位置误差: {final_err:.4f} m (阈值 <= 0.25m: {'✅ 成功达标' if final_err <= 0.25 else '❌ 未达标'})")
    print(f"🛡️ 全程最近障碍物距离: {min_obstacle_dist:.3f} m (安全阈值 >= 0.25m: {'✅ 安全无碰撞' if min_obstacle_dist >= 0.25 else '❌ 发生干涉'})")
    print(f"🖼️ 拓扑导航多模态报告图已保存至: {out_img_path}")
    print(f"===========================================================================\n")

    return final_err, min_obstacle_dist, out_img_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", type=str, default="去厨房里的冰箱", help="自然语言指令")
    args = parser.parse_args()
    run_hierarchical_semantic_navigation(args.instruction)

