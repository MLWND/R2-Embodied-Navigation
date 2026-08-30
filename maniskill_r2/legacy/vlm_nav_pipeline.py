"""R2 轮式人形机器人: 基于 Qwen-VL 视觉语言大模型的自然语言导航与避障系统 (VLM Navigation Pipeline)

功能流程:
1. 搭建包含多个家具实体 (桌子、冰箱、沙发) 与沿途障碍物的 SAPIEN 具身仿真环境
2. 机器人头部 RGB-D 相机拍摄场景视觉与深度数据
3. Qwen-VL 进行视觉定位 (Visual Grounding)，提取目标物体 2D BBox 与空间方位意图 ("旁边", "前面", "附近")
4. Depth Projector 通过内/外参反投影计算目标 3D 物理空间坐标与安全停靠点 (x_goal, y_goal, θ_goal)
5. 260° 激光雷达与差速控制器驱动底盘，自主绕过沿途障碍物，精准停靠在目标物体指定方位
6. 生成多模态可视化全景报告图 (RGB 检测框 + 深度图 + BEV 轨迹图)
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sapien

from depth_projector import compute_robust_bbox_center_depth, backproject_pixel_to_3d, compute_standoff_waypoint
from vlm_detector import VLMTargetDetector

URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_swivel.urdf"
MESH = "/home/xujinlong/test/robot_example1_EN_demo/robot_example1_EN_demo/robot_model/r2_urdf_v01/meshes"
TMP = "/tmp/r2_swivel_tmp.urdf"

# 差速动力学参数
WHEEL_R = 0.085
TRACK = 0.458
MAX_V = 0.50
MAX_W = 1.5
SAFE_RADIUS = 0.85

def euler(q):
    w, x, y, z = q
    roll = np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y-x*z), -1, 1)))
    yaw = np.degrees(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    return roll, pitch, yaw

def build_embodied_scene():
    """构建包含多个家具目标与未知障碍物的物理场景"""
    with open(URDF) as f:
        urdf = f.read()
    urdf = urdf.replace("package://r2_urdf_v01/meshes/", MESH + "/")
    with open(TMP, "w") as f:
        f.write(urdf)

    scene = sapien.Scene()
    scene.set_timestep(0.005)
    scene.add_ground(altitude=0, render=True)

    # 场景目标物体定义: (名称, x, y, size_x, size_y, size_z, 颜色 RGBA)
    targets_info = {
        "Table (桌子)": {"pos": [3.2, 0.9, 0.38], "size": [0.8, 1.2, 0.75], "color": [0.7, 0.45, 0.2, 1.0]},
        "Fridge (冰箱)": {"pos": [2.8, -1.3, 0.80], "size": [0.65, 0.65, 1.60], "color": [0.85, 0.85, 0.9, 1.0]},
        "Sofa (沙发)": {"pos": [4.2, -0.2, 0.40], "size": [0.9, 1.8, 0.80], "color": [0.3, 0.5, 0.7, 1.0]},
    }
    
    # 静态沿途障碍物
    obstacles_info = [
        {"pos": [1.4, 0.0, 0.50], "size": [0.4, 0.4, 1.0], "color": [0.8, 0.2, 0.2, 1.0]},
    ]

    for name, info in targets_info.items():
        b = scene.create_actor_builder()
        sx, sy, sz = info["size"]
        b.add_box_collision(half_size=[sx/2, sy/2, sz/2])
        b.add_box_visual(half_size=[sx/2, sy/2, sz/2])
        actor = b.build_static(name=name)
        actor.set_pose(sapien.Pose(p=info["pos"]))

    for i, info in enumerate(obstacles_info):
        b = scene.create_actor_builder()
        sx, sy, sz = info["size"]
        b.add_box_collision(half_size=[sx/2, sy/2, sz/2])
        b.add_box_visual(half_size=[sx/2, sy/2, sz/2])
        actor = b.build_static(name=f"obstacle_{i}")
        actor.set_pose(sapien.Pose(p=info["pos"]))

    loader = scene.create_urdf_loader()
    loader.fix_root_link = False
    loader.set_material(1.5, 1.5, 0.0)
    robot = loader.load(TMP)
    robot.set_pose(sapien.Pose(p=[0, 0, 0.0]))
    robot.set_qpos(np.zeros(robot.dof))

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
                sh.set_physical_material(sapien.physx.PhysxMaterial(0.05, 0.05, 0.0))

    wl_j = [j for j in robot.get_joints() if j.get_name() == "wheel_left_joint"][0]
    wr_j = [j for j in robot.get_joints() if j.get_name() == "wheel_right_joint"][0]
    base = [l for l in robot.get_links() if l.get_name() == "base_link"][0]

    wl_j.set_drive_properties(0.0, 500.0, 2000.0, "force")
    wr_j.set_drive_properties(0.0, 500.0, 2000.0, "force")

    for _ in range(150):
        scene.step()

    return scene, robot, wl_j, wr_j, base, robot_shapes, targets_info, obstacles_info

def capture_head_camera_rgbd(robot_base_pos, robot_base_yaw_deg, targets_info, obstacles_info):
    W, H = 640, 480
    fov_rad = np.radians(75.0)
    fx = (W / 2.0) / np.tan(fov_rad / 2.0)
    fy = fx
    cx = W / 2.0
    cy = H / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    cam_pos = np.array([robot_base_pos[0], robot_base_pos[1], 1.35], dtype=np.float32)
    pitch_rad = np.radians(-8.0)
    yaw_rad = np.radians(robot_base_yaw_deg)
    
    cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
    cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
    
    fwd = np.array([cos_y * cos_p, sin_y * cos_p, sin_p])
    right = np.array([sin_y, -cos_y, 0.0])
    up = np.cross(fwd, right)
    
    R_cam_world = np.stack([right, -up, fwd], axis=1)
    T_world_cam = np.eye(4, dtype=np.float32)
    T_world_cam[:3, :3] = R_cam_world
    T_world_cam[:3, 3] = cam_pos

    depth_map = np.full((H, W), 10.0, dtype=np.float32)
    img_rgb = Image.new("RGB", (W, H), color=(235, 238, 242))
    draw = ImageDraw.Draw(img_rgb)
    draw.rectangle([0, int(H*0.4), W, H], fill=(210, 215, 220))

    all_objects = list(targets_info.items()) + [(f"Obstacle_{i+1}", obs) for i, obs in enumerate(obstacles_info)]
    sorted_objs = sorted(all_objects, key=lambda item: -np.linalg.norm(np.array(item[1]["pos"]) - cam_pos))

    for name, info in sorted_objs:
        ox, oy, oz = info["pos"]
        sx, sy, sz = info["size"]
        P_w = np.array([ox, oy, oz, 1.0])
        P_c = np.linalg.inv(T_world_cam) @ P_w
        Xc, Yc, Zc = P_c[:3]

        if Zc > 0.3:
            u_c = int(cx + fx * Xc / Zc)
            v_c = int(cy + fy * Yc / Zc)
            w_px = max(int(fx * sy / Zc), 15)
            h_px = max(int(fy * sz / Zc), 20)

            x1 = max(0, u_c - w_px // 2)
            y1 = max(0, v_c - h_px // 2)
            x2 = min(W - 1, u_c + w_px // 2)
            y2 = min(H - 1, v_c + h_px // 2)

            if x2 > x1 and y2 > y1:
                depth_map[y1:y2, x1:x2] = Zc
                color = tuple((np.array(info["color"][:3]) * 255).astype(int))
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=(40, 40, 40), width=2)
                draw.text((x1 + 4, y1 + 4), name.split()[0], fill=(255, 255, 255))

    return img_rgb, depth_map, K, T_world_cam

def simulate_lidar(scene, base_pos, base_yaw_deg, robot_shapes, num_rays=48, fov_deg=260.0, max_range=12.0):
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

def execute_vlm_navigation(instruction: str, model_path="Qwen/Qwen3-VL-4B-Instruct"):
    print(f"\n" + "="*70)
    print(f"🤖 启动 R2 机器人 VLM 具身导航任务")
    print(f"💬 自然语言指令: 「{instruction}」")
    print(f"="*70 + "\n")

    scene, robot, wl_j, wr_j, base, robot_shapes, targets_info, obstacles_info = build_embodied_scene()
    p_init = base.get_entity_pose().p
    yaw_init = euler(base.get_entity_pose().q)[2]

    # 1. 头部 RGB-D 相机拍摄
    print("[Perception] 正在通过头部相机采集当前环境的 RGB 彩色图与 Depth 深度矩阵...")
    img_rgb, depth_map, K, T_world_cam = capture_head_camera_rgbd(p_init, yaw_init, targets_info, obstacles_info)

    # 2. Qwen-VL 视觉定位与语义解析
    print(f"[VLM] 激活 Qwen-VL 视觉语言大模型执行 Visual Grounding 与空间意图推断...")
    vlm = VLMTargetDetector(model_id_or_path=model_path, device="cuda:0")
    vlm_result = vlm.detect(img_rgb, instruction)
    
    target_name = vlm_result["target_name"]
    bbox_norm = vlm_result["bbox_norm"]
    spatial_relation = vlm_result["spatial_relation"]
    print(f"[VLM] ✅ 识别完成:")
    print(f"      - 检测目标: {target_name}")
    print(f"      - 边界框 (0~1000): {bbox_norm}")
    print(f"      - 空间方位意图: {spatial_relation} (前面/旁边/附近)")

    # 3. 深度反投影计算 3D 世界坐标与停靠航路点
    u_c, v_c, median_depth, pixel_bbox = compute_robust_bbox_center_depth(depth_map, bbox_norm)
    obj_world_pos = backproject_pixel_to_3d(u_c, v_c, median_depth, K, T_world_cam)
    goal_x, goal_y, theta_goal_deg = compute_standoff_waypoint(
        obj_world_pos, p_init, spatial_relation=spatial_relation, standoff_dist=0.75
    )

    print(f"\n[Depth Projector] 深度反投影与 3D 航路点计算结果:")
    print(f"      - 目标中心像素: ({u_c}, {v_c}), 中值深度: {median_depth:.2f} m")
    print(f"      - 物体 3D 物理空间坐标: ({obj_world_pos[0]:.2f}, {obj_world_pos[1]:.2f}, {obj_world_pos[2]:.2f}) m")
    print(f"      - 生成规划停靠点: (x={goal_x:.2f}, y={goal_y:.2f}) m, 期望对准朝向={theta_goal_deg:+.1f}°\n")

    # 4. 激光雷达局部避障与底盘导航闭环
    final_goal = (goal_x, goal_y)
    current_subgoal = final_goal
    bypassing = False
    trajectory = []
    min_obstacle_dist = float('inf')

    print("[Navigation] 启动 260° 激光雷达与底盘差速闭环导航，开始执行路径行驶与避障...")
    for step in range(5000):
        pos = base.get_entity_pose().p
        yaw = euler(base.get_entity_pose().q)[2]
        trajectory.append((pos[0], pos[1]))

        # 激光雷达扫描
        ranges, ray_dirs, rel_angles = simulate_lidar(scene, pos, yaw, robot_shapes)
        cur_min_range = np.min(ranges)
        if cur_min_range < min_obstacle_dist:
            min_obstacle_dist = cur_min_range

        # 计算距最终停靠点的距离
        dist_to_final = np.sqrt((pos[0]-final_goal[0])**2 + (pos[1]-final_goal[1])**2)
        if dist_to_final < 0.12:
            print(f"\n🎯 成功精准到达指定停靠点! 总步数={step}, 终点定位误差={dist_to_final:.3f}m")
            break

        # 前方障碍物检测与绕障 (仅在远离终点 > 0.8m 时触发沿途绕障，避免误避目标物体本身)
        fwd_mask = (np.abs(rel_angles) < 28)
        fwd_min_dist = np.min(ranges[fwd_mask])

        if fwd_min_dist < 1.15 and not bypassing and dist_to_final > 0.85:
            obs_x = pos[0] + np.cos(np.radians(yaw)) * fwd_min_dist
            obs_y = pos[1] + np.sin(np.radians(yaw)) * fwd_min_dist
            
            # 选择朝向目标且开阔的一侧绕行
            detour_y = obs_y + SAFE_RADIUS if final_goal[1] >= obs_y else obs_y - SAFE_RADIUS
            current_subgoal = (obs_x + 0.5, detour_y)
            bypassing = True
            print(f"[Obstacle Avoidance] 前方 {fwd_min_dist:.2f}m 探测到障碍物，生成侧向绕行点: ({current_subgoal[0]:.2f}, {current_subgoal[1]:.2f})")

        if bypassing:
            dist_to_subgoal = np.sqrt((pos[0]-current_subgoal[0])**2 + (pos[1]-current_subgoal[1])**2)
            if dist_to_subgoal < 0.28 or (pos[0] > current_subgoal[0] and fwd_min_dist > 1.25) or dist_to_final <= 0.85:
                bypassing = False
                current_subgoal = final_goal
                print(f"[Obstacle Avoidance] 成功绕开障碍物，重新切回目标停靠点: ({final_goal[0]:.2f}, {final_goal[1]:.2f})")

        dx = current_subgoal[0] - pos[0]
        dy = current_subgoal[1] - pos[1]
        dist_to_target = np.sqrt(dx**2 + dy**2)
        target_angle_deg = np.degrees(np.arctan2(dy, dx))
        heading_err = (target_angle_deg - yaw + 180) % 360 - 180

        omega_cmd = np.clip(2.5 * np.radians(heading_err), -MAX_W, MAX_W)
        align = max(0.0, np.cos(np.radians(heading_err)))
        v_cmd = MAX_V * (align ** 2) * np.clip(dist_to_target / 0.4, 0.30, 1.0)

        v_l, v_r = diff_drive(v_cmd, omega_cmd)
        wl_j.set_drive_velocity_target(v_l)
        wr_j.set_drive_velocity_target(v_r)
        scene.step()

        if step % 250 == 0:
            print(f"[Step {step:4d}] 坐标=({pos[0]:+.2f}, {pos[1]:+.2f}) 距停靠点={dist_to_final:.2f}m 航向={yaw:+.1f}° v={v_cmd:.2f}m/s")

    # 5. 最终朝向对齐微调
    print("[Alignment] 正在微调最终朝向，正对目标物体中心...")
    for _ in range(150):
        yaw = euler(base.get_entity_pose().q)[2]
        yaw_err = (theta_goal_deg - yaw + 180) % 360 - 180
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
    final_pos_err = np.sqrt((final_pos[0]-final_goal[0])**2 + (final_pos[1]-final_goal[1])**2)

    # 6. 生成可视化报告图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    img_draw = img_rgb.copy()
    d_draw = ImageDraw.Draw(img_draw)
    x1, y1, x2, y2 = pixel_bbox
    d_draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    d_draw.text((x1+5, y1+5), f"{target_name} ({spatial_relation})", fill="red")
    axes[0].imshow(img_draw)
    axes[0].set_title(f"1. VLM Visual Grounding (RGB)\nInstruction: '{instruction}'")
    axes[0].axis("off")

    depth_vis = np.clip(depth_map, 0.0, 6.0)
    im_d = axes[1].imshow(depth_vis, cmap="viridis")
    axes[1].plot(u_c, v_c, "r*", markersize=14, label=f"Center (Z={median_depth:.2f}m)")
    axes[1].legend(loc="lower right")
    axes[1].set_title("2. Aligned Depth Buffer & 3D Center")
    axes[1].axis("off")
    plt.colorbar(im_d, ax=axes[1], fraction=0.046, pad=0.04)

    traj_arr = np.array(trajectory)
    axes[2].plot(traj_arr[:, 0], traj_arr[:, 1], "b-", linewidth=2.5, label="R2 Driven Trajectory")
    axes[2].plot(p_init[0], p_init[1], "go", markersize=9, label="Start (0,0)")
    axes[2].plot(goal_x, goal_y, "r*", markersize=12, label=f"Standoff Goal ({goal_x:.2f},{goal_y:.2f})")
    
    for name, info in targets_info.items():
        ox, oy, _ = info["pos"]
        sx, sy, _ = info["size"]
        rect = plt.Rectangle((ox - sx/2, oy - sy/2), sx, sy, color="blue", alpha=0.3)
        axes[2].add_patch(rect)
        axes[2].text(ox, oy, name.split()[0], ha="center", va="center", fontsize=9, fontweight="bold")

    for i, info in enumerate(obstacles_info):
        ox, oy, _ = info["pos"]
        sx, sy, _ = info["size"]
        rect = plt.Rectangle((ox - sx/2, oy - sy/2), sx, sy, color="red", alpha=0.5)
        axes[2].add_patch(rect)
        axes[2].text(ox, oy, f"Obs {i+1}", ha="center", va="center", color="white", fontsize=8)

    axes[2].set_xlabel("X (meters)")
    axes[2].set_ylabel("Y (meters)")
    axes[2].set_title("3. BEV Navigation & Obstacle Avoidance Trajectory")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="upper left")
    axes[2].set_xlim(-0.5, 5.0)
    axes[2].set_ylim(-2.0, 2.0)
    axes[2].set_aspect("equal")

    out_img_path = "/home/xujinlong/test/output/vlm_nav_result.png"
    plt.tight_layout()
    plt.savefig(out_img_path, dpi=150)
    plt.close()

    print(f"\n==================== VLM 导航评测汇总报告 ====================")
    print(f"💬 自然语言指令: 「{instruction}」")
    print(f"🎯 VLM 识别目标: {target_name} | 方位意图: {spatial_relation}")
    print(f"📍 物体 3D 真实中心: ({obj_world_pos[0]:.3f}, {obj_world_pos[1]:.3f}) m")
    print(f"🚩 规划停靠目标点: ({goal_x:.3f}, {goal_y:.3f}) m | 期望朝向: {theta_goal_deg:+.1f}°")
    print(f"🏁 机器人最终停止位姿: ({final_pos[0]:.3f}, {final_pos[1]:.3f}) m | 实际朝向: {final_yaw:+.1f}°")
    print(f"📏 最终停靠位置误差: {final_pos_err:.4f} m (阈值 < 0.15m: {'✅ 成功达标' if final_pos_err < 0.15 else '❌ 未达标'})")
    print(f"🛡️ 全程最近障碍物距离: {min_obstacle_dist:.3f} m (安全阈值 > 0.30m: {'✅ 安全无碰撞' if min_obstacle_dist > 0.30 else '❌ 发生干涉'})")
    print(f"🖼️ 可视化多模态报告图已保存至: {out_img_path}")
    print(f"==============================================================\n")
    return final_pos_err, min_obstacle_dist, out_img_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", type=str, default="去桌子旁边", help="自然语言导航指令")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-4B-Instruct", help="模型路径或名称")
    args = parser.parse_args()
    execute_vlm_navigation(args.instruction, model_path=args.model)
