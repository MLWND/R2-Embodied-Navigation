"""R2 轮式人形机器人: 自主导航到指定目标点 + 静态障碍物智能绕行避障 (多障碍物 Subgoal / Bypass 避障导航系统)"""
import os
import numpy as np
import sapien

URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_swivel.urdf"
MESH = "/home/xujinlong/test/robot_example1_EN_demo/robot_example1_EN_demo/robot_model/r2_urdf_v01/meshes"
TMP = "/tmp/r2_swivel_tmp.urdf"

# 机器人差速动力学参数
WHEEL_R = 0.085    # 驱动轮半径 (m)
TRACK = 0.458      # 轮距 (m)
MAX_V = 0.50       # 最大前进线速度 (m/s)
MAX_W = 1.5        # 最大旋转角速度 (rad/s)
SAFE_RADIUS = 0.85 # 绕障安全避让半径 (m)

def euler(q):
    w, x, y, z = q
    roll = np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y-x*z), -1, 1)))
    yaw = np.degrees(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    return roll, pitch, yaw

def build_nav_scene(obstacles):
    with open(URDF) as f:
        urdf = f.read()
    urdf = urdf.replace("package://r2_urdf_v01/meshes/", MESH + "/")
    with open(TMP, "w") as f:
        f.write(urdf)

    scene = sapien.Scene()
    scene.set_timestep(0.005)
    scene.add_ground(altitude=0, render=False)

    for i, (ox, oy, size_x, size_y, size_z) in enumerate(obstacles):
        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=[size_x/2, size_y/2, size_z/2])
        builder.add_box_visual(half_size=[size_x/2, size_y/2, size_z/2])
        actor = builder.build_static(name=f"obstacle_{i}")
        actor.set_pose(sapien.Pose(p=[ox, oy, size_z/2]))

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

    return scene, robot, wl_j, wr_j, base, robot_shapes

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

def run_navigation(scene, wl_j, wr_j, base, robot_shapes, final_goal, obstacles):
    min_obstacle_dist = float('inf')
    p_init = base.get_entity_pose().p
    print(f"\n=======================================================")
    print(f"🚀 开始自主导航与避障验证 (多障碍物绕行场景):")
    print(f"   起点坐标: ({p_init[0]:.2f}, {p_init[1]:.2f})")
    print(f"   最终目标点: ({final_goal[0]:.2f}, {final_goal[1]:.2f})")
    print(f"   静态障碍物数量: {len(obstacles)} 个")
    for idx, (ox, oy, sx, sy, sz) in enumerate(obstacles):
        print(f"     障碍物 {idx+1}: 中心=({ox:.2f}, {oy:.2f}), 尺寸=({sx:.2f}x{sy:.2f}x{sz:.2f})m")
    print(f"=======================================================\n")

    current_subgoal = final_goal
    bypassing = False

    for step in range(6000):
        pos = base.get_entity_pose().p
        yaw = euler(base.get_entity_pose().q)[2]

        # 1. 激光雷达扫描
        ranges, ray_dirs, rel_angles = simulate_lidar(scene, pos, yaw, robot_shapes)
        cur_min_range = np.min(ranges)
        if cur_min_range < min_obstacle_dist:
            min_obstacle_dist = cur_min_range

        # 2. 前方障碍物检测
        fwd_mask = (np.abs(rel_angles) < 28)
        fwd_min_dist = np.min(ranges[fwd_mask])

        if fwd_min_dist < 1.15 and not bypassing:
            obs_x = pos[0] + np.cos(np.radians(yaw)) * fwd_min_dist
            obs_y = pos[1] + np.sin(np.radians(yaw)) * fwd_min_dist
            
            # 动态选择左绕或右绕
            left_mask = (rel_angles > 15) & (rel_angles < 60)
            right_mask = (rel_angles < -15) & (rel_angles > -60)
            left_clear = np.mean(ranges[left_mask]) if np.any(left_mask) else 0.0
            right_clear = np.mean(ranges[right_mask]) if np.any(right_mask) else 0.0
            
            detour_y = obs_y + SAFE_RADIUS if left_clear >= right_clear else obs_y - SAFE_RADIUS
            current_subgoal = (obs_x + 0.5, detour_y)
            bypassing = True
            print(f"[Plan] 前方 {fwd_min_dist:.2f}m 探测到障碍物, 动态生成侧向绕行目标点: ({current_subgoal[0]:.2f}, {current_subgoal[1]:.2f})")

        if bypassing:
            dist_to_subgoal = np.sqrt((pos[0]-current_subgoal[0])**2 + (pos[1]-current_subgoal[1])**2)
            if dist_to_subgoal < 0.28 or (pos[0] > current_subgoal[0] and fwd_min_dist > 1.25):
                bypassing = False
                current_subgoal = final_goal
                print(f"[Plan] 障碍物绕行完成! 重新对准目标终点: ({final_goal[0]:.2f}, {final_goal[1]:.2f})")

        # 3. 目标航向与误差计算
        dx = current_subgoal[0] - pos[0]
        dy = current_subgoal[1] - pos[1]
        dist_to_target = np.sqrt(dx**2 + dy**2)
        target_angle_deg = np.degrees(np.arctan2(dy, dx))
        heading_err = (target_angle_deg - yaw + 180) % 360 - 180

        # 到达最终目标判定
        dist_to_final = np.sqrt((pos[0]-final_goal[0])**2 + (pos[1]-final_goal[1])**2)
        if dist_to_final < 0.10:
            print(f"\n🎯 成功精准到达指定目标点! 耗费总步数={step}, 最终定位误差={dist_to_final:.3f}m")
            break

        # 4. 闭环控制
        omega_cmd = np.clip(2.5 * np.radians(heading_err), -MAX_W, MAX_W)
        align = max(0.0, np.cos(np.radians(heading_err)))
        v_cmd = MAX_V * (align ** 2) * np.clip(dist_to_target / 0.4, 0.35, 1.0)

        # 5. 底盘差速执行
        v_l, v_r = diff_drive(v_cmd, omega_cmd)
        wl_j.set_drive_velocity_target(v_l)
        wr_j.set_drive_velocity_target(v_r)
        scene.step()

        if step % 250 == 0:
            print(f"[Step {step:4d}] 坐标=({pos[0]:+.2f}, {pos[1]:+.2f}) 距终点={dist_to_final:.2f}m 最近障碍={cur_min_range:.2f}m 航向={yaw:+.1f}° Δyaw={heading_err:+.1f}° v={v_cmd:.2f}m/s")

    # 停车刹车
    wl_j.set_drive_velocity_target(0.0)
    wr_j.set_drive_velocity_target(0.0)
    for _ in range(50):
        scene.step()

    final_pos = base.get_entity_pose().p
    final_err = np.sqrt((final_pos[0]-final_goal[0])**2 + (final_pos[1]-final_goal[1])**2)
    print(f"\n================ 导航避障评测结果 ================")
    print(f"🏁 最终停止位置: ({final_pos[0]:.3f}, {final_pos[1]:.3f})")
    print(f"🎯 指定目标位置: ({final_goal[0]:.3f}, {final_goal[1]:.3f})")
    print(f"📏 最终定位误差: {final_err:.4f} m (目标阈值 < 0.15m: {'✅ 达标' if final_err < 0.15 else '❌ 未达标'})")
    print(f"🛡️ 全程最近障碍物距离: {min_obstacle_dist:.3f} m (安全阈值 > 0.35m: {'✅ 安全无碰撞' if min_obstacle_dist > 0.35 else '❌ 发生干涉'})")
    print(f"==================================================\n")
    return final_err, min_obstacle_dist

if __name__ == "__main__":
    # 在直线必经之路上放置 1.3m 处障碍物 (模拟立柱)
    obstacles = [
        (1.3,  0.0, 0.4, 0.4, 1.0),   # 挡在正前方 x=1.3 处的立柱
    ]
    # 目标点在障碍物后方 3.0 米处
    goal = (3.0, 0.0)

    scene, robot, wl_j, wr_j, base, robot_shapes = build_nav_scene(obstacles)
    run_navigation(scene, wl_j, wr_j, base, robot_shapes, goal, obstacles)
