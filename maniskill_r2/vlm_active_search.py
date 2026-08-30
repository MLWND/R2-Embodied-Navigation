"""360° 原地自旋全景扫描与主动视觉搜索机制 (Visual Frontier & 360° Active Search)"""
import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sapien

from depth_projector import compute_robust_bbox_center_depth, backproject_pixel_to_3d, compute_standoff_waypoint, fit_bbox_3d_center_from_depth_patch
from vlm_detector import VLMTargetDetector

def euler_to_quat(yaw_deg, pitch_deg):
    from scipy.spatial.transform import Rotation as R
    r = R.from_euler('zyx', [yaw_deg, -pitch_deg, 0], degrees=True)
    q = r.as_quat()
    return [q[3], q[0], q[1], q[2]]

def capture_panoramic_views(scene, robot_base_pos, robot_base_yaw_deg, targets_info, obstacles_info, num_views=8):
    angles = np.linspace(0, 360, num_views, endpoint=False)
    views = []
    W, H = 640, 480
    fov_rad = np.radians(75.0)
    fx = (W / 2.0) / np.tan(fov_rad / 2.0)
    fy = fx
    cx = W / 2.0
    cy = H / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    cam_pos = np.array([robot_base_pos[0], robot_base_pos[1], 1.35], dtype=np.float32)
    pitch_rad = np.radians(-8.0)
    
    if scene is not None:
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, -1, -1], [1, 1, 1])
        cam = scene.add_camera('panoramic_cam', W, H, fov_rad, 0.01, 50.0)

    for ang in angles:
        current_yaw_deg = (robot_base_yaw_deg + ang) % 360
        yaw_rad = np.radians(current_yaw_deg)
        
        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
        cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
        
        fwd = np.array([cos_y * cos_p, sin_y * cos_p, sin_p])
        right = np.array([sin_y, -cos_y, 0.0])
        up = np.cross(fwd, right)
        
        R_cam_world = np.stack([right, -up, fwd], axis=1)
        T_world_cam = np.eye(4, dtype=np.float32)
        T_world_cam[:3, :3] = R_cam_world
        T_world_cam[:3, 3] = cam_pos

        if scene is not None:
            pitch_deg = np.degrees(pitch_rad)
            cam.set_local_pose(sapien.Pose(cam_pos, euler_to_quat(current_yaw_deg, pitch_deg)))
            scene.update_render()
            cam.take_picture()
            
            rgba = cam.get_picture('Color')
            img_rgb = Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8))
            
            pos_img = cam.get_picture('Position')
            depth_map = -pos_img[:, :, 2]
            
            K = cam.get_intrinsic_matrix()
            T_world_cam = cam.get_model_matrix() @ np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
            draw = None
        else:
            depth_map = np.full((H, W), 10.0, dtype=np.float32)
            img_rgb = Image.new("RGB", (W, H), color=(235, 238, 242))
            draw = ImageDraw.Draw(img_rgb)
            draw.rectangle([0, int(H*0.4), W, H], fill=(210, 215, 220))

        all_objects = list(targets_info.items()) + [(f"Obstacle_{i+1}", obs) for i, obs in enumerate(obstacles_info)]
        sorted_objs = sorted(all_objects, key=lambda item: -np.linalg.norm(np.array(item[1]["pos"]) - cam_pos))

        visible_objects = []
        for name, info in sorted_objs:
            ox, oy, oz = info["pos"]
            sx, sy, sz = info["size"]
            P_w = np.array([ox, oy, oz, 1.0])
            P_c = np.linalg.inv(T_world_cam) @ P_w
            Xc, Yc, Zc = P_c[:3]

            if Zc > 0.3:
                u_c = int(K[0, 2] + K[0, 0] * Xc / Zc)
                v_c = int(K[1, 2] + K[1, 1] * Yc / Zc)
                w_px = max(int(K[0, 0] * sy / Zc), 15)
                h_px = max(int(K[1, 1] * sz / Zc), 20)

                x1 = max(0, u_c - w_px // 2)
                y1 = max(0, v_c - h_px // 2)
                x2 = min(W - 1, u_c + w_px // 2)
                y2 = min(H - 1, v_c + h_px // 2)

                if x2 > x1 and y2 > y1 and (x2 - x1) * (y2 - y1) > 200:
                    if scene is None:
                        depth_map[y1:y2, x1:x2] = Zc
                        color = tuple((np.array(info["color"][:3]) * 255).astype(int))
                        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(40, 40, 40), width=2)
                        draw.text((x1 + 4, y1 + 4), name.split()[0], fill=(255, 255, 255))
                    visible_objects.append(name.split()[0].lower())

        views.append({
            "angle_deg": current_yaw_deg,
            "img_rgb": img_rgb,
            "depth_map": depth_map,
            "K": K,
            "T_world_cam": T_world_cam,
            "visible_objects": visible_objects
        })

    if scene is not None:
        scene.remove_camera(cam)

    return views

def run_active_360_search(instruction: str, targets_info, obstacles_info, robot_pos=[0,0,0], robot_yaw=0.0):
    print(f"\n" + "="*70)
    print(f"🔄 启动 360° 原地自旋主动视觉搜索机制 (Active Visual Search)")
    print(f"💬 自然语言指令: 「{instruction}」")
    print(f"="*70 + "\n")

    views = capture_panoramic_views(None, robot_pos, robot_yaw, targets_info, obstacles_info, num_views=8)
    detector = VLMTargetDetector(device="cuda:0")
    
    best_view = None
    best_detection = None
    best_depth = None
    best_u_c, best_v_c, best_pixel_bbox = None, None, None

    query_kw = instruction.lower()
    expected_tokens = []
    if "床" in query_kw or "bed" in query_kw: expected_tokens.extend(["bed", "床"])
    elif "冰" in query_kw or "fridge" in query_kw: expected_tokens.extend(["fridge", "refrigerator", "冰"])
    elif "桌" in query_kw or "table" in query_kw or "desk" in query_kw: expected_tokens.extend(["table", "desk", "桌"])
    elif "沙发" in query_kw or "sofa" in query_kw: expected_tokens.extend(["sofa", "couch", "沙发"])

    for i, v in enumerate(views):
        res = detector.detect(v["img_rgb"], instruction)
        target_name = res["target_name"].lower()
        u_c, v_c, median_d, pixel_bbox = compute_robust_bbox_center_depth(v["depth_map"], res["bbox_norm"])
        
        is_target_matched = any(tok in target_name for tok in expected_tokens)
        is_visible = any(tok in obj for tok in expected_tokens for obj in v["visible_objects"])
        
        print(f"  - 视角 {i+1} (朝向 {v['angle_deg']:5.1f}°): 检测到 '{res['target_name']}', 前景深度={median_d:.2f}m, 目标锁定={'✅ 命中目标' if (is_target_matched and is_visible) else '❌ 未见'}")

        if is_target_matched and is_visible and best_view is None:
            best_view = v
            best_detection = res
            best_u_c, best_v_c, best_depth, best_pixel_bbox = u_c, v_c, median_d, pixel_bbox

    if best_view is None:
        best_view = views[0]
        best_detection = detector.detect(best_view["img_rgb"], instruction)
        best_u_c, best_v_c, best_depth, best_pixel_bbox = compute_robust_bbox_center_depth(best_view["depth_map"], best_detection["bbox_norm"])

    print(f"\n[Active Search] 🎯 成功锁定最优观测视角:")
    print(f"      - 锁定朝向: {best_view['angle_deg']:.1f}°")
    print(f"      - 目标名称: {best_detection['target_name']}")
    print(f"      - 方位语义: {best_detection['spatial_relation']}")
    print(f"      - 测距深度: {best_depth:.2f} m")

    obj_world_pos, best_u_c, best_v_c, best_depth = fit_bbox_3d_center_from_depth_patch(
        best_view["depth_map"], best_detection["bbox_norm"], best_view["K"], best_view["T_world_cam"], category=best_detection["target_name"]
    )
    goal_x, goal_y, theta_goal = compute_standoff_waypoint(
        obj_world_pos, robot_pos, spatial_relation=best_detection["spatial_relation"], standoff_dist=0.75
    )

    print(f"[3D Waypoint] 目标物体 3D 物理坐标: ({obj_world_pos[0]:.2f}, {obj_world_pos[1]:.2f}, {obj_world_pos[2]:.2f}) m")
    print(f"[3D Waypoint] 规划停靠目标点: (x={goal_x:.2f}, y={goal_y:.2f}) m, 期望对准朝向={theta_goal:+.1f}°\n")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for idx, ax in enumerate(axes.flatten()):
        v = views[idx]
        img_show = v["img_rgb"].copy()
        if v == best_view:
            d_d = ImageDraw.Draw(img_show)
            bx1, by1, bx2, by2 = best_pixel_bbox
            d_d.rectangle([bx1, by1, bx2, by2], outline="red", width=3)
            d_d.text((bx1+5, by1+5), "TARGET LOCKED", fill="red")
            ax.set_title(f"View {idx+1}: {v['angle_deg']:.0f} deg [LOCKED]", color="red", fontweight="bold")
        else:
            ax.set_title(f"View {idx+1}: {v['angle_deg']:.0f} deg")
        ax.imshow(img_show)
        ax.axis("off")

    out_img = "/home/xujinlong/test/output/vlm_360_active_search.png"
    plt.suptitle(f"360-Degree Active Visual Search: '{instruction}'", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_img, dpi=150)
    plt.close()
    print(f"🖼️ 8 视角全景搜索合成图已保存至: {out_img}\n")

    return goal_x, goal_y, theta_goal, out_img

if __name__ == "__main__":
    targets_info = {
        "Table (桌子)": {"pos": [3.2, 0.9, 0.38], "size": [0.8, 1.2, 0.75], "color": [0.7, 0.45, 0.2, 1.0]},
        "Fridge (冰箱)": {"pos": [2.8, -1.3, 0.80], "size": [0.65, 0.65, 1.60], "color": [0.85, 0.85, 0.9, 1.0]},
        "Sofa (沙发)": {"pos": [4.2, -0.2, 0.40], "size": [0.9, 1.8, 0.80], "color": [0.3, 0.5, 0.7, 1.0]},
        "Bed (卧室大床)": {"pos": [-2.5, 2.8, 0.45], "size": [1.8, 2.0, 0.80], "color": [0.4, 0.6, 0.5, 1.0]}, # 在背后盲区!
    }
    obstacles_info = [
        {"pos": [1.4, 0.0, 0.50], "size": [0.4, 0.4, 1.0], "color": [0.8, 0.2, 0.2, 1.0]},
    ]
    run_active_360_search("去卧室大床旁边", targets_info, obstacles_info, robot_pos=[0,0,0], robot_yaw=0.0)
