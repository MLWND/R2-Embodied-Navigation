"""RGB-D 深度反投影与 3D 导航停靠航路点生成器 (Depth Projector & Standoff Waypoint Generator)"""
import numpy as np

def compute_robust_bbox_center_depth(depth_map, bbox_coords):
    """从 2D Bounding Box 和深度图中提取稳健的物体中心像素与中值深度
    
    自动自适应解析 [x1, y1, x2, y2] 或 [y1, x1, y2, x2] 归一化坐标格式 (0~1000)
    """
    H, W = depth_map.shape
    c1, c2, c3, c4 = bbox_coords
    
    # 判定坐标格式: 候选 A 为 [x1, y1, x2, y2], 候选 B 为 [y1, x1, y2, x2]
    # 计算两种排列下对应深度图的前景命中质量
    candA_x1, candA_y1, candA_x2, candA_y2 = int(c1*W/1000), int(c2*H/1000), int(c3*W/1000), int(c4*H/1000)
    candB_x1, candB_y1, candB_x2, candB_y2 = int(c2*W/1000), int(c1*H/1000), int(c4*W/1000), int(c3*H/1000)
    
    def eval_patch(x1, y1, x2, y2):
        x1, x2 = max(0, min(x1, x2)), min(W-1, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(H-1, max(y1, y2))
        if x2 <= x1 or y2 <= y1: return float('inf'), (x1, y1, x2, y2), int((x1+x2)/2), int((y1+y2)/2)
        patch = depth_map[y1:y2, x1:x2]
        valid = (patch > 0.3) & (patch < 8.5) & (~np.isnan(patch))
        if np.any(valid):
            return float(np.median(patch[valid])), (x1, y1, x2, y2), int((x1+x2)/2), int((y1+y2)/2)
        return float('inf'), (x1, y1, x2, y2), int((x1+x2)/2), int((y1+y2)/2)

    dA, boxA, uA, vA = eval_patch(candA_x1, candA_y1, candA_x2, candA_y2)
    dB, boxB, uB, vB = eval_patch(candB_x1, candB_y1, candB_x2, candB_y2)
    
    if dA < 8.5 and (dB >= 8.5 or dA <= dB):
        return uA, vA, dA, boxA
    elif dB < 8.5:
        return uB, vB, dB, boxB
    else:
        # Fallback to candA
        return uA, vA, (dA if dA < 20 else 4.0), boxA

def backproject_pixel_to_3d(u, v, depth, K, T_world_cam):
    """将单个像素 (u, v) 与深度值反投影到世界三维空间坐标系 (Xw, Yw, Zw)"""
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    X_c = (u - cx) * depth / fx
    Y_c = (v - cy) * depth / fy
    Z_c = depth
    
    P_cam = np.array([X_c, Y_c, Z_c, 1.0], dtype=np.float32)
    P_world = (T_world_cam @ P_cam)[:3]
    return P_world

def fit_bbox_3d_center_from_depth_patch(depth_map, bbox_coords, K, T_world_cam, category="general"):
    """基于 ROI 点云聚类与 3D 几何包围盒拟合，从局部可见侧面反推真实 3D 几何体中心
    
    解决超宽家具 (如 L 型沙发、电视柜、大床) 可见侧表面中心偏离真实几何体中心的问题
    """
    H, W = depth_map.shape
    c1, c2, c3, c4 = bbox_coords
    candA_x1, candA_y1, candA_x2, candA_y2 = int(c1*W/1000), int(c2*H/1000), int(c3*W/1000), int(c4*H/1000)
    candB_x1, candB_y1, candB_x2, candB_y2 = int(c2*W/1000), int(c1*H/1000), int(c4*W/1000), int(c3*H/1000)
    
    def extract_cluster(x1, y1, x2, y2):
        x1, x2 = max(0, min(x1, x2)), min(W-1, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(H-1, max(y1, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return np.zeros((0, 3), dtype=np.float32), 0, 0, 0.0
        u_grid, v_grid = np.meshgrid(np.arange(x1, x2, 4), np.arange(y1, y2, 4))
        d_vals = depth_map[v_grid, u_grid]
        valid = (d_vals > 0.3) & (d_vals < 8.5) & (~np.isnan(d_vals))
        if not np.any(valid):
            return np.zeros((0, 3), dtype=np.float32), 0, 0, 0.0
            
        u_val, v_val, d_val = u_grid[valid], v_grid[valid], d_vals[valid]
        med_d = float(np.median(d_val))
        # 聚类前景点，去除背景墙面深度
        fg_mask = np.abs(d_val - med_d) < 0.60
        if not np.any(fg_mask):
            fg_mask = np.ones_like(d_val, dtype=bool)
            
        u_fg, v_fg, d_fg = u_val[fg_mask], v_val[fg_mask], d_val[fg_mask]
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        Xc = (u_fg - cx) * d_fg / fx
        Yc = (v_fg - cy) * d_fg / fy
        Zc = d_fg
        pts_c = np.stack([Xc, Yc, Zc, np.ones_like(Zc)], axis=1)
        pts_w = (T_world_cam @ pts_c.T).T[:, :3]
        return pts_w, int(np.median(u_fg)), int(np.median(v_fg)), med_d

    ptsA, uA, vA, dA = extract_cluster(candA_x1, candA_y1, candA_x2, candA_y2)
    ptsB, uB, vB, dB = extract_cluster(candB_x1, candB_y1, candB_x2, candB_y2)
    
    if len(ptsA) >= len(ptsB) and len(ptsA) > 0:
        pts, u_c, v_c, d_c = ptsA, uA, vA, dA
    elif len(ptsB) > 0:
        pts, u_c, v_c, d_c = ptsB, uB, vB, dB
    else:
        u_c, v_c, d_c, _ = compute_robust_bbox_center_depth(depth_map, bbox_coords)
        return backproject_pixel_to_3d(u_c, v_c, d_c, K, T_world_cam), u_c, v_c, d_c

    p_min = np.min(pts, axis=0)
    p_max = np.max(pts, axis=0)
    p_med = np.median(pts, axis=0)
    aabb_center = (p_min + p_max) / 2.0
    cam_pos = T_world_cam[:3, 3]
    
    ray_xy = p_med[:2] - cam_pos[:2]
    norm_ray = ray_xy / max(np.linalg.norm(ray_xy), 1e-4)
    proj_depth = np.dot(pts[:, :2] - cam_pos[:2], norm_ray)
    depth_span = float(np.max(proj_depth) - np.min(proj_depth))
    
    cat = str(category).lower()
    if depth_span > 0.35:
        # 点云已覆盖物体水平顶面/纵深，直接使用 3D AABB 几何中心
        body_center = np.array([aabb_center[0], aabb_center[1], p_med[2]], dtype=np.float32)
    else:
        # 仅观测到单一垂直表面 (如沙发前侧板或柜门)，根据家具类别向内法向推算体中心
        if any(k in cat for k in ["couch", "sofa", "沙发"]):
            d_est = 0.80
        elif any(k in cat for k in ["bed", "床"]):
            d_est = 1.00
        elif any(k in cat for k in ["table", "desk", "桌"]):
            d_est = 0.60
        elif any(k in cat for k in ["cabinet", "wardrobe", "柜"]):
            d_est = 0.40
        else:
            d_est = 0.25
        body_center = np.copy(p_med)
        body_center[:2] += norm_ray * (d_est * 0.5)

    return body_center, u_c, v_c, d_c

def compute_standoff_waypoint(obj_world_pos, robot_world_pos, spatial_relation="front", standoff_dist=0.75, obj_half_extents=None):
    """根据目标物体 3D 坐标与用户空间方位意图，计算机器人最终停靠目标点 (x_goal, y_goal, θ_goal)
    
    Args:
        obj_half_extents: 可选 (hx, hy) 物体 XY 方向半尺寸。若提供，会检查停靠点是否在物体内部并自动修正。
    """
    ox, oy = float(obj_world_pos[0]), float(obj_world_pos[1])
    rx, ry = float(robot_world_pos[0]), float(robot_world_pos[1])
    
    dx = rx - ox
    dy = ry - oy
    dist = np.sqrt(dx**2 + dy**2)
    if dist < 1e-4:
        v_los = np.array([1.0, 0.0])
    else:
        v_los = np.array([dx / dist, dy / dist])
        
    v_perp = np.array([-v_los[1], v_los[0]])
    
    rel = str(spatial_relation).lower().strip()
    if rel in ["front", "near", "前面", "附近", "周围"]:
        goal_xy = np.array([ox, oy]) + v_los * standoff_dist
    elif rel in ["side", "旁边", "侧面"]:
        goal_xy = np.array([ox, oy]) + (v_perp * standoff_dist if ry >= oy else -v_perp * standoff_dist)
    elif rel in ["left", "左侧", "左边"]:
        goal_xy = np.array([ox, oy]) - v_perp * standoff_dist
    elif rel in ["right", "右侧", "右边"]:
        goal_xy = np.array([ox, oy]) + v_perp * standoff_dist
    else:
        goal_xy = np.array([ox, oy]) + v_los * standoff_dist

    # 安全检查：若停靠点落入物体碰撞体内部，沿接近方向退避至表面外
    if obj_half_extents is not None:
        hx, hy = float(obj_half_extents[0]), float(obj_half_extents[1])
        margin = 0.10
        if abs(goal_xy[0] - ox) < (hx + margin) and abs(goal_xy[1] - oy) < (hy + margin):
            # 沿 v_los（从物体指向机器人）方向退避，确保停靠点在物体表面外
            surface_along_los = abs(v_los[0]) * hx + abs(v_los[1]) * hy
            goal_xy = np.array([ox, oy]) + v_los * (surface_along_los + standoff_dist)
        
    theta_goal_deg = float(np.degrees(np.arctan2(oy - goal_xy[1], ox - goal_xy[0])))
    return float(goal_xy[0]), float(goal_xy[1]), theta_goal_deg
