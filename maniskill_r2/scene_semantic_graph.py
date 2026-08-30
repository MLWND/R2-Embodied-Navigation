"""GRScenes-100 层次化场景语义拓扑图 (Scene Semantic Graph)

层次架构:
Scene (Home)
  ├── Room (LivingRoom, Kitchen, Bedroom, DiningRoom, Bathroom, Corridor...)
  │     ├── Bounds: [xmin, ymin, xmax, ymax]
  │     ├── Center Waypoint: [cx, cy]
  │     ├── Doorway / Portal: [dx, dy]
  │     └── Semantic Objects (Furniture, Appliances with 3D Bounding Box & Standoff Waypoints)
  └── Adjacency Graph: Room <-> Corridor <-> Room (Topological Routing)
"""
import os
import json
import heapq
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

try:
    from pxr import Usd, UsdGeom
except ImportError:
    pass

from depth_projector import compute_standoff_waypoint

def route_topological_path(targets_dict, start_xy, goal_xy):
    """基于 2D 实体占用栅格与 A* 搜索生成无碰撞拓扑航线通道点"""
    res = 0.05
    x_min, x_max = -12.0, 15.0
    y_min, y_max = -12.0, 12.0
    nx = int((x_max - x_min) / res)
    ny = int((y_max - y_min) / res)
    grid = np.zeros((nx, ny), dtype=bool)

    robot_r = 0.30
    for k, v in targets_dict.items():
        p, s = v['pos'], v['size']
        if len(p) >= 3 and len(s) >= 3:
            z_min = p[2] - s[2] / 2.0
            if z_min > 1.40:
                continue
        x0 = max(0, int((p[0] - s[0]/2 - robot_r - x_min) / res))
        x1 = min(nx, int((p[0] + s[0]/2 + robot_r - x_min) / res) + 1)
        y0 = max(0, int((p[1] - s[1]/2 - robot_r - y_min) / res))
        y1 = min(ny, int((p[1] + s[1]/2 + robot_r - y_min) / res) + 1)
        grid[x0:x1, y0:y1] = True

    sx, sy = int((start_xy[0] - x_min) / res), int((start_xy[1] - y_min) / res)
    gx, gy = int((goal_xy[0] - x_min) / res), int((goal_xy[1] - y_min) / res)

    def free_nearest(ix, iy):
        if 0 <= ix < nx and 0 <= iy < ny and not grid[ix, iy]:
            return ix, iy
        for r in range(1, 30):
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    if 0 <= ix+dx < nx and 0 <= iy+dy < ny and not grid[ix+dx, iy+dy]:
                        return ix+dx, iy+dy
        return ix, iy

    sx, sy = free_nearest(sx, sy)
    gx, gy = free_nearest(gx, gy)

    pq = [(0.0, 0.0, sx, sy, [(start_xy[0], start_xy[1])])]
    visited = np.zeros((nx, ny), dtype=bool)

    while pq:
        _, cost, cx, cy, path = heapq.heappop(pq)
        if visited[cx, cy]:
            continue
        visited[cx, cy] = True

        if abs(cx - gx) <= 2 and abs(cy - gy) <= 2:
            path.append((goal_xy[0], goal_xy[1]))
            def simplify_2d(pts, eps=0.15):
                if len(pts) <= 2:
                    return pts
                p1 = np.array(pts[0])
                p2 = np.array(pts[-1])
                dists = []
                for p in pts[1:-1]:
                    p = np.array(p)
                    if np.all(p1 == p2):
                        d = np.linalg.norm(p - p1)
                    else:
                        d = np.abs((p2[1]-p1[1])*p[0] - (p2[0]-p1[0])*p[1] + p2[0]*p1[1] - p2[1]*p1[0]) / (np.linalg.norm(p2-p1) + 1e-6)
                    dists.append(d)
                max_d = max(dists)
                max_idx = dists.index(max_d) + 1
                if max_d > eps:
                    left = simplify_2d(pts[:max_idx+1], eps)
                    right = simplify_2d(pts[max_idx:], eps)
                    return left[:-1] + right
                else:
                    return [pts[0], pts[-1]]
            res_pts = simplify_2d(path, eps=0.15)
            if np.linalg.norm(np.array(res_pts[-1]) - np.array(goal_xy)) > 0.01:
                res_pts.append((goal_xy[0], goal_xy[1]))
            return res_pts

        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
            nx_c, ny_c = cx + dx, cy + dy
            if 0 <= nx_c < nx and 0 <= ny_c < ny and not grid[nx_c, ny_c] and not visited[nx_c, ny_c]:
                step_cost = 1.414 if (dx != 0 and dy != 0) else 1.0
                heur = np.sqrt((nx_c - gx)**2 + (ny_c - gy)**2)
                new_cost = cost + step_cost * res
                heapq.heappush(pq, (new_cost + heur * res, new_cost, nx_c, ny_c, path + [(x_min + nx_c * res, y_min + ny_c * res)]))
    return None

# 中文语义映射表
CATEGORY_ZH_MAP = {
    "bed": "大床",
    "nightstand": "床头柜",
    "couch": "沙发",
    "sofa": "沙发",
    "teatable": "茶几",
    "table": "桌子",
    "chair": "椅子",
    "desk": "书桌",
    "shelf": "书架",
    "tv": "电视",
    "tvstand": "电视柜",
    "refrigerator": "冰箱",
    "microwave": "微波炉",
    "oven": "烤箱",
    "hearth": "灶台",
    "dishwasher": "洗碗机",
    "toilet": "马桶",
    "washingmachine": "洗衣机",
    "shoecabinet": "鞋柜",
    "sideboardcabinet": "餐边柜",
    "chestofdrawers": "斗柜",
    "door": "门",
    "faucet": "水龙头",
    "mirror": "镜子",
    "curtain": "窗帘",
}

ROOM_ZH_MAP = {
    "living_room": "客厅",
    "kitchen": "厨房",
    "dining_room": "餐厅",
    "master_bedroom": "主卧",
    "bedroom": "卧室",
    "bathroom": "卫生间",
    "entrance_hallway": "玄关走廊",
    "corridor": "过道走廊",
}

@dataclass
class SemanticObject:
    name: str
    category: str
    zh_name: str
    pos: List[float]  # [x, y, z] (meters)
    size: List[float]  # [sx, sy, sz] (meters)
    room_id: str = "unknown"

    def get_standoff_waypoint(self, robot_pos=[0.0, 0.0, 0.0], spatial_rel="front", standoff_dist=0.75):
        """计算物体外部的安全停靠航路点"""
        hx, hy = self.size[0] / 2.0, self.size[1] / 2.0
        gx, gy, theta = compute_standoff_waypoint(
            self.pos, robot_pos, spatial_relation=spatial_rel,
            standoff_dist=standoff_dist, obj_half_extents=[hx, hy]
        )
        return gx, gy, theta

@dataclass
class SemanticRoom:
    room_id: str
    room_type: str
    zh_name: str
    bounds: List[float]  # [xmin, ymin, xmax, ymax]
    center: List[float]  # [cx, cy]
    doorway: List[float] # [dx, dy] 连接走廊/外部的通道点
    objects: List[SemanticObject] = field(default_factory=list)

class SceneSemanticGraph:
    def __init__(self, scene_id: str):
        self.scene_id = scene_id
        self.rooms: Dict[str, SemanticRoom] = {}
        self.corridor_waypoints: List[List[float]] = []
        self.all_objects: List[SemanticObject] = []

SYNONYMS = {
    "table": ["餐桌", "桌子", "大餐桌", "书桌", "table", "dining table", "desk", "大桌"],
    "bed": ["床", "大床", "双人床", "主卧床", "次卧床", "bed"],
    "couch": ["沙发", "大沙发", "皮沙发", "couch", "sofa"],
    "sofa": ["沙发", "大沙发", "皮沙发", "couch", "sofa"],
    "refrigerator": ["冰箱", "双门冰箱", "冷柜", "fridge", "refrigerator"],
    "tv": ["电视", "电视机", "tv", "television"],
    "tvstand": ["电视柜", "tv stand", "tvstand"],
    "shelf": ["书架", "置物架", "展示架", "shelf", "bookshelf"],
    "chair": ["椅子", "餐椅", "办公椅", "chair"],
    "sideboardcabinet": ["餐边柜", "餐边", "sideboard", "sideboardcabinet"],
    "shoecabinet": ["鞋柜", "shoecabinet"],
    "cabinet": ["柜子", "储物柜", "cabinet"],
    "washingmachine": ["洗衣机", "washing machine", "washer"],
    "dishwasher": ["洗碗机", "dishwasher"],
    "microwave": ["微波炉", "microwave"],
    "oven": ["烤箱", "oven"],
    "toilet": ["马桶", "便池", "toilet"]
}

def _obj_matches_query(obj: SemanticObject, query: str) -> bool:
    q = query.lower()
    if obj.category in q or obj.zh_name in q or q in obj.zh_name:
        return True
    syns = SYNONYMS.get(obj.category, [obj.category, obj.zh_name])
    for s in syns:
        if s in q or q in s:
            return True
    return False

class SceneSemanticGraph:
    """全场景层次化语义拓扑图 (Scene Semantic Graph)"""
    def __init__(self, scene_id: str = "GRScene_Home"):
        self.scene_id = scene_id
        self.rooms: Dict[str, SemanticRoom] = {}
        self.corridor_waypoints: List[List[float]] = []
        self.all_objects: List[SemanticObject] = []

    def add_room(self, room: SemanticRoom):
        self.rooms[room.room_id] = room

    def get_room(self, room_id: str) -> Optional[SemanticRoom]:
        return self.rooms.get(room_id)

    def find_objects(self, query: str) -> List[Tuple[SemanticObject, SemanticRoom]]:
        """根据自然语言指令或关键词检索语义图中的目标物体及其所在房间"""
        matches = []
        for r_id, room in self.rooms.items():
            for obj in room.objects:
                if _obj_matches_query(obj, query):
                    matches.append((obj, room))
        return matches

    def locate_room(self, pos: List[float]) -> SemanticRoom:
        """根据 2D/3D 坐标定位机器人当前所处的语义房间"""
        px, py = pos[0], pos[1]
        # 起始点原点 (0, 0) 是全局玄关走廊
        if abs(px) < 0.6 and abs(py) < 0.6:
            return self.rooms.get("corridor", list(self.rooms.values())[0])
            
        for r_id, room in self.rooms.items():
            if r_id == "corridor": continue
            b = room.bounds
            dist_to_center = np.sqrt((px - room.center[0])**2 + (py - room.center[1])**2)
            if b[0] <= px <= b[2] and b[1] <= py <= b[3] and dist_to_center < 3.0:
                return room
        return self.rooms.get("corridor", list(self.rooms.values())[0])

    def plan_topological_route(self, start_pos: List[float], instruction: str, spatial_rel: str = None) -> Optional[Dict]:
        """根据自然语言指令与起始位置，生成层次化拓扑导航计划
        
        路线结构:
        [起始位置] -> [出当前房间门洞] -> [主走廊中继点] -> [目标房间门洞] -> [目标房间中心] -> [目标物体 Standoff 停靠点]
        """
        query = instruction.lower()
        
        # 空间方位意图自动解析
        if spatial_rel is None:
            if any(k in query for k in ["旁边", "侧边", "边上", "side"]):
                spatial_rel = "side"
            elif any(k in query for k in ["前面", "正前", "前方", "front"]):
                spatial_rel = "front"
            elif any(k in query for k in ["左边", "左侧", "left"]):
                spatial_rel = "left"
            elif any(k in query for k in ["右边", "右侧", "right"]):
                spatial_rel = "right"
            else:
                spatial_rel = "front"
        
        # 1. 目标房间与物体语义解析
        target_room_candidate = None
        for r_id, room in self.rooms.items():
            if room.zh_name in query or room.room_type in query or r_id in query:
                target_room_candidate = room
                break

        target_obj = None
        target_room = None

        # 优先在指定房间查找目标
        if target_room_candidate:
            for obj in target_room_candidate.objects:
                if _obj_matches_query(obj, query):
                    target_obj = obj
                    target_room = target_room_candidate
                    break

        # 全图全局检索目标
        if target_obj is None:
            matches = self.find_objects(query)
            if matches:
                target_obj, target_room = matches[0]

        # 容错兜底
        if target_obj is None:
            for r_id, room in self.rooms.items():
                if len(room.objects) > 0:
                    target_obj = room.objects[0]
                    target_room = room
                    break

        # 2. 定位起始房间
        current_room = self.locate_room(start_pos)
        
        # 3. 计算目标物体外部安全 Standoff 停靠航路点
        gx, gy, theta_goal = target_obj.get_standoff_waypoint(
            robot_pos=[start_pos[0], start_pos[1], 0.0],
            spatial_rel=spatial_rel,
            standoff_dist=0.75
        )
        standoff_pos = [gx, gy]
        
        # 4. 规划拓扑航路点序列 (基于全场景实体栅格的拓扑 A* 航线生成)
        grid_targets = getattr(self, 'raw_targets_info', None)
        if not grid_targets:
            grid_targets = {f"obj_{i}": {"pos": o.pos, "size": o.size} for i, o in enumerate(self.all_objects)}
        topo_pts = route_topological_path(grid_targets, start_pos, standoff_pos)
        
        waypoints = []
        if topo_pts and len(topo_pts) >= 2:
            waypoints.append({
                "stage": "start",
                "desc": f"起点 ({current_room.zh_name})",
                "pos": [float(start_pos[0]), float(start_pos[1])],
                "room_id": current_room.room_id
            })
            for i, pt in enumerate(topo_pts[1:-1]):
                waypoints.append({
                    "stage": "corridor_transit",
                    "desc": f"拓扑航路通道点 {i+1} (对准 {target_room.zh_name})",
                    "pos": [float(pt[0]), float(pt[1])],
                    "room_id": "corridor"
                })
            waypoints.append({
                "stage": "target_docking",
                "desc": f"最终停靠点: {target_obj.zh_name} ({spatial_rel} 方位)",
                "pos": [gx, gy],
                "theta_goal": theta_goal,
                "room_id": target_room.room_id
            })
        else:
            # 备用规则拓扑路径
            waypoints.append({
                "stage": "start",
                "desc": f"起点 ({current_room.zh_name})",
                "pos": [float(start_pos[0]), float(start_pos[1])],
                "room_id": current_room.room_id
            })
            if current_room.room_id != target_room.room_id:
                waypoints.append({
                    "stage": "corridor_transit",
                    "desc": f"主走廊过道中继 (对准 {target_room.zh_name})",
                    "pos": list(target_room.doorway),
                    "room_id": "corridor"
                })
            waypoints.append({
                "stage": "target_docking",
                "desc": f"最终停靠点: {target_obj.zh_name} ({spatial_rel} 方位)",
                "pos": [gx, gy],
                "theta_goal": theta_goal,
                "room_id": target_room.room_id
            })

        return {
            "instruction": instruction,
            "current_room": current_room,
            "target_room": target_room,
            "target_object": target_obj,
            "waypoints": waypoints,
            "final_goal": [gx, gy],
            "theta_goal": theta_goal
        }

    def print_hierarchy(self):
        """打印场景语义图的完整层次拓扑"""
        print(f"\n🏠 场景语义拓扑图 (Scene Semantic Graph) · [{self.scene_id}]")
        print("="*65)
        for r_id, room in self.rooms.items():
            print(f"├── 🚪 房间: {room.zh_name} [{r_id}] (类型: {room.room_type})")
            print(f"│   ├── 边界范围: X=[{room.bounds[0]:.2f}, {room.bounds[2]:.2f}], Y=[{room.bounds[1]:.2f}, {room.bounds[3]:.2f}]")
            print(f"│   ├── 导航中心: ({room.center[0]:.2f}, {room.center[1]:.2f}) | 门洞入口: ({room.doorway[0]:.2f}, {room.doorway[1]:.2f})")
            print(f"│   └── 包含实体 ({len(room.objects)} 个):")
            for obj in room.objects:
                print(f"│       └── [{obj.category:14s}] {obj.zh_name} -> 坐标=({obj.pos[0]:.2f}, {obj.pos[1]:.2f}) 尺寸=({obj.size[0]:.2f}x{obj.size[1]:.2f})m")
        print("="*65 + "\n")


def build_scene_semantic_graph(usd_path: str = None) -> SceneSemanticGraph:
    """从 GRScenes USD 及布局数据自动构建层次化语义拓扑图"""
    if usd_path is None:
        usd_path = "/home/xujinlong/test/GRScenes/scenes/GRScenes-100/home_scenes/scenes/MV7J6NIKTKJZ2AABAAAAADA8_usd/start_result_navigation.usd"

    scene_id = os.path.basename(os.path.dirname(usd_path)).replace("_usd", "")
    graph = SceneSemanticGraph(scene_id)

    raw_objects = []
    from grscenes_importer import extract_grscene_layout
    targets_info, walls_info = {}, []
    try:
        targets_info, walls_info = extract_grscene_layout(usd_path)
        for k, v in targets_info.items():
            cat = k.split("_")[0].lower()
            raw_objects.append({
                'category': cat,
                'name': k,
                'pos': v['pos'],
                'size': v['size']
            })
    except Exception as e:
        print(f"Warning: Layout extract fallback ({e})")
    
    try:
        stage = Usd.Stage.Open(usd_path)
        if stage:
            def get_prim_info(prim):
                bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
                bbox = bbox_cache.ComputeWorldBound(prim)
                rng = bbox.ComputeAlignedRange()
                min_pt = np.array(rng.GetMin()) / 100.0
                max_pt = np.array(rng.GetMax()) / 100.0
                pos = (min_pt + max_pt) / 2.0
                size = max_pt - min_pt
                return pos.tolist(), size.tolist()

            for scope_path in ['/Root/Meshes/Furnitures', '/Root/Meshes/Animation', '/Root/Meshes/BaseAnimation/door']:
                scope = stage.GetPrimAtPath(scope_path)
                if not scope.IsValid(): continue
                if 'door' in scope_path.lower():
                    for model in scope.GetChildren():
                        pos, size = get_prim_info(model)
                        if all(s > 0 for s in size):
                            raw_objects.append({
                                'category': 'door', 'name': model.GetName(),
                                'pos': pos, 'size': size
                            })
                else:
                    for cat in scope.GetChildren():
                        cat_name = cat.GetName().lower()
                        if cat_name in CATEGORY_ZH_MAP:
                            for model in cat.GetChildren():
                                pos, size = get_prim_info(model)
                                if all(s > 0 for s in size) and sum(s > 0.25 for s in size) >= 1:
                                    raw_objects.append({
                                        'category': cat_name,
                                        'name': f"{cat_name}_{model.GetName()}",
                                        'pos': pos, 'size': size
                                    })
    except Exception as e:
        print(f"Warning: USD parse in graph builder fallback ({e})")

    if not raw_objects:
        # Fallback objects
        raw_objects = [
            {"category": "couch", "name": "LivingRoom_Couch", "pos": [6.17, 3.21, 0.42], "size": [1.65, 4.13, 0.84]},
            {"category": "table", "name": "DiningRoom_Table", "pos": [6.33, -2.94, 0.38], "size": [1.06, 2.60, 0.75]},
            {"category": "refrigerator", "name": "Kitchen_Fridge", "pos": [10.66, -1.75, 0.90], "size": [0.70, 0.70, 1.80]},
            {"category": "bed", "name": "Master_Bed", "pos": [16.02, -3.55, 0.45], "size": [2.00, 2.20, 0.80]},
            {"category": "shelf", "name": "Hallway_Shelf", "pos": [1.12, 4.62, 0.80], "size": [0.60, 1.60, 1.60]},
        ]

    # 动态锚点驱动的房间空间语义聚类 (Anchor-Driven Spatial-Semantic Room Clustering)
    room_definitions = []
    
    def _compute_anchor_doorway(anchor_objs, center):
        xs = [o["pos"][0] for o in anchor_objs]
        ys = [o["pos"][1] for o in anchor_objs]
        c_x, c_y = center[0], center[1]
        dw_x = c_x * 0.7 if abs(c_x) > 2.0 else c_x * 0.5
        dw_y = max(1.0, float(min(ys)) - 0.5) if c_y > 0 else -1.0
        return [float(dw_x), float(dw_y)]

    # 1. 厨房聚类 (以冰箱/微波炉/烤箱为锚点，聚类其周围 3.2m 内的厨电)
    k_anchors = [o for o in raw_objects if any(k in str(o["category"]).lower() for k in ["refrigerator", "microwave", "oven", "dishwasher"])]
    if k_anchors:
        k_pos = k_anchors[0]["pos"][:2]
        k_objs = [o for o in raw_objects if np.linalg.norm(np.array(o["pos"][:2]) - np.array(k_pos)) < 3.2]
        xs = [o["pos"][0] for o in k_objs]
        ys = [o["pos"][1] for o in k_objs]
        k_center = [float(np.mean(xs)), float(np.mean(ys))]
        k_bounds = [float(min(xs) - 1.2), float(min(ys) - 1.2), float(max(xs) + 1.2), float(max(ys) + 1.2)]
        room_definitions.append({
            "id": "kitchen", "type": "kitchen", "zh": "厨房",
            "x_range": [k_bounds[0], k_bounds[2]],
            "y_range": [k_bounds[1], k_bounds[3]],
            "center": k_center,
            "doorway": _compute_anchor_doorway(k_anchors, k_center)
        })
    else:
        room_definitions.append({
            "id": "kitchen", "type": "kitchen", "zh": "厨房",
            "x_range": [9.5, 12.5], "y_range": [-6.0, -1.2],
            "center": [11.0, -3.2], "doorway": [8.8, -1.0]
        })

    # 2. 客厅聚类 (以沙发为唯一核心锚点，聚类其周围 3.2m 内的茶几与电视柜)
    couches = [o for o in raw_objects if "couch" in str(o["category"]).lower() or "sofa" in str(o["category"]).lower()]
    if couches:
        c_pos = couches[0]["pos"][:2]
        l_objs = [o for o in raw_objects if np.linalg.norm(np.array(o["pos"][:2]) - np.array(c_pos)) < 3.2]
        xs = [o["pos"][0] for o in l_objs]
        ys = [o["pos"][1] for o in l_objs]
        l_center = [float(np.mean(xs)), float(np.mean(ys))]
        l_bounds = [float(min(xs) - 1.2), float(min(ys) - 1.2), float(max(xs) + 1.2), float(max(ys) + 1.2)]
        room_definitions.append({
            "id": "living_room", "type": "living_room", "zh": "客厅",
            "x_range": [l_bounds[0], l_bounds[2]],
            "y_range": [l_bounds[1], l_bounds[3]],
            "center": l_center,
            "doorway": _compute_anchor_doorway(couches, l_center)
        })
    else:
        room_definitions.append({
            "id": "living_room", "type": "living_room", "zh": "客厅",
            "x_range": [4.0, 9.5], "y_range": [1.0, 5.5],
            "center": [6.8, 3.2], "doorway": [5.5, 1.0]
        })

    # 3. 餐厅聚类 (利用多把餐椅环绕特征识别大餐桌，聚类其周围 2.2m 区域)
    chair_positions = [o["pos"][:2] for o in raw_objects if "chair" in str(o["category"]).lower()]
    all_tables = [o for o in raw_objects if "table" in str(o["category"]).lower() and "tea" not in str(o["category"]).lower()]
    
    def _count_nearby_chairs(t_pos):
        return sum(1 for cp in chair_positions if np.linalg.norm(np.array(cp) - np.array(t_pos[:2])) < 2.0)
    
    dining_tables = [t for t in all_tables if _count_nearby_chairs(t["pos"]) >= 2]
    if not dining_tables and all_tables:
        k_pos = room_definitions[0]["center"]
        dining_tables = [min(all_tables, key=lambda t: np.linalg.norm(np.array(t["pos"][:2]) - np.array(k_pos)))]

    if dining_tables:
        dt = dining_tables[0]
        d_objs = [o for o in raw_objects if np.linalg.norm(np.array(o["pos"][:2]) - np.array(dt["pos"][:2])) < 2.5]
        xs = [o["pos"][0] for o in d_objs]
        ys = [o["pos"][1] for o in d_objs]
        d_center = [float(np.mean(xs)), float(np.mean(ys))]
        d_bounds = [float(min(xs) - 1.2), float(min(ys) - 1.2), float(max(xs) + 1.2), float(max(ys) + 1.2)]
        room_definitions.append({
            "id": "dining_room", "type": "dining_room", "zh": "餐厅",
            "x_range": [d_bounds[0], d_bounds[2]],
            "y_range": [d_bounds[1], d_bounds[3]],
            "center": d_center,
            "doorway": _compute_anchor_doorway(dining_tables, d_center)
        })
    else:
        room_definitions.append({
            "id": "dining_room", "type": "dining_room", "zh": "餐厅",
            "x_range": [4.5, 9.0], "y_range": [-5.5, -1.0],
            "center": [7.8, -2.8], "doorway": [6.2, -1.0]
        })

    # 4. 卧室聚类 (按各床的位置独立划分卧室)
    beds = [o for o in raw_objects if "bed" in str(o["category"]).lower()]
    if beds:
        for b_idx, bed in enumerate(beds):
            bx, by = bed["pos"][0], bed["pos"][1]
            zh_b = "主卧" if b_idx == 0 else f"次卧 {b_idx}"
            r_id = "master_bedroom" if b_idx == 0 else f"bedroom_{b_idx}"
            b_objs = [o for o in raw_objects if np.linalg.norm(np.array(o["pos"][:2]) - np.array([bx, by])) < 2.5]
            xs = [o["pos"][0] for o in b_objs]
            ys = [o["pos"][1] for o in b_objs]
            b_center = [bx, by]
            room_definitions.append({
                "id": r_id, "type": "bedroom", "zh": zh_b,
                "x_range": [bx - 2.2, bx + 2.2],
                "y_range": [by - 2.2, by + 2.2],
                "center": b_center,
                "doorway": _compute_anchor_doorway([bed], b_center)
            })
    else:
        room_definitions.append({
            "id": "master_bedroom", "type": "master_bedroom", "zh": "主卧",
            "x_range": [14.0, 18.5], "y_range": [-6.0, -1.0],
            "center": [16.0, -3.5], "doorway": [12.8, -1.0]
        })

    # 5. 卫生间
    toilets = [o for o in raw_objects if "toilet" in str(o["category"]).lower()]
    if toilets:
        tx, ty = toilets[0]["pos"][0], toilets[0]["pos"][1]
        t_objs = [o for o in raw_objects if np.linalg.norm(np.array(o["pos"][:2]) - np.array([tx, ty])) < 2.0]
        xs = [o["pos"][0] for o in t_objs]
        ys = [o["pos"][1] for o in t_objs]
        t_center = [tx, ty]
        room_definitions.append({
            "id": "bathroom", "type": "bathroom", "zh": "卫生间",
            "x_range": [tx - 1.8, tx + 1.8],
            "y_range": [ty - 1.8, ty + 1.8],
            "center": t_center,
            "doorway": _compute_anchor_doorway([toilets[0]], t_center)
        })
    else:
        room_definitions.append({
            "id": "bathroom", "type": "bathroom", "zh": "卫生间",
            "x_range": [12.2, 14.2], "y_range": [-5.5, -1.8],
            "center": [13.0, -3.2], "doorway": [12.5, -1.5]
        })

    # 6. 主走廊过道
    room_definitions.append({
        "id": "corridor", "type": "corridor", "zh": "主走廊过道",
        "x_range": [-20.0, 25.0], "y_range": [-2.0, 2.0],
        "center": [0.0, 0.0], "doorway": [0.0, 0.0]
    })

    # 构建房间
    created_rooms = {}
    for r_def in room_definitions:
        room = SemanticRoom(
            room_id=r_def["id"],
            room_type=r_def["type"],
            zh_name=r_def["zh"],
            bounds=[r_def["x_range"][0], r_def["y_range"][0], r_def["x_range"][1], r_def["y_range"][1]],
            center=r_def["center"],
            doorway=r_def["doorway"],
            objects=[]
        )
        created_rooms[r_def["id"]] = room

    # 将物体分配到对应的房间
    for item in raw_objects:
        ox, oy, oz = item["pos"]
        cat = item["category"]
        zh_name = CATEGORY_ZH_MAP.get(cat, cat)
        sem_obj = SemanticObject(
            name=item["name"],
            category=cat,
            zh_name=zh_name,
            pos=item["pos"],
            size=item["size"]
        )

        assigned_room = None
        for r_def in room_definitions:
            if r_def["id"] == "corridor": continue
            xr, yr = r_def["x_range"], r_def["y_range"]
            if xr[0] <= ox <= xr[1] and yr[0] <= oy <= yr[1]:
                assigned_room = created_rooms[r_def["id"]]
                break
        
        if assigned_room is None:
            assigned_room = created_rooms["corridor"]

        sem_obj.room_id = assigned_room.room_id
        assigned_room.objects.append(sem_obj)
        graph.all_objects.append(sem_obj)

    # 过滤掉没有任何物体的空房间（保留主走廊）
    for r_id, room in created_rooms.items():
        if len(room.objects) > 0 or room.room_type == "corridor":
            graph.add_room(room)

    graph.raw_targets_info = targets_info
    graph.raw_walls_info = walls_info
    return graph
