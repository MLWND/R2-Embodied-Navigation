from pxr import Usd, UsdGeom
import numpy as np

usd_path = "/home/xujinlong/test/GRScenes/scenes/GRScenes-100/home_scenes/scenes/MV7J6NIKTKJZ2AABAAAAADA8_usd/start_result_navigation.usd"
stage = Usd.Stage.Open(usd_path)
furnitures = stage.GetPrimAtPath("/Root/Meshes/Furnitures")
targets = {}

cat_map = {
    "bed": "卧室大床",
    "couch": "客厅沙发",
    "sofa": "客厅沙发",
    "table": "餐桌",
    "desk": "书桌",
    "shelf": "书架",
    "tv": "电视",
    "chair": "椅子",
    "cabinet": "柜子",
    "wardrobe": "衣柜",
    "fridge": "冰箱"
}
allowed = {"bed", "couch", "sofa", "table", "desk", "shelf", "tv", "chair", "cabinet", "wardrobe", "fridge"}

for cat in furnitures.GetChildren():
    cat_name = cat.GetName().lower()
    if cat_name not in allowed:
        continue
    for model in cat.GetChildren():
        bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(model)
        rng = bbox.ComputeAlignedRange()
        min_pt = np.array(rng.GetMin())
        max_pt = np.array(rng.GetMax())
        
        size = (max_pt - min_pt) / 100.0
        if sum(s > 0.3 for s in size) < 2:
            continue
            
        pos = (min_pt + max_pt) / 200.0
        
        zh_name = cat_map.get(cat_name, cat_name)
        key = f"{cat_name}_{model.GetName()} ({zh_name})"
        targets[key] = {
            "pos": pos.tolist(),
            "size": size.tolist(),
            "color": [0.5, 0.5, 0.5, 1.0],
            "room": "Unknown"
        }

for k, v in targets.items():
    print(k, v)
