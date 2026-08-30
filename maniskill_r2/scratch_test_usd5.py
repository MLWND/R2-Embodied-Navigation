from pxr import Usd, UsdGeom
import numpy as np

usd_path = "/home/xujinlong/test/GRScenes/scenes/GRScenes-100/home_scenes/scenes/MV7J6NIKTKJZ2AABAAAAADA8_usd/start_result_navigation.usd"
stage = Usd.Stage.Open(usd_path)
walls = stage.GetPrimAtPath("/Root/Meshes/Base/wall")
if walls.IsValid():
    for child in walls.GetChildren():
        bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(child)
        rng = bbox.ComputeAlignedRange()
        min_pt = np.array(rng.GetMin())
        max_pt = np.array(rng.GetMax())
        pos = (min_pt + max_pt) / 200.0
        size = (max_pt - min_pt) / 100.0
        print(f"Wall {child.GetName()} pos {pos}, size {size}")
