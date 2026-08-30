from pxr import Usd, UsdGeom
usd_path = "/home/xujinlong/test/GRScenes/scenes/GRScenes-100/home_scenes/scenes/MV7J6NIKTKJZ2AABAAAAADA8_usd/start_result_navigation.usd"
stage = Usd.Stage.Open(usd_path)
if not stage:
    print("Failed to open")
else:
    print("Opened USD")
    furnitures = stage.GetPrimAtPath("/Root/Meshes/Furnitures")
    if furnitures.IsValid():
        print("Found Furnitures")
        for cat in furnitures.GetChildren():
            print(f"Cat: {cat.GetName()}")
            for model in cat.GetChildren():
                print(f"  Model: {model.GetName()}")
                bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
                bbox = bbox_cache.ComputeWorldBound(model)
                range = bbox.ComputeAlignedRange()
                min_pt = range.GetMin()
                max_pt = range.GetMax()
                print(f"    Min: {min_pt}, Max: {max_pt}")
