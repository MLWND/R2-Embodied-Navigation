from pxr import Usd, UsdGeom

usd_path = "/home/xujinlong/test/GRScenes/scenes/GRScenes-100/home_scenes/scenes/MV7J6NIKTKJZ2AABAAAAADA8_usd/start_result_navigation.usd"
stage = Usd.Stage.Open(usd_path)
base = stage.GetPrimAtPath("/Root/Meshes/Base")
if base.IsValid():
    for child in base.GetChildren():
        print(f"Base child: {child.GetName()}")
