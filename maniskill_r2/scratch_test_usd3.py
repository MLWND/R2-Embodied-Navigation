from pxr import Usd, UsdGeom

usd_path = "/home/xujinlong/test/GRScenes/scenes/GRScenes-100/home_scenes/scenes/MV7J6NIKTKJZ2AABAAAAADA8_usd/start_result_navigation.usd"
stage = Usd.Stage.Open(usd_path)
meshes = stage.GetPrimAtPath("/Root/Meshes")
for child in meshes.GetChildren():
    print(f"Mesh child: {child.GetName()}")
