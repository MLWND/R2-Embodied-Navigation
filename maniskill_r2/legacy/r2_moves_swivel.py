"""swivel caster 完整运动序列 (运动间 settle)"""
import os
import numpy as np
import sapien

URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_swivel.urdf"
MESH = "/home/xujinlong/test/robot_example1_EN_demo/robot_example1_EN_demo/robot_model/r2_urdf_v01/meshes"
TMP = "/tmp/r2_swivel_tmp.urdf"
with open(URDF) as f:
    urdf = f.read()
urdf = urdf.replace("package://r2_urdf_v01/meshes/", MESH + "/")
with open(TMP, "w") as f:
    f.write(urdf)

TARGET = 10.0
RAMP = 150
SETTLE = 100

def euler(q):
    w, x, y, z = q
    roll = np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y-x*z), -1, 1)))
    yaw = np.degrees(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    return roll, pitch, yaw

scene = sapien.Scene()
scene.set_timestep(0.005)
scene.add_ground(altitude=0, render=False)
loader = scene.create_urdf_loader()
loader.fix_root_link = False
loader.set_material(1.5, 1.5, 0.0)
robot = loader.load(TMP)
robot.set_pose(sapien.Pose(p=[0, 0, 0.0]))
robot.set_qpos(np.zeros(robot.dof))
for l in robot.get_links():
    for sh in l.get_collision_shapes():
        sh.set_collision_groups([1, 1, 1<<29, 0])
for l in robot.get_links():
    if 'caster' in l.get_name() and 'wheel' in l.get_name():
        for sh in l.get_collision_shapes():
            sh.set_physical_material(sapien.physx.PhysxMaterial(0.2, 0.2, 0.0))
wl_j = [j for j in robot.get_joints() if j.get_name() == "wheel_left_joint"][0]
wr_j = [j for j in robot.get_joints() if j.get_name() == "wheel_right_joint"][0]
base = [l for l in robot.get_links() if l.get_name() == "base_link"][0]
wl_j.set_drive_properties(0.0, 500.0, 2000.0, "force")
wr_j.set_drive_properties(0.0, 500.0, 2000.0, "force")
for i in range(200):
    scene.step()

def drive(steps, wl_cmd, wr_cmd, label):
    p0 = base.get_pose().p.copy()
    yaw0 = euler(base.get_pose().q)[2]
    for i in range(steps):
        tgt = min(TARGET, TARGET*(i+1)/RAMP)
        wl_j.set_drive_velocity_target(wl_cmd * tgt / TARGET)
        wr_j.set_drive_velocity_target(wr_cmd * tgt / TARGET)
        scene.step()
    # settle
    wl_j.set_drive_velocity_target(0.0)
    wr_j.set_drive_velocity_target(0.0)
    for i in range(SETTLE):
        scene.step()
    p1 = base.get_pose().p
    yaw1 = euler(base.get_pose().q)[2]
    d = float(np.linalg.norm(p1[:2]-p0[:2]))
    dyaw = (yaw1 - yaw0 + 180) % 360 - 180
    z = p1[2]
    print(f"[MOVE] {label}: 位移={d:.3f}m 方向=({p1[0]-p0[0]:+.3f},{p1[1]-p0[1]:+.3f}) Δyaw={dyaw:+.1f}° base_z={z:.4f}", flush=True)

print(f"[SETTLE] base_z={base.get_pose().p[2]:.4f}", flush=True)
drive(300, +TARGET, -TARGET, "前进")
drive(300, -TARGET, +TARGET, "后退")
drive(300, +TARGET*0.3, -TARGET, "左转弧")
drive(300, +TARGET, -TARGET*0.3, "右转弧")
drive(600, -TARGET, -TARGET, "原地左转")
drive(600, +TARGET, +TARGET, "原地右转")
