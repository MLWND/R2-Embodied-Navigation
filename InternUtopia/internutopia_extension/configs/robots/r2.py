from typing import Optional

from internutopia.core.config import RobotCfg
from internutopia.macros import gm
from internutopia_extension.configs.controllers import (
    DifferentialDriveControllerCfg,
    JointControllerCfg,
    MoveAlongPathPointsControllerCfg,
    MoveToPointBySpeedControllerCfg,
    RotateControllerCfg,
)
from internutopia_extension.configs.sensors import RepCameraCfg

# R2 实测参数（来自 URDF/STL 测量）
# 轮距 = 2 × 0.229m（wheel joint y 坐标），轮径 = 0.17m（STL 直径）
WHEEL_BASE = 0.458
WHEEL_RADIUS = 0.085

move_by_speed_cfg = DifferentialDriveControllerCfg(
    name='move_by_speed', wheel_base=WHEEL_BASE, wheel_radius=WHEEL_RADIUS
)

move_to_point_cfg = MoveToPointBySpeedControllerCfg(
    name='move_to_point',
    forward_speed=1.0,
    rotation_speed=1.0,
    threshold=0.1,
    sub_controllers=[move_by_speed_cfg],
)

move_along_path_cfg = MoveAlongPathPointsControllerCfg(
    name='move_along_path',
    forward_speed=1.0,
    rotation_speed=1.0,
    threshold=0.1,
    sub_controllers=[move_to_point_cfg],
)

rotate_cfg = RotateControllerCfg(
    name='rotate',
    rotation_speed=2.0,
    threshold=0.02,
    sub_controllers=[move_by_speed_cfg],
)

camera_cfg = RepCameraCfg(
    name='camera',
    prim_path='base_link/Camera',
    resolution=(640, 480),
    rgba=True,
    depth=True,
)

# 手臂关节位置控制器（按名字映射 DOF，自动处理顺序）
left_arm_cfg = JointControllerCfg(
    name='left_arm',
    joint_names=[f'J{i}_left_joint' for i in range(1, 8)],
)

right_arm_cfg = JointControllerCfg(
    name='right_arm',
    joint_names=[f'J{i}_right_joint' for i in range(1, 8)],
)


class R2RobotCfg(RobotCfg):
    # meta info
    name: Optional[str] = 'r2'
    type: Optional[str] = 'R2Robot'
    prim_path: Optional[str] = '/r2'
    usd_path: Optional[str] = gm.ASSET_PATH + '/robots/r2/r2.usd'
