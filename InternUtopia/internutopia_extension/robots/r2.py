import numpy as np

from internutopia.core.robot.articulation import IArticulation
from internutopia.core.robot.articulation_action import ArticulationAction
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene
from internutopia.core.util import log
from internutopia_extension.configs.robots.r2 import R2RobotCfg
from internutopia_extension.robots.jetbot import WheeledBaseRobot

# 驱动轮关节名（排除 caster 万向轮）
WHEEL_JOINT_NAMES = ('wheel_left_joint', 'wheel_right_joint')


@BaseRobot.register('R2Robot')
class R2Robot(WheeledBaseRobot):
    def __init__(self, config: R2RobotCfg, scene: IScene):
        super().__init__(config, scene)

        usd_path = config.usd_path
        self.articulation = IArticulation.create(
            prim_path=config.prim_path,
            name=config.name,
            position=self._start_position,
            orientation=self._start_orientation,
            usd_path=usd_path,
            scale=self._robot_scale,
        )

    def post_reset(self):
        super().post_reset()
        log.info(f'R2 DOF names: {self.articulation.dof_names}')
        log.info(f'R2 num_dof: {self.articulation.num_dof}')
        # 预计算驱动轮 DOF 索引（Isaac Sim DOF 顺序与 URDF 不同，且初始化后不变）
        self._wheel_idx = [i for i, n in enumerate(self.articulation.dof_names) if n in WHEEL_JOINT_NAMES]
        self.set_control_modes()
        self.set_gains()

    def _expand_wheel_action(self, control: ArticulationAction) -> ArticulationAction:
        """将差速控制器的部分轮子动作映射到驱动轮 DOF（Isaac Sim DOF 顺序与 URDF 不同）

        注意：URDF 中右轮轴 (0,-1,0) 与左轮 (0,1,0) 方向相反（镜像轮），
        差速控制器公式假设两轮"正向=前进"，因此右轮速度必须取反。
        用 joint_indices 做部分动作，避免 NaN 填充污染 PhysX 求解器。
        """
        # 差速控制器只输出 [左轮速, 右轮速]（无 joint_positions）；手臂控制器输出 7 个值，不在此展开
        if (control.joint_velocities is not None and control.joint_positions is None
                and len(control.joint_velocities) == 2):
            v_left, v_right = control.joint_velocities
            # 按关节名映射到 wheel_idx 顺序（不依赖 DOF 枚举顺序）
            velocities = np.array([
                v_left if self.articulation.dof_names[i] == 'wheel_left_joint' else -v_right
                for i in self._wheel_idx
            ])
            return ArticulationAction(joint_velocities=velocities, joint_indices=self._wheel_idx)
        return control

    def _dof_category(self, name: str) -> str:
        """返回关节类别：wheel / lift / caster_steer / caster_wheel / arm"""
        if name in WHEEL_JOINT_NAMES:
            return 'wheel'
        if 'lift' in name:
            return 'lift'
        if 'caster' in name:
            return 'caster_steer' if 'steer' in name else 'caster_wheel'
        return 'arm'

    def set_control_modes(self):
        """驱动轮用速度控制，其余关节用位置控制（caster 万向轮自由）"""
        controller = self.articulation._articulation_controller
        for i, name in enumerate(self.articulation.dof_names):
            if self._dof_category(name) == 'wheel':
                controller.switch_dof_control_mode(i, 'velocity')
            else:
                controller.switch_dof_control_mode(i, 'position')

    def set_gains(self):
        """设置关节刚度/阻尼：轮子高阻尼，手臂适中刚度"""
        dof_names = self.articulation.dof_names
        num_dof = len(dof_names)

        kps = np.zeros(num_dof)
        kds = np.zeros(num_dof)
        for i, name in enumerate(dof_names):
            cat = self._dof_category(name)
            if cat == 'wheel':
                # 驱动轮速度控制：零刚度，高阻尼
                kps[i] = 0.0
                kds[i] = 15000.0
            elif cat == 'lift':
                # 升降：高刚度
                kps[i] = 5000.0
                kds[i] = 500.0
            elif cat == 'caster_steer':
                # caster 转向关节高阻尼防抖动（NVIDIA 论坛建议 1000+）
                # 参考: https://forums.developer.nvidia.com/t/caster-wheel-jitter-causes-unintended-rotation-and-unstable-motion-in-a-two-wheel-robot/359262
                kps[i] = 0.0
                kds[i] = 1000.0
            elif cat == 'caster_wheel':
                # caster 滚动关节中阻尼
                kps[i] = 0.0
                kds[i] = 100.0
            else:
                # 手臂/腰部/头部：高刚度（实验：2000）
                kps[i] = 2000.0
                kds[i] = 100.0

        self.articulation.set_gains(kps=kps, kds=kds)
