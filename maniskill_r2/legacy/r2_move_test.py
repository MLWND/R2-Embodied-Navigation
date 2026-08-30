"""R2 nomimic URDF 完整运动验证: 前进/后退/转弯/掉头 (ManiSkill3 GPU)"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
import numpy as np, torch, sapien
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.registration import register_agent
from mani_skill.agents.controllers import PDJointVelControllerConfig
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig, SceneConfig
from PIL import Image

@register_agent()
class R2Robot(BaseAgent):
    uid = "r2"
    urdf_path = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_nomimic.urdf"
    urdf_config = dict()
    fix_root_link = False          # 自由基座, 靠轮子驱动移动
    disable_self_collisions = True # 手指碰撞体穿插会炸, 已禁用
    keyframes = {"stand": None}

    @property
    def _controller_configs(self):
        wn = ["wheel_left_joint", "wheel_right_joint"]
        return dict(
            pd_joint_pos=dict(
                wheels=PDJointVelControllerConfig(wn, lower=-10, upper=10, damping=500, force_limit=2000),
            ),
        )
    def _initialize(self): pass

@register_env("R2Move-v1", max_episode_steps=1000)
class R2MoveEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["r2"]
    SUPPORTED_REWARD_MODES = ["none"]
    def __init__(self, *args, robot_uids="r2", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    @property
    def _default_sim_config(self):
        return SimConfig(
            scene_config=SceneConfig(solver_position_iterations=4, solver_velocity_iterations=1),
            spacing=20, sim_freq=200, control_freq=40,
            gpu_memory_config=GPUMemoryConfig(found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18))
    @property
    def _default_sensor_configs(self):
        return [CameraConfig("base_camera", sapien_utils.look_at([3,-3,2],[0,0,0.5]), 512, 512, np.pi/2, 0.01, 100)]
    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig("render_camera", sapien_utils.look_at([3,-3,2],[0,0,0.5]), 1024, 1024, 1, 0.01, 100)
    def _load_agent(self, options):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0.085]))
    def _load_scene(self, options):
        self.ground = build_ground(self.scene)
        self.ground.set_collision_group_bit(group=2, bit_idx=30, bit=1)
    def _initialize_episode(self, env_idx, options):
        agent = self.agent
        n = len(agent.robot.active_joints)
        agent.robot.set_qpos(torch.zeros(self.num_envs, n, device=self.device))
        agent.robot.set_qvel(torch.zeros(self.num_envs, n, device=self.device))
    def evaluate(self): return {}
    def _get_obs_extra(self, info): return dict()

import gymnasium as gym
env = gym.make("R2Move-v1", obs_mode="state_dict", render_mode="sensors", num_envs=1, control_mode="pd_joint_pos")
obs, info = env.reset(seed=0)
agent = env.unwrapped.agent
robot = agent.robot
nas = env.action_space.shape
print(f"[R2] active={len(robot.active_joints)} action={nas} (轮速控制)", flush=True)

# settle
for i in range(100):
    env.step(np.zeros(nas))
bl = next(l for l in robot.links if l.name == 'base_link')
print(f"[R2] settle base z={bl.pose.p.reshape(-1)[2]:.4f}", flush=True)

def drive(steps, wl, wr, label):
    """差速驱动: wl=左轮速(rad/s), wr=右轮速(rad/s)"""
    start = bl.pose.p.reshape(-1).clone()
    for i in range(steps):
        a = np.zeros(nas)
        a[0] = wl; a[1] = wr
        env.step(a)
    p = bl.pose.p.reshape(-1)
    d = float(torch.sqrt((p[0]-start[0])**2 + (p[1]-start[1])**2))
    yaw = float(np.degrees(np.arctan2(p[1]-start[1], p[0]-start[0]))) if d > 0.02 else 0
    nan = bool(torch.any(torch.isnan(robot.get_qpos())))
    print(f"[R2] {label}: base=({p[0]:.2f},{p[1]:.2f},{p[2]:.3f}) 位移={d:.3f}m 朝向={yaw:.0f}° nan={nan}", flush=True)
    return d, yaw, nan

results = []
# 前进: 检查打滑
wl_j = next(j for j in robot.active_joints if j.name == 'wheel_left_joint')
q0 = robot.get_qpos().clone()
r0 = bl.pose.p.reshape(-1).clone()
for i in range(150):
    a = np.zeros(nas); a[0] = 5.0; a[1] = -5.0
    env.step(a)
q1 = robot.get_qpos(); r1 = bl.pose.p.reshape(-1)
# fix_root_link=False 时 qpos 前7个是 base free joint, wheel 索引需偏移
active = robot.active_joints
wl_idx = list(active).index(wl_j)
# fix_root_link=False: qpos 前7个是 base free joint, wheel 索引偏移+7
qpos_off = 7
dq = float(q1[0,wl_idx] - q0[0,wl_idx])  # 左轮转角 (get_qpos 不含 free joint)
# 读实际角速度
qvel = robot.get_qvel()
print(f'[R2] 驱动后 wheel qvel={qvel[0,wl_idx].item():.3f} rad/s (目标5.0)', flush=True)
print(f'[R2] qpos 维度={q1.shape}', flush=True)
print(f'[R2] wheel_left qpos_idx={qpos_off+wl_idx} (偏移7=freejoint)', flush=True)
dr = float(torch.sqrt((r1[0]-r0[0])**2 + (r1[1]-r0[1])**2))
eff_r = dr / max(abs(dq), 1e-6)
print(f'[R2] 前进: 轮转{dq:.2f}rad 位移{dr:.3f}m 等效半径={eff_r:.4f}m (理论0.085)', flush=True)
results.append((dr, 0, False))
results.append(drive(150, -5.0, 5.0, "后退"))
results.append(drive(150, 8.0, -2.0, "左转"))
results.append(drive(150, -2.0, 8.0, "右转"))
results.append(drive(300, 5.0, -5.0, "掉头"))

# 渲染
img = env.render()
rgb = img['sensor_data']['render_camera']['rgb'] if isinstance(img, dict) else img
arr = rgb.cpu().numpy() if hasattr(rgb, 'cpu') else np.asarray(rgb)
Image.fromarray(arr[0] if arr.ndim == 4 else arr).save("/tmp/r2_move_test.png")
print("[R2] 已保存 /tmp/r2_move_test.png", flush=True)

all_ok = not any(r[2] for r in results)
print(f"\n[R2] ★{'全部通过,无nan' if all_ok else '有nan'}", flush=True)
env.close()