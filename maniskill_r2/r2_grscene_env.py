"""GRScenes-100 多场景标准 Gymnasium 强化学习导航避障环境 (R2GRSceneNavEnv)"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sapien

from grscenes_importer import extract_grscene_layout
from depth_projector import compute_standoff_waypoint

URDF = "/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_swivel.urdf"
MESH = "/home/xujinlong/test/robot_example1_EN_demo/robot_example1_EN_demo/robot_model/r2_urdf_v01/meshes"
TMP = "/tmp/r2_swivel_ppo.urdf"

WHEEL_R = 0.085
TRACK = 0.458
MAX_V = 0.55
MAX_W = 1.5

def euler(q):
    w, x, y, z = q
    roll = np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y-x*z), -1, 1)))
    yaw = np.degrees(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    return roll, pitch, yaw

def diff_drive(v, omega):
    v_l = (v - omega * TRACK / 2) / WHEEL_R
    v_r = -(v + omega * TRACK / 2) / WHEEL_R
    return v_l, v_r

class R2GRSceneNavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, scene_dir=None, max_steps=600):
        super().__init__()
        self.max_steps = max_steps
        self.step_count = 0
        self.scene_dir = scene_dir
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.scene = None
        self.robot = None
        self.wl_j = None
        self.wr_j = None
        self.base = None
        self.robot_shapes = set()
        
        self.goal_pos = np.array([5.60, -2.47], dtype=np.float32)
        self.prev_dist = 0.0
        self.targets_info = {}
        self.walls_info = []
        self.prev_pos = np.zeros(2, dtype=np.float32)
        self.prev_yaw = 0.0
        self.sim_dt = 0.01 * 5  # scene.timestep * steps_per_action

        # 单次初始化 PhysX 物理场景 (常驻内存，无需重复重建)
        self._init_simulation()

    def _init_simulation(self):
        with open(URDF) as f:
            urdf = f.read().replace("package://r2_urdf_v01/meshes/", MESH + "/")
        with open(TMP, "w") as f:
            f.write(urdf)

        self.scene = sapien.Scene()
        self.scene.set_timestep(0.01)
        self.scene.add_ground(altitude=0, render=True)

        usd_file = None
        if self.scene_dir and os.path.exists(os.path.join(self.scene_dir, "start_result_navigation.usd")):
            usd_file = os.path.join(self.scene_dir, "start_result_navigation.usd")
            
        self.targets_info, self.walls_info = extract_grscene_layout(usd_file)

        for name, info in self.targets_info.items():
            b = self.scene.create_actor_builder()
            sx, sy, sz = info["size"]
            b.add_box_collision(half_size=[sx/2, sy/2, sz/2])
            b.add_box_visual(half_size=[sx/2, sy/2, sz/2])
            actor = b.build_static(name=name)
            actor.set_pose(sapien.Pose(p=info["pos"]))

        for i, w in enumerate(self.walls_info):
            b = self.scene.create_actor_builder()
            sx, sy, sz = w["size"]
            b.add_box_collision(half_size=[sx/2, sy/2, sz/2])
            b.add_box_visual(half_size=[sx/2, sy/2, sz/2])
            actor = b.build_static(name=f"wall_{i+1}")
            actor.set_pose(sapien.Pose(p=w["pos"]))

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = False
        loader.set_material(1.5, 1.5, 0.0)
        self.robot = loader.load(TMP)
        self.robot.set_pose(sapien.Pose(p=[0, 0, 0.0]))
        
        # 设定标准巡航站姿 (收起机械臂，抬高腰部，避免双手拖地摩擦)
        self.default_qpos = np.zeros(self.robot.dof, dtype=np.float32)
        for idx, j in enumerate(self.robot.get_active_joints()):
            name = j.get_name()
            if 'J2_left' in name: self.default_qpos[idx] = 0.6
            elif 'J2_right' in name: self.default_qpos[idx] = -0.6
            elif 'J4_left' in name: self.default_qpos[idx] = 0.8
            elif 'J4_right' in name: self.default_qpos[idx] = 0.8
            elif 'lift' in name: self.default_qpos[idx] = 0.25
            elif 'waist' in name: self.default_qpos[idx] = 0.0

        self.robot.set_qpos(self.default_qpos)

        self.robot_shapes = set()
        for l in self.robot.get_links():
            for sh in l.get_collision_shapes():
                self.robot_shapes.add(sh)
                sh.set_collision_groups([1, 1, 1<<29, 0])

        for j in self.robot.get_joints():
            if 'swivel' in j.get_name():
                j.friction = 0.0
        for l in self.robot.get_links():
            if 'caster' in l.get_name() and 'wheel' in l.get_name():
                for sh in l.get_collision_shapes():
                    sh.set_physical_material(sapien.physx.PhysxMaterial(0.01, 0.01, 0.0))

        self.wl_j = [j for j in self.robot.get_joints() if j.get_name() == "wheel_left_joint"][0]
        self.wr_j = [j for j in self.robot.get_joints() if j.get_name() == "wheel_right_joint"][0]
        self.base = [l for l in self.robot.get_links() if l.get_name() == "base_link"][0]

        self.wl_j.set_drive_properties(0.0, 800.0, 4000.0, "force")
        self.wr_j.set_drive_properties(0.0, 800.0, 4000.0, "force")

        for _ in range(50): self.scene.step()

    def _get_lidar_scan(self, base_pos, base_yaw_deg, num_rays=48, fov_deg=260.0, max_range=15.0):
        angles = np.linspace(-fov_deg/2, fov_deg/2, num_rays)
        ranges = []
        px = self.scene.physx_system
        origin = np.array([base_pos[0], base_pos[1], 0.30], dtype=np.float32)

        for deg in angles:
            rad = np.radians(base_yaw_deg + deg)
            d = np.array([np.cos(rad), np.sin(rad), 0.0], dtype=np.float32)
            curr_origin = origin.copy()
            total_dist = 0.0
            hit_external = False

            while total_dist < max_range:
                hit = px.raycast(curr_origin.astype(np.float32), d, max_range - total_dist)
                if not hit:
                    total_dist = max_range
                    break
                total_dist += hit.distance
                if hit.shape in self.robot_shapes:
                    curr_origin = np.array(hit.position) + d * 0.02
                    total_dist += 0.02
                else:
                    hit_external = True
                    break
            ranges.append(float(total_dist if hit_external else max_range))

        return np.array(ranges, dtype=np.float32)

    def _get_obs(self):
        pos = self.base.get_entity_pose().p
        yaw = euler(self.base.get_entity_pose().q)[2]
        lidar = self._get_lidar_scan(pos, yaw)
        lidar_norm = np.clip(lidar / 10.0, 0.0, 1.0)

        dx = self.goal_pos[0] - pos[0]
        dy = self.goal_pos[1] - pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        target_angle_deg = np.degrees(np.arctan2(dy, dx))
        heading_err = (target_angle_deg - yaw + 180) % 360 - 180
        heading_rad = np.radians(heading_err)

        # 从位置差分计算实际线速度与角速度
        dp = np.array([pos[0] - self.prev_pos[0], pos[1] - self.prev_pos[1]])
        v_linear = float(np.linalg.norm(dp)) / max(self.sim_dt, 1e-6)
        yaw_diff = (yaw - self.prev_yaw + 180) % 360 - 180
        v_angular = float(np.radians(yaw_diff)) / max(self.sim_dt, 1e-6)
        self.prev_pos[:] = [pos[0], pos[1]]
        self.prev_yaw = yaw

        obs = np.zeros(52, dtype=np.float32)
        obs[:48] = lidar_norm
        obs[48] = dist / 10.0
        obs[49] = heading_rad / np.pi
        obs[50] = np.clip(v_linear / MAX_V, 0.0, 1.0)
        obs[51] = np.clip(v_angular / MAX_W, -1.0, 1.0)
        return obs, dist, heading_err, np.min(lidar)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        # 原地复位机器人状态
        self.robot.set_pose(sapien.Pose(p=[0, 0, 0.0]))
        self.robot.set_qpos(self.default_qpos)
        self.wl_j.set_drive_velocity_target(0.0)
        self.wr_j.set_drive_velocity_target(0.0)
        self.prev_pos[:] = [0.0, 0.0]
        self.prev_yaw = 0.0
        
        if self.targets_info:
            target_keys = list(self.targets_info.keys())
            chosen_key = target_keys[self.np_random.integers(0, len(target_keys))]
            chosen = self.targets_info[chosen_key]
            p = chosen["pos"]
            sx, sy = chosen["size"][0], chosen["size"][1]
            gx, gy, _ = compute_standoff_waypoint(
                p, [0, 0, 0], spatial_relation="front", standoff_dist=0.75,
                obj_half_extents=[sx / 2.0, sy / 2.0]
            )
            self.goal_pos = np.array([gx, gy], dtype=np.float32)

        for _ in range(20): self.scene.step()

        obs, dist, _, _ = self._get_obs()
        self.prev_dist = dist
        return obs, {}

    def step(self, action):
        self.step_count += 1
        
        v_cmd = float(np.clip((action[0] + 1.0) * 0.5 * MAX_V, 0.0, MAX_V))
        w_cmd = float(np.clip(action[1] * MAX_W, -MAX_W, MAX_W))

        v_l, v_r = diff_drive(v_cmd, w_cmd)
        self.wl_j.set_drive_velocity_target(v_l)
        self.wr_j.set_drive_velocity_target(v_r)
        
        for _ in range(5):
            self.scene.step()

        obs, dist, heading_err, min_lidar = self._get_obs()

        r_progress = 3.0 * (self.prev_dist - dist)
        r_heading = 0.1 * np.cos(np.radians(heading_err))
        r_step = -0.01
        
        reward = r_progress + r_heading + r_step
        self.prev_dist = dist

        terminated = False
        truncated = (self.step_count >= self.max_steps)
        info = {"success": False, "docking_error": dist, "min_clearance": min_lidar}

        if min_lidar < 0.20:
            reward -= 5.0

        if dist < 0.20:
            reward += 10.0
            terminated = True
            info["success"] = True

        return obs, reward, terminated, truncated, info
