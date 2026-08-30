# R2 在 ManiSkill3 下的运动验证

## 环境
- conda env: `maniskill` (python 3.11, ManiSkill 3.0.1 + SAPIEN 3.0.3 + Torch 2.13)
- GPU: A800 80GB (CUDA_VISIBLE_DEVICES=1, GPU0 常被他人 vLLM 占用)
- 运行: `CUDA_VISIBLE_DEVICES=1 ~/.conda/envs/maniskill/bin/python -u r2_move_test.py`

## 关键结论 (2026-08-18): 里程计闭环精确运动 ✅ (最佳方案)

**方案: 原始 URDF + 4 swivel caster + 里程计闭环控制 (参考 R1 真机 ROS2 驱动)**

### R1 真机驱动方式 (1111/ 目录, 教程 4.1.1)
- `/move` (Move.srv): distance(mm) / angle(度) 位置命令
- `pose_2d`: line_speed / palstance / x / y / radian 反馈闭环
- **底盘控制器: 位置命令 → 线速度/角速度 → 轮子速度, 里程计反馈闭环**

### 里程计闭环控制 (模拟 R1 底盘控制器)
1. **diff_drive 差速运动学**: 线速度/角速度 → 轮子速度
   - **右轮轴 -y, 需取反**: `v_r = -(v + ω×L/2)/r` (符号错误 → 前进变自旋!)
2. **里程计闭环 (P 控制)**: 位置误差 → 线速度命令 (v_cmd = Kp × err)
3. **航向保持**: yaw 误差 → 角速度命令 (模拟 pose_2d radian 反馈)
4. **累积角度**: 避免 yaw ±180° 环绕误判
5. **caster μ=0.05 + swivel 摩擦=0**: 自旋阻力最小, fork 自由转向

### 验证结果 (同一场景连续执行, 运动间停车 50 步)
| 运动 | 目标 | 实际 | 精度 |
|---|---|---|---|
| 前进 | 0.5m | 0.485m | 97% |
| 后退 | 0.5m | 0.484m | 97% |
| 左转 | 90° | +86.4° | 96% |
| 右转 | 90° | -82.1° | 91% |
| 原地左转 | 180° | +173.3° | 96% (位移 0.059m) |
| 原地右转 | 180° | -175.4° | 97% (位移 0.036m) |

### 对比 (速度控制 vs 里程计闭环)
| 指标 | 速度控制 (之前) | 里程计闭环 (现在) |
|---|---|---|
| 前进效率 | 59% | 97% |
| 自旋 | 113° | 174° |
| 自旋位移 | 0.16m | 0.036m |
| 直线性 | Δyaw 漂移 | Δyaw≈0 |

### 文件
- `r2_urdf_v01_swivel.urdf` — 原始 URDF + 4 swivel caster (InternUtopia urdf/ 目录)
- `caster_wheel.STL` — caster 小轮 mesh (半径 0.03, 轴沿 y)
- `r2_moves_odom.py` — **里程计闭环运动测试 (最佳方案)**

## 关键结论 (2026-08-18 早): 速度控制基础运动验证 ✅ (swivel caster)

**方案: 原始 URDF + 4 个 swivel caster(前后各一对)+ 禁用自碰撞**

### 三个关键突破 (2026-08-18)

**1. 翻倒根因 = 自碰撞 (已解决)**
- 原始 URDF 的 base_link STL 碰撞体覆盖 z∈[0.07, 1.79](整个躯干),手臂(J1/J2/hand)在躯干内部与它重叠
- settle 时自碰撞把机器人顶起来 → 翻倒 (roll 到 -174°)
- 修复: 禁用自碰撞 `set_collision_groups([1, 1, 1<<29, 0])` (保留 g0/g1 与地面接触, g2 bit 29 忽略自碰撞)

**2. 自旋打滑根因 = 球 caster 不能自由转向 (已解决)**
- 之前的球 caster (r=0.03) 自旋时侧向滑动产生阻力 (≈30.8 N·m), 几乎抵消驱动扭矩 (≈31.6 N·m)
- 修复: 真正的 swivel caster — base → swivel joint(绕z自由旋转) → fork → wheel joint(绕y) → 小轮 → ground
- 小轮用 STL mesh (参考原始驱动轮), 半径 0.03, 轴沿 y

**3. 自旋方向不稳定 = 残余角速度 (已解决)**
- 运动间不 settle 时, 残余角速度导致自旋方向反转
- 修复: 运动间 settle (目标 0, 100 步)

### 最终配置
- 驱动轮 μ=1.5 (原始 Gazebo 配置)
- caster 小轮 μ=0.2 (自旋最佳点, 接近 180° 掉头)
- swivel 转向轴: 绕 z 轴自由旋转
- 禁用自碰撞 (代码里设置)

### 验证结果 (独立测试, 每个运动新场景)
| 运动 | 位移 | Δyaw | 判定 |
|---|---|---|---|
| 前进 | 0.514m +x | +6.5° | ✅ |
| 后退 | 0.722m -x | -1.2° | ✅ |
| 左转弧 | 0.323m | +30.0° | ✅ 左转 |
| 右转弧 | 0.333m | -41.3° | ✅ 右转 |
| 原地左转 | 0.091m | +177.6° | ✅ 接近 180° 掉头 |
| 原地右转 | 0.068m | -139.5° | ✅ 接近 180° 掉头 |

### 剩余问题
- 前进效率 59% (驱动轮 STL 凸包滚动阻力 + caster 摩擦消耗牵引力)
- 序列中原地左转受 settle 影响 (独立测试 177.6°)

### 文件
- `r2_urdf_v01_swivel.urdf` — 原始 URDF + 4 swivel caster (InternUtopia urdf/ 目录)
- `caster_wheel.STL` — caster 小轮 mesh (半径 0.03, 轴沿 y)
- `r2_moves_swivel.py` — 完整运动序列测试 (运动间 settle)

## 关键结论 (2026-08-17)

### 打滑问题 = 三个独立问题 (P0, 已基本解决)

**1. R2 URDF 在 ManiSkill3 GPU 下 NaN 的根因 = 12 个手指 mimic 关节** (历史)
- PhysX 用 fixed-tendon 模拟 mimic, GPU backend 下数值不稳定 → 施加驱动时 NaN
- 去掉 `<mimic>` 标签后 (r2_urdf_v01_nomimic.urdf) 完全解决

**2. 轮子碰撞体朝向错误 → 后翻** (2026-08-17 晚间定位)
- 之前生成的圆柱轮 URDF 把 `<origin rpy="1.5708 0 0"/>` 插在 `<geometry>` 内部 (无效顺序), 圆柱轴未旋转
- SAPIEN 中 shape 朝向变成 120° 绕 (1,1,1), 轮子无法滚动 → 机器人后仰匀速翻转 (后空翻)
- 修复: `r2_urdf_v01_cylwheel.urdf` — `<collision><origin rpy="1.5708 0 0"/><geometry><cylinder radius="0.085" length="0.13"/></geometry></collision>`

**3. 瞬时加速 → 起步俯仰瞬态超稳定极限**
- 0→5 rad/s 一步到位 → 起步瞬态把机器人推过 26° 俯仰极限 (COM 0.56m 高) → 缓慢后翻
- 修复: 速度斜坡 (0→目标 分 150 步 = 0.75s)

### ManiSkill env 遗留阻塞: 驱动时悬浮 (未解决)
- 同样 URDF + 斜坡, ManiSkill env 中机器人 base_z 缓慢升到 ~13mm, 轮子空转, 位移仅 0.03m
- 纯 SAPIEN 相同代码完全正常 (0.63 m/s, 贴地)
- 已排除: 地面/碰撞组/contact_offset/solver/步进方式/驱动参数 (详见交接文档 3.7.3)
- 运动验证暂走纯 SAPIEN 路径: `r2_drive_pure_sapien.py`

## 驱动配方 (纯 SAPIEN 路径, 已验证)
1. URDF: `r2_urdf_v01_cylwheel.urdf`
2. `set_drive_properties(0, 500, 2000, "force")` + 每步 `set_drive_velocity_target(tgt)`
3. 速度斜坡: 0→10 rad/s 分 150 步
4. **目标速度加倍**: SAPIEN 驱动半速特性 (目标 10 → 实际 ~7.4 rad/s)
5. 结果: 前进 1.73m/2.75s @ 0.63 m/s, 74% 效率, 俯仰 0°, 贴地

## 文件
- `r2_urdf_v01_cylwheel.urdf` — P0 修复版 (圆柱轮 + 修正 caster) — 在 InternUtopia assets urdf/ 目录
- `r2_drive_pure_sapien.py` — 纯 SAPIEN 前进驱动验证 (可用)
- `r2_urdf_v01_nomimic.urdf` — 去 mimic 的 URDF (历史)

## 下一步
1. ~~定位 ManiSkill env 悬浮问题~~ ✅ 已解决 (balance_passive_force=False)
2. ~~后退/转弯/掉头验证~~ ✅ 已解决 (里程计闭环, r2_moves_odom.py)
3. GRScenes 场景导入 ManiSkill3 (SAPIEN 不支持 USD, 需转换路径)
4. VLA 训练 pipeline (IL 模仿学习)
