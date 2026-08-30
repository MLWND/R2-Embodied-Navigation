# Genesis R2/R1 仿真调试工作区

InternUtopia R2 机器人从 Isaac Sim 迁移到 Genesis 的调试脚本集。
R1 是真实机器人(深圳启明),R2 是同类产品,URDF 相同(参考 `robot_example1_EN_demo`)。

## 目录结构

```
genesis_smoke/
├── urdf_tools/    # URDF 修改脚本(按顺序执行)
├── verify/        # 验证/渲染脚本
├── diag/          # 物理诊断脚本
├── logs/          # 调试日志(归档)
├── archive/       # 临时脚本/图片(归档,不删除)
└── output/        # 关键渲染图
```

## URDF 修改脚本(urdf_tools/,按顺序)

| 脚本 | 作用 |
|---|---|
| `fix_wheel_height.py` | 驱动轮 z 0.075→0.085(轮底归零) |
| `fix_hand_orientation.py` | 手指从手掌顶部→底部伸出(朝下) |
| `fix_finger_layout.py` | 手指 y 排开→x 排开(掌朝内) |
| `enlarge_fingers.py` | 加大手指尺寸/间距(5 指可见) |
| `finger_segments.py` | 手指 3 段 box 近似指节(视觉圆滑) |
| `add_fingers.py` | 补充 6 指手爪(thumb_flex/abduct + 4 指) |

**URDF 路径**: `/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf`

## 验证脚本(verify/)

| 脚本 | 作用 |
|---|---|
| `natural_pose_check2.py` | 手臂自然下垂(0 位姿)+ 强制保持,验证不插地 |
| `natural_front_side.py` | 正面+侧面渲染自然下垂姿态 |
| `hand_grasp_check.py` | 手指弯曲抓取动作验证 |
| `arm_length_check.py` | 手臂长度测量(URDF vs 仿真) |
| `arm_up_check.py` | J2 抬臂 ±90° 验证(旧方案) |
| `drive_final.py` | 最终驱动验证(前进→转向→前进) |
| `smoke_test.py` | 阶段 0 冒烟测试 |

## 诊断脚本(diag/)

| 脚本 | 作用 |
|---|---|
| `pose_diag.py` | 加载后各 link 位置 + 轮底高度 + 接触 |
| `links_z_diag.py` | 关键 link 世界 z 位置 |
| `bbox_diag.py` | link 碰撞体 bbox |
| `grounding_diag.py` | 轮底高度 + 接触力 |
| `contact_force_drive.py` | 驱动时切向力 |
| `cdof_diag.py` | 接触点速度手算 |
| `box_friction_test.py` | box 摩擦对照 |
| `speed_vel_diag.py` | 速度控制时轮子角速度 |
| `scan_wheel_height.py` | 轮高扫描 |
| `quick_wheel_test.py` | 快速轮子测试 |

## 关键结论(记忆文件 genesis-migration-stage0.md)

1. **驱动轮 z=0.083**(插地 2mm 确保接触稳定)— 0.085025 时轮底在接触阈值边缘导致打滑
2. **手臂自然下垂**:0 位姿 + 关节 kp=5000 保持,手离地 54cm 不插地
3. **6 指灵巧手**:thumb_flex/abduct + index/middle/ring/pinky,57 DOF(GR1 手移植)
4. **手指朝下**:从手掌底部伸出,沿 x 排开(掌朝内),GR1 真实 mesh
5. **驱动**:每步 set_dofs_velocity,轮子 kp=1000/kv=100,前进 [v,-v] 转向 [v,v]
6. **求解器**:substeps=8 + friction_cone='elliptic' + impratio=100 + contact_resolution='signorini'
7. **运行环境**:conda env `genesis`,GPU1(CUDA_VISIBLE_DEVICES=1)

## 运行方式

```bash
cd /home/xujinlong/test/genesis_smoke
CUDA_VISIBLE_DEVICES=1 ~/.conda/envs/genesis/bin/python verify/natural_front_side.py
```

## GRScenes 场景着色(urdf_tools/grscene_extract_color.py)

**问题**: GRScenes-100 材质全部是 MDL(`info:mdl:sourceAsset`),Genesis 只解析
UsdPreviewSurface,MDL 烘焙又依赖 omniverse-kit(装了但会引发崩溃问题),所以直接
加载整个场景全是灰白色。Genesis 不读"已绑定材质"mesh 上的 displayColor,
旧 displayColor 方案无效。

**方案**: 把 MDL 材质转成标准 UsdPreviewSurface 网络写回 USD
(`*_colored.usd`),Genesis 原生可解析:

1. UE4 参数模板(Num*): 颜色在 USD `inputs:BaseColor_Color`/`BaseColor_Tex`,
   `IsBaseColorTex` 开关选择(占材质大头,含全部房间墙体);
2. KooPbr::KooMtl 模板: 参数在 MDL 文件,diffuse 是 color() 或贴图;
3. WorldGridMaterial(墙体/天花): 程序化网格材质,固定灰色近似;
4. DayMaterial(天空球)等自由格式跳过。

```bash
# 转换(输出 *_colored.usd, 贴图用相对路径引用,随 GRScenes 目录整体迁移有效)
~/.conda/envs/genesis/bin/python urdf_tools/grscene_extract_color.py [scene.usd] [输出.usd]
# 渲染验证
~/.conda/envs/genesis/bin/python verify/grscene_load.py <scene>_colored.usd output/grscene_colored.png
```

## 环境坑(重要)

- **`PXR_WORK_THREAD_LIMIT=8` 必须在 import genesis/pxr 前设置**(已写进
  grscene_load.py): 机器 112 核,usd-core 按核数起线程池;用户线程配额
  ulimit -u 1024 被 IDE 进程占掉大半时,USD 起线程失败 → 堆损坏/段错误
  (`malloc(): unsorted double linked list corrupted`)。
- GRScenes 下载包里的 `Materials` 软链接被打平成了文本文件(内容是相对路径
  如 `../../Materials`),解析时需读内容定位共享材质目录。
- `omniverse-kit` pip 包装过一版(2026-08-15),Genesis 检测到后会走 MDL 烘焙
  子进程;如需禁用可卸载或忽略,转换脚本不依赖它。

### 渲染大片黑色?(2026-08-16)

黑区像素恒为 RGB(10,20,31) = Genesis 默认背景色 `background_color=(0.04,0.08,0.12)`,
深度图无命中——**不是材质/光照问题,是那些方向没有几何体**。GRScenes 场景只包含
房间内部(墙/地板/家具),没有室外、天空和建筑外观;场景里的 "HDR_Sphere" 实测
只有 1.7m 直径,不是天空盒。相机朝房间开放方向看就会看到背景色。

修复:调亮背景 + 环境光(`grscene_lightbg.png` 验证,暗像素 69.9%→0.6%):

```python
scene = gs.Scene(
    show_viewer=False,
    vis_options=gs.options.VisOptions(
        background_color=(0.75, 0.78, 0.82),
        ambient_light=(0.45, 0.45, 0.45),
    ),
)
```

## GRScenes navigation 碰撞物理 + R2 机器人入场景(2026-08-16)

**结论**: 掉落探针 z=0.1(落地), R2 站立 base z=0.002(与平面基准完全一致),
差速驱动方向正确——navigation.usd 碰撞物理可用。

### 加载配方 (verify/grscene_robot_test.py)

```python
# 视觉: 上色后的 navigation (collision=False)
# 碰撞: urdf_tools/grscene_nav_collision.py 提取的 *_collision.usd (r8后缀已取消——本数据集全部模型都在住宅范围内, 半径过滤无效果, 两个文件内容一致)
scene.add_entity(
    morph=gs.morphs.USD(file=COL, collision=True, visualization=False, fixed=True,
        decimate=True, decimate_face_num=300,
        decompose_object_error_threshold=float('inf')),   # 跳过 CoACD(慢)
    material=gs.materials.Rigid(sdf_cell_size=0.25, friction=1.0))
# RigidOptions: friction_cone=elliptic + impratio=100 + signorini + use_gjk_collision=True
```

### 踩坑记录(重要)

1. **Genesis 空mesh丢子树 bug(已打补丁)**: `usd_geometry.py` 对无顶点 mesh
   `warning+return`, 但 GRScenes 的空壳 mesh(SM_xx, 1391个)是真实几何的父节点,
   return 连子树遍历一起跳过 → 2697 个 mesh 只导入 1183 个, **地板/墙体全部
   丢失**(球穿地、渲染大片黑)。补丁: 空mesh时先遍历子节点再 return
   (site-packages/genesis/utils/usd/usd_geometry.py, 已改)。升级 Genesis 需重打。
2. **SDF 爆炸**: 898 个家具 mesh 在 convexify=True 下仍非凸, build 逐个建
   0.1m 网格 SDF, 总格数 14.8 亿 → `sdf_cell_size=0.25` 降到 ~1/16。
3. **CoACD 巨慢**: 默认阈值 0.15 会对几百个凹网格做凸分解(25min+ 未完成),
   `decompose_object_error_threshold=inf` 跳过(单凸包, 非凸走 SDF)。
4. **运行资源(共享服务器)**: 必须在 import 前
   `PXR_WORK_THREAD_LIMIT=8, OPENBLAS/OMP/MKL/NUMEXPR_NUM_THREADS=4`,
   并 `taskset -c 0-7` —— 否则 112 核起满线程池撞 ulimit -u 1024,
   线程池半建死锁(build 阶段全 futex 挂起)。GPU 用 `CUDA_VISIBLE_DEVICES=1`
   (GPU0 常被其他用户 vLLM 占 ~76GB)。
5. navigation 变体结构: 515 个模型全部带 CollisionAPI(含375个空Instance mesh),
   无独立简化碰撞体(碰撞=全精度网格); `add_stage` 走不通(无 RigidBodyAPI)。
6. Genesis 1.3.2 API: 摩擦参数在 `gs.options.RigidOptions`(非 SimOptions),
   枚举传对象 `gs.friction_cone.elliptic`; 实体必须 build 前全部加入;
   `get_pos()` n_envs=0 时一维, GPU 张量先 .cpu()。

## 机器人"身体到地面"排查(2026-08-16, 已修复)

**问题**: 渲染图里机器人身体塌到地面。根因两个, 都在机器人控制侧, 场景碰撞无问题:

1. **姿态关节没保持**: 平面配方(natural_pose_check2)要求非 wheel 全部关节 kp=5000/kv=200
   且每步 set_dofs_position(0) 锁姿态; 早期脚本只设了 J2 → 臂/腰重力下塌拖地
   (这也解释了"驱动慢且改摩擦无效"——是身体拖地不是打滑)。
2. **FREE 根关节被误锁 → 基座瞬移世界原点**: 锁姿态时必须排除根关节;
   `j.type` 是**整数枚举**(4=FREE), 用 `'FREE' in str(j.type)` 判断无效
   (str 是 '4'), 必须写 `j.type == gs.JOINT_TYPE.FREE`。误锁后每步
   set_dofs_position(0) 把基座位姿命令到 (0,0,0), 机器人瞬移到原点餐桌区。
   平面测试从未暴露: 出生点恰在原点。
   判别方法: 同点位放一个无控制 box, box 不动=场景地板无倾斜, 是机器人侧问题。

修复后(verify/grscene_robot_test.py): base 稳定在出生点 (1.489,1.511,0.002),
pitch -0.4°, 姿态关节偏移 0.089 rad, 探针球 z=0.100。
**遗留**: 驱动 0.057m/2s(平面 0.718m)——轮地牵引不足, `set_friction(1.0)` 后 build
已确认执行(2697 geom)但无改善, 怀疑 USD PhysicsMaterial 绑定优先级仍高或 SDF 地板
接触特性; 建议下一步: 打印 drive 时轮子角速度与接触法向力, 或从碰撞 USD 里删除
physics:material 绑定让 material 级 friction 生效。

## 工作区清理(2026-08-16)

- 删除 `verify/grscene_physics_test.py` (Sphere(r=) 参数 bug + 功能已被
  `grscene_robot_test.py physics-only` 模式覆盖)
- 碰撞 USD 去重: `*_collision_r8.usd` 与全量版内容一致(515模型, 半径过滤无效果),
  统一为 `*_collision.usd`
- 新增调试工具: `diag/joint_type_check.py` (关节类型/dofs 检查, 排查 FREE 根关节
  误入控制集的基座瞬移 bug)、`diag/contact_control_test.py` (接触系统 30s 对照实验)
- 清理 /tmp 本项目调试脚本与日志 61 个
