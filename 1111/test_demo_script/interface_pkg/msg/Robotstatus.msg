bool        is_connected                # 与机器人控制器的连接状态 (True: 已连接, False: 断开连接)
bool        is_alarming                 # 机器人当前是否处于报警状态 (True: 报警中, False: 正常)
string[]    alarm_messages              # 详细的报警信息列表 (字符串数组)，当alarming为True时包含具体信息
bool        is_enabled                  # 机器人的使能状态 (True: 已使能, False: 未使能/禁用)
float32     global_speed_ratio          # 机器人当前的全局速度百分比 (%) 或比例因子

bool        world_move_complete         # 世界坐标系下的运动是否完成标志 (True: 运动完成, False: 运动中)
bool        joint_move_complete         # 关节空间下的运动是否完成标志 (True: 运动完成, False: 运动中)

float32[]   left_joint_positions        # 机器人各关节的实时反馈角度/位置 (单位: 度 或 弧度，需明确)，数组长度对应关节数
float32[]   left_tcp_pose               # 机器人末端在世界坐标系下的实时反馈位姿 (XYZ-位置, UVW-欧拉角姿态)，格式: [X, Y, Z, U, V, W]

float32[]   right_joint_positions       # 机器人各关节的实时反馈角度/位置 (单位: 度 或 弧度，需明确)，数组长度对应关节数
float32[]   right_tcp_pose              # 机器人末端在世界坐标系下的实时反馈位姿 (XYZ-位置, UVW-欧拉角姿态)，格式: [X, Y, Z, U, V, W]