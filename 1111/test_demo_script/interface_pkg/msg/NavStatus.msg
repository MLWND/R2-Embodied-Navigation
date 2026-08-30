uint8 state      # 0=初始, 1=导航中, 2=暂停, 3=完成, 4=取消, 5=恢复, 6=指令接收成功
int8 code        # 状态码：0=成功, 负值=错误
float32 x        # 当前目标点x坐标
float32 y        # 当前目标点y坐标
float32 radian   # 当前朝向(弧度)
float32 dist_to_goal  # 距目标点距离
float32 mileage   # 累计导航里程
uint8 mode       # 1=导航模式, 2=建图模式, 3=增量建图