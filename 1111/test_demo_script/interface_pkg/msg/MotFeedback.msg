float32 angle       # 关节角度（度）
uint32 pos         # 位置（单位：mm） 
uint16 waist_err    # 腰部模块错误码 0正常 非零有错误
uint16 ascend_err   # 升降模块错误码
uint8 waist_ready   # 腰部到位状态 1-到位 2-没到位
uint8 ascend_ready  # 升降到位状态 1-到位 2-没到位
uint8 waist_temp    # 腰部温度（℃）
uint8 ascend_temp   # 升降温度（℃）
uint16 waist_current #电流单位mA
uint16 ascend_current #升降电流mA
float32 head_back_angle #头部角度（度）deg
uint8 head_flag #头部到位标志位