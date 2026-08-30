cmake_minimum_required(VERSION 3.8)
project(interface_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
# 编译的依赖包
find_package(rosidl_default_generators REQUIRED)
 
 
#为接口文件生成源代码,接口名字首字母必须大写
rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/CoreStatus.msg"
    "msg/WheelStatus.msg"
    "msg/SensorStatus.msg"
    "msg/NavStatus.msg"
    "msg/PosStatus.msg"
    "msg/Robotstatus.msg"
    "msg/MoveJoint.msg"
    "msg/MoveWorld.msg"
    "msg/MotFeedback.msg" 
    "msg/DeviceStatus.msg"
    "msg/Setfingerpos.msg"

    "srv/SetFingerPositions.srv"
    "srv/SetHeartBeat.srv"
    "srv/MotControl.srv"
    "srv/AscendControl.srv"
    "srv/HeadControl.srv"
    "srv/WaistControl.srv"
    "srv/GlobalSpeedSet.srv"
    "srv/MoveToJointPositions.srv"
    "srv/MoveToWorldPose.srv"
    "srv/RobotEnableControl.srv"
    "srv/ClearError.srv"
    "srv/ChargingControl.srv"
    "srv/GetMap.srv"
    "srv/Move.srv"
    "srv/SetMode.srv"
    "srv/NavControl.srv"
    "srv/Relocate.srv"
    "srv/ProtectControl.srv"
    DEPENDENCIES
)
 
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
