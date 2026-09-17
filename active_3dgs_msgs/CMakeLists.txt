cmake_minimum_required(VERSION 3.10)
project(active_3dgs_msgs)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  message_generation
)

add_message_files(
  FILES
  PointWithSize.msg
  GaussianPoints.msg
)


add_service_files(
  FILES
  SetTask.srv
)


generate_messages(
  DEPENDENCIES
  std_msgs
)


catkin_package(
  CATKIN_DEPENDS 
  std_msgs 
  message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
