cmake_minimum_required(VERSION 3.10)
project(active_3dgs)

find_package(catkin REQUIRED COMPONENTS
  rospy mpl_py active_3dgs_msgs cv_bridge geometry_msgs message_filters
  move_base_msgs nav_msgs sensor_msgs std_msgs std_srvs tf tf2_ros visualization_msgs
)

catkin_python_setup()
catkin_package(CATKIN_DEPENDS
  rospy mpl_py active_3dgs_msgs cv_bridge geometry_msgs message_filters
  move_base_msgs nav_msgs sensor_msgs std_msgs std_srvs tf tf2_ros visualization_msgs
)

catkin_install_python(PROGRAMS src/node/active_gs_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch config models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

# The upstream inference module loads YAML and weights relative to its Python
# files. Preserve its complete runtime tree in the install space as well.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/clip_dinoiser/clipdino_gpu.py")
  install(DIRECTORY src/clip_dinoiser/
    DESTINATION ${CATKIN_GLOBAL_PYTHON_DESTINATION}/clip_dinoiser
    PATTERN ".git" EXCLUDE
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.ipynb" EXCLUDE
    PATTERN "assets" EXCLUDE
    PATTERN "tools" EXCLUDE
    REGEX "/clip_dinoiser/datasets(/|$)" EXCLUDE
  )
endif()
