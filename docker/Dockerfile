FROM nvidia/cuda:12.4.1-devel-ubuntu20.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    ROS_DISTRO=noetic CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9+PTX" MAX_JOBS=2 \
    PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg git build-essential python3-dev python3-pip \
      python3-venv libgl1 libglib2.0-0 locales && \
    locale-gen en_US.UTF-8 && \
    curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu focal main" \
      > /etc/apt/sources.list.d/ros1.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      ros-noetic-ros-base ros-noetic-cv-bridge ros-noetic-tf ros-noetic-tf2-ros \
      ros-noetic-move-base-msgs ros-noetic-rviz ros-noetic-pcl-ros \
      ros-noetic-tf-conversions ros-noetic-eigen-conversions ros-noetic-cmake-modules \
      libeigen3-dev libopencv-dev libyaml-cpp-dev libsdl1.2-dev libsdl-image1.2-dev \
      python3-rosdep python3-catkin-tools python3-rtree python3-parse && \
    rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# Keep Ubuntu's Python 3.8 so binary ROS Python modules (cv_bridge/tf) match.
RUN python3 -m pip install --upgrade 'pip<25.1' 'setuptools<70' wheel 'empy==3.3.4' && \
    python3 -m pip install torch==2.4.1 torchvision==0.19.1 \
      --index-url https://download.pytorch.org/whl/cu124
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

# Dependencies come from the host's initialized, pinned Git submodules.
COPY mpl_py /opt/catkin_ws/src/mpl_py
COPY diff-gaussian-rasterization-w-depth.git /opt/rasterizer-depth
COPY diff-gaussian-rasterization-multi /opt/rasterizer-multi
RUN for tree in /opt/rasterizer-depth /opt/rasterizer-multi; do \
      test -f "$tree/third_party/glm/glm/glm.hpp" || \
        { echo "Initialize recursive submodules before building (see README)."; exit 1; }; \
    done && \
    python3 -m pip install --no-build-isolation /opt/rasterizer-depth && \
    python3 -m pip install --no-build-isolation /opt/rasterizer-multi

# Build the external C++ tracker/planner before copying Python and config files,
# so pipeline edits do not rebuild unchanged external dependencies.
WORKDIR /opt/catkin_ws
RUN source /opt/ros/noetic/setup.bash && \
    catkin config --extend /opt/ros/noetic --cmake-args \
      -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 && \
    catkin build jackal_mp_tracker --no-status -j2 -p1
COPY active_3dgs /opt/catkin_ws/src/SplaTAM/active_3dgs
COPY active_3dgs_msgs /opt/catkin_ws/src/SplaTAM/active_3dgs_msgs
# Catkin install uses Ubuntu distutils' --install-layout=deb extension.
ENV SETUPTOOLS_USE_DISTUTILS=stdlib
RUN test -f /opt/catkin_ws/src/SplaTAM/active_3dgs/src/clip_dinoiser/checkpoints/last.pt || \
    { echo "Initialize recursive submodules before building (see README)."; exit 1; }
COPY requirements-language.txt /tmp/requirements-language.txt
RUN python3 -m pip install -r /tmp/requirements-language.txt && \
    python3 -m pip check && \
    source /opt/ros/noetic/setup.bash && \
    catkin build active_3dgs --no-status -j2 -p1

COPY LICENSE /opt/catkin_ws/src/SplaTAM/LICENSE
COPY docker/entrypoint.sh /usr/local/bin/active-3dgs-entrypoint
RUN chmod +x /usr/local/bin/active-3dgs-entrypoint
ENTRYPOINT ["/usr/local/bin/active-3dgs-entrypoint"]
CMD ["roslaunch", "active_3dgs", "active_3dgs_robot.launch"]
