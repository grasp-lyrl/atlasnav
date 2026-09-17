#!/usr/bin/env bash
set -e
source /opt/ros/noetic/setup.bash
source /opt/catkin_ws/devel/setup.bash
exec "$@"
