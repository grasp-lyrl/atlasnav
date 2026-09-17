#!/usr/bin/env python3
import rospy
from core.active_gs import ActiveGS

if __name__ == "__main__":
    rospy.init_node("active_gs", anonymous=True)
    gs_node = ActiveGS()
    rospy.spin()
