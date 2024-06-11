#!/usr/bin/env python

import rospy
import tf
import tf2_ros
from geometry_msgs.msg import TransformStamped

def broadcast_transforms(ns):
    br = tf2_ros.TransformBroadcaster()
    rate = rospy.Rate(10.0)

    while not rospy.is_shutdown():
        # world to odom transform
        t_world_odom = TransformStamped()
        t_world_odom.header.stamp = rospy.Time.now()
        t_world_odom.header.frame_id = 'world'
        t_world_odom.child_frame_id = ns + '/odom'
        t_world_odom.transform.translation.x = 0.0
        t_world_odom.transform.translation.y = 0.0
        t_world_odom.transform.translation.z = 0.0
        t_world_odom.transform.rotation.x = 0.0
        t_world_odom.transform.rotation.y = 0.0
        t_world_odom.transform.rotation.z = 0.0
        t_world_odom.transform.rotation.w = 1.0

        # odom to base_link transform
        t_odom_base_link = TransformStamped()
        t_odom_base_link.header.stamp = rospy.Time.now()
        t_odom_base_link.header.frame_id = ns + '/odom'
        t_odom_base_link.child_frame_id = ns + '/base_link'
        t_odom_base_link.transform.translation.x = 0.0
        t_odom_base_link.transform.translation.y = 0.0
        t_odom_base_link.transform.translation.z = 0.0
        t_odom_base_link.transform.rotation.x = 0.0
        t_odom_base_link.transform.rotation.y = 0.0
        t_odom_base_link.transform.rotation.z = 0.0
        t_odom_base_link.transform.rotation.w = 1.0

        # Send the transforms
        br.sendTransform(t_world_odom)
        br.sendTransform(t_odom_base_link)

        rate.sleep()

if __name__ == '__main__':
    rospy.init_node('tf_broadcaster')
    namespace = rospy.get_param('~namespace', '')
    broadcast_transforms(namespace)
    rospy.spin()
