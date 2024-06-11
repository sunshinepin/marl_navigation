#!/usr/bin/env python
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from gazebo_msgs.msg import *
from gazebo_msgs.srv import *
import rospy
import tf2_ros

rospy.init_node('publish_odom_from_gazebo', anonymous=True)

car_name = rospy.get_param('~car_name')
topic_name = '/' + car_name + '/odom_gazebo'
odom_frame_id = car_name + "/odom_gazebo"  # 新的odom帧ID
root_relative_entity_name = '' # this is the full model and not only the base_link
car_frame_id = car_name + "/base_link"  # car_name的base_link帧

# 初始化发布器和TF广播器
odom_pub = rospy.Publisher(topic_name, Odometry, queue_size=30)
tf_broadcaster = tf2_ros.TransformBroadcaster()
rate = rospy.Rate(30)

while not rospy.is_shutdown():
    rospy.wait_for_service('/gazebo/get_model_state')
    try:
        gms = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        state = gms(car_name, root_relative_entity_name)
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)
        continue

    # 获取当前时间
    current_time = rospy.Time.now()

    # 准备并发布Odometry消息
    odom_msg = Odometry()
    odom_msg.header.stamp = current_time
    odom_msg.header.frame_id = odom_frame_id
    odom_msg.child_frame_id = car_frame_id
    odom_msg.pose.pose = state.pose
    odom_msg.twist.twist = state.twist
    odom_pub.publish(odom_msg)

    # 发布world到odom的转换
    odom_trans = TransformStamped()
    odom_trans.header.stamp = current_time
    odom_trans.header.frame_id = "world"
    odom_trans.child_frame_id = odom_frame_id
    # 假设odom帧初始时与world帧重合
    odom_trans.transform.translation.x = 0
    odom_trans.transform.translation.y = 0
    odom_trans.transform.translation.z = 0
    odom_trans.transform.rotation.x = 0
    odom_trans.transform.rotation.y = 0
    odom_trans.transform.rotation.z = 0
    odom_trans.transform.rotation.w = 1
    tf_broadcaster.sendTransform(odom_trans)

    # 发布odom到base_link的转换
    base_trans = TransformStamped()
    base_trans.header.stamp = current_time
    base_trans.header.frame_id = odom_frame_id
    base_trans.child_frame_id = car_frame_id
    base_trans.transform.translation.x = state.pose.position.x
    base_trans.transform.translation.y = state.pose.position.y
    base_trans.transform.translation.z = state.pose.position.z
    base_trans.transform.rotation = state.pose.orientation
    tf_broadcaster.sendTransform(base_trans)

    rate.sleep()
