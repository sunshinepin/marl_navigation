#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from gazebo_msgs.srv import GetModelState

def publish_odom():
    rospy.init_node('publish_odom_from_gazebo', anonymous=True)
    
    # Get parameters from the launch file
    car_name = rospy.get_param('~car_name')
    topic_name = '/' + car_name + '/odom_gazebo'
    ref_frame_id = "world"
    root_relative_entity_name = ''  # Full model, not just base_link
    car_frame_id = '/' + car_name + '/' + 'base_link'
    
    # Publisher for odometry
    odom_pub = rospy.Publisher(topic_name, Odometry, queue_size=30)
    rate = rospy.Rate(30)
    
    while not rospy.is_shutdown():
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            gms = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            msg = gms(car_name, root_relative_entity_name)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
            continue
        
        # Create and publish the Odometry message
        send_Od_data = Odometry()
        send_Od_data.header.frame_id = ref_frame_id
        send_Od_data.header.stamp = rospy.Time.now()
        send_Od_data.child_frame_id = car_frame_id
        send_Od_data.pose.pose = msg.pose
        send_Od_data.twist.twist = msg.twist
        odom_pub.publish(send_Od_data)
        
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_odom()
    except rospy.ROSInterruptException:
        pass
