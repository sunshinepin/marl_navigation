#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

class RvizVisualizer:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('rviz_visualizer', anonymous=True)
        
        # 初始化发布者
        self.publishers = {}
        self.car_names = ["car1", "car2", "car3"]  # 可以根据需要更改
        for car_name in self.car_names:
            self.publishers[car_name] = rospy.Publisher(f'/{car_name}/goal_marker', MarkerArray, queue_size=10)

    def publish_goal_marker(self, car_name, goal_position, action):
        """
        发布目标位置和动作的Marker
        car_name: 智能体名称
        goal_position: 目标位置 [x, y]
        action: [线性速度, 角速度]
        """
        # 创建目标Marker
        markerArray = MarkerArray()

        marker = Marker()
        marker.header.frame_id = "world"
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.scale.x = 0.6  # 目标Marker的半径
        marker.scale.y = 0.6
        marker.scale.z = 0.01  # 高度
        marker.color.a = 1.0

        # 根据车的名字设置颜色
        if car_name == "car1":
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        elif car_name == "car2":
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
        elif car_name == "car3":
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0

        marker.pose.orientation.w = 1.0
        marker.pose.position.x = goal_position[0]
        marker.pose.position.y = goal_position[1]
        marker.pose.position.z = 0  # 放置在地面上

        markerArray.markers.append(marker)

        # 发布Marker到Rviz
        self.publishers[car_name].publish(markerArray)

        # 可视化动作
        # 线性速度（action[0]）
        marker2 = Marker()
        marker2.header.frame_id = "world"
        marker2.type = Marker.CUBE
        marker2.action = Marker.ADD
        marker2.scale.x = abs(action[0])  # 根据action[0]设置Marker的大小
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0
        markerArray.markers.append(marker2)

        # 角速度（action[1]）
        marker3 = Marker()
        marker3.header.frame_id = "world"
        marker3.type = Marker.CUBE
        marker3.action = Marker.ADD
        marker3.scale.x = abs(action[1])  # 根据action[1]设置Marker的大小
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0
        markerArray.markers.append(marker3)

        # 发布动作Marker
        self.publishers[car_name].publish(markerArray)

if __name__ == '__main__':
    # 创建Rviz可视化对象
    visualizer = RvizVisualizer()

    # 使用ROS的循环来不断发布数据
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
