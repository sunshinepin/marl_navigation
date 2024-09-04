from visualization_msgs.msg import Marker, MarkerArray

GOAL_REACHED_DIST = 0.3

def publish_markers(index, action, goal_positions, publishers, publishers2, publishers3, car_names):
    markerArray = MarkerArray()
    marker = Marker()
    marker.header.frame_id = "world"
    marker.type = marker.CYLINDER
    marker.action = marker.ADD
    marker.scale.x = GOAL_REACHED_DIST * 2
    marker.scale.y = GOAL_REACHED_DIST * 2
    marker.scale.z = 0.01
    marker.color.a = 1.0

    # 处理四辆车的目标点颜色
    if car_names[index] == "car1":
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
    elif car_names[index] == "car2":
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
    elif car_names[index] == "car3":
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
    elif car_names[index] == "car4":
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0

    marker.pose.orientation.w = 1.0
    marker.pose.position.x = goal_positions[index][0]
    marker.pose.position.y = goal_positions[index][1]
    marker.pose.position.z = 0
    markerArray.markers.append(marker)
    publishers[index].publish(markerArray)

    markerArray2 = MarkerArray()
    marker2 = Marker()
    marker2.header.frame_id = "world"
    marker2.type = marker.CUBE
    marker2.action = marker.ADD
    marker2.scale.x = abs(float(action[0]))  # 确保是浮点数
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
    markerArray2.markers.append(marker2)
    publishers2[index].publish(markerArray2)

    markerArray3 = MarkerArray()
    marker3 = Marker()
    marker3.header.frame_id = "world"
    marker3.type = marker.CUBE
    marker3.action = marker.ADD
    marker3.scale.x = abs(float(action[1]))  # 确保是浮点数
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
    markerArray3.markers.append(marker3)
    publishers3[index].publish(markerArray3)
