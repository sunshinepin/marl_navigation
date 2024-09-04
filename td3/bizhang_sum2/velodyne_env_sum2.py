import math
import os
import random
import time

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.1

def check_pos(x, y):
    goal_ok = True
    if -3.8 > x > -6.2 and 6.2 > y > 3.8:
        goal_ok = False
    if -1.3 > x > -2.7 and 4.7 > y > -0.2:
        goal_ok = False
    if -0.3 > x > -4.2 and 2.7 > y > 1.3:
        goal_ok = False
    if -0.8 > x > -4.2 and -2.3 > y > -4.2:
        goal_ok = False
    if -1.3 > x > -3.7 and -0.8 > y > -2.7:
        goal_ok = False
    if 4.2 > x > 0.8 and -1.8 > y > -3.2:
        goal_ok = False
    if 4 > x > 2.5 and 0.7 > y > -3.2:
        goal_ok = False
    if 6.2 > x > 3.8 and -3.3 > y > -4.2:
        goal_ok = False
    if 4.2 > x > 1.3 and 3.7 > y > 1.5:
        goal_ok = False
    if -3.0 > x > -7.2 and 0.5 > y > -1.5:
        goal_ok = False
    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goal_ok = False
    return goal_ok

class GazeboEnv:
    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim
        self.odom_x = {"car1": 0, "car2": 0}
        self.odom_y = {"car1": 0, "car2": 0}
        self.goal_x = {"car1": 1, "car2": 1}
        self.goal_y = {"car1": 0.0, "car2": 0.0}
        self.upper = 5.0
        self.lower = -5.0
        self.velodyne_data = {"car1": np.ones(environment_dim) * 10, "car2": np.ones(environment_dim) * 10}
        self.last_odom = {"car1": None, "car2": None}

        self.set_self_state = {"car1": ModelState(), "car2": ModelState()}
        self.set_self_state["car1"].model_name = "car1"
        self.set_self_state["car1"].pose.position.x = 0.0
        self.set_self_state["car1"].pose.position.y = 0.0
        self.set_self_state["car1"].pose.position.z = 0.0
        self.set_self_state["car1"].pose.orientation.x = 0.0
        self.set_self_state["car1"].pose.orientation.y = 0.0
        self.set_self_state["car1"].pose.orientation.z = 0.0
        self.set_self_state["car1"].pose.orientation.w = 1.0

        self.set_self_state["car2"].model_name = "car2"
        self.set_self_state["car2"].pose.position.x = 0.0
        self.set_self_state["car2"].pose.position.y = 0.0
        self.set_self_state["car2"].pose.position.z = 0.0
        self.set_self_state["car2"].pose.orientation.x = 0.0
        self.set_self_state["car2"].pose.orientation.y = 0.0
        self.set_self_state["car2"].pose.orientation.z = 0.0
        self.set_self_state["car2"].pose.orientation.w = 1.0

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / environment_dim]]
        for m in range(environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / environment_dim]
            )
        self.gaps[-1][-1] += 0.03

        print("Roscore launched!")
        rospy.init_node("gym", anonymous=True)

        self.vel_pub = {"car1": rospy.Publisher("/car1/cmd_vel", Twist, queue_size=1),
                        "car2": rospy.Publisher("/car2/cmd_vel", Twist, queue_size=1)}
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = {"car1": rospy.Publisher("car1/goal_point", MarkerArray, queue_size=3),
                          "car2": rospy.Publisher("car2/goal_point", MarkerArray, queue_size=3)}
        self.publisher2 = {"car1": rospy.Publisher("car1/linear_velocity", MarkerArray, queue_size=1),
                           "car2": rospy.Publisher("car2/linear_velocity", MarkerArray, queue_size=1)}
        self.publisher3 = {"car1": rospy.Publisher("car1/angular_velocity", MarkerArray, queue_size=1),
                           "car2": rospy.Publisher("car2/angular_velocity", MarkerArray, queue_size=1)}
        self.velodyne = {"car1": rospy.Subscriber(
            "/car1/velodyne/velodyne_points", PointCloud2, self.velodyne_callback_car1, queue_size=1),
            "car2": rospy.Subscriber(
                "/car2/velodyne/velodyne_points", PointCloud2, self.velodyne_callback_car2, queue_size=1)}
        self.odom = {"car1": rospy.Subscriber(
            "/car1/odom_gazebo", Odometry, self.odom_callback_car1, queue_size=1),
            "car2": rospy.Subscriber(
                "/car2/odom_gazebo", Odometry, self.odom_callback_car2, queue_size=1)}
        self.start_point = {"car1": None, "car2": None}
        self.goal_point = {"car1": None, "car2": None}
        self.trajectory_file = {"car1": "trajectory_car1.txt", "car2": "trajectory_car2.txt"}
        self.clear_trajectory_file()
        self.trajectory_pub = {"car1": rospy.Publisher("car1/trajectory", Marker, queue_size=10),
                               "car2": rospy.Publisher("car2/trajectory", Marker, queue_size=10)}
        self.trajectory = {"car1": Marker(), "car2": Marker()}
        for car in ["car1", "car2"]:
            self.trajectory[car].header.frame_id = "world"
            self.trajectory[car].type = Marker.LINE_STRIP
            self.trajectory[car].action = Marker.ADD
            self.trajectory[car].scale.x = 0.05
            self.trajectory[car].color.a = 1.0
            self.trajectory[car].color.r = 0.0
            self.trajectory[car].color.g = 0.0
            self.trajectory[car].color.b = 1.0
            self.trajectory[car].points = []

    def clear_trajectory_file(self):
        for car in ["car1", "car2"]:
            with open(self.trajectory_file[car], "w") as file:
                file.write("")

    def record_trajectory(self, car):
        with open(self.trajectory_file[car], "a") as file:
            file.write(f"{self.odom_x[car]},{self.odom_y[car]}\n")

    def set_start_goal(self, car, start, goal):
        self.start_point[car] = start
        self.goal_point[car] = goal

    def velodyne_callback_car1(self, v):
        self.velodyne_callback(v, "car1")

    def velodyne_callback_car2(self, v):
        self.velodyne_callback(v, "car2")

    def velodyne_callback(self, v, car):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data[car] = np.ones(self.environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)
                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[car][j] = min(self.velodyne_data[car][j], dist)
                        break

    def odom_callback_car1(self, od_data):
        self.odom_callback(od_data, "car1")

    def odom_callback_car2(self, od_data):
        self.odom_callback(od_data, "car2")

    def odom_callback(self, od_data, car):
        self.last_odom[car] = od_data

    def record_trajectory(self, car):
        point = Point()
        point.x = self.odom_x[car]
        point.y = self.odom_y[car]
        point.z = 0
        self.trajectory[car].points.append(point)
        self.trajectory[car].header.stamp = rospy.Time.now()
        self.trajectory_pub[car].publish(self.trajectory[car])

    def step(self, car, action):
        target = False
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub[car].publish(vel_cmd)
        self.publish_markers(car, action)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        done, collision, min_laser = self.observe_collision(self.velodyne_data[car])
        v_state = []
        v_state[:] = self.velodyne_data[car][:]
        laser_state = [v_state]

        self.odom_x[car] = self.last_odom[car].pose.pose.position.x
        self.odom_y[car] = self.last_odom[car].pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom[car].pose.pose.orientation.w,
            self.last_odom[car].pose.pose.orientation.x,
            self.last_odom[car].pose.pose.orientation.y,
            self.last_odom[car].pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        distance = np.linalg.norm(
            [self.odom_x[car] - self.goal_x[car], self.odom_y[car] - self.goal_y[car]]
        )
        skew_x = self.goal_x[car] - self.odom_x[car]
        skew_y = self.goal_y[car] - self.odom_y[car]
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        if distance < GOAL_REACHED_DIST:
            target = True
            done = True

        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        reward = self.get_reward(target, collision, action, min_laser)

        self.record_trajectory(car)

        return state, reward, done, target

    def reset(self, car):
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        if self.start_point[car]:
            self.set_self_state[car].pose.position.x = self.start_point[car][0]
            self.set_self_state[car].pose.position.y = self.start_point[car][1]

        self.set_state.publish(self.set_self_state[car])
        self.odom_x[car] = self.set_self_state[car].pose.position.x
        self.odom_y[car] = self.set_self_state[car].pose.position.y

        if self.goal_point[car]:
            self.goal_x[car] = self.goal_point[car][0]
            self.goal_y[car] = self.goal_point[car][1]

        self.publish_markers(car, [0.0, 0.0])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        v_state = []
        v_state[:] = self.velodyne_data[car][:]
        laser_state = [v_state]

        distance = np.linalg.norm([self.odom_x[car] - self.goal_x[car], self.odom_y[car] - self.goal_y[car]])
        skew_x = self.goal_x[car] - self.odom_x[car]
        skew_y = self.goal_y[car] - self.odom_y[car]
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)

        self.trajectory[car].points = []
        self.record_trajectory(car)

        return state

    def change_goal(self, car):
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        goal_ok = False
        while not goal_ok:
            self.goal_x[car] = self.odom_x[car] + random.uniform(self.upper, self.lower)
            self.goal_y[car] = self.odom_y[car] + random.uniform(self.upper, self.lower)
            goal_ok = check_pos(self.goal_x[car], self.goal_y[car])

    def random_box(self):
        for i in range(4):
            name = "cardboard_box_" + str(i)
            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x["car1"], y - self.odom_y["car1"]])
                distance_to_goal = np.linalg.norm([x - self.goal_x["car1"], y - self.goal_y["car1"]])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    def publish_markers(self, car, action):
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "world"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x[car]
        marker.pose.position.y = self.goal_y[car]
        marker.pose.position.z = 0
        markerArray.markers.append(marker)
        self.publisher[car].publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "world"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
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
        self.publisher2[car].publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "world"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
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
        self.publisher3[car].publish(markerArray3)

    @staticmethod
    def observe_collision(laser_data):
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser):
        if target:
            return 100.0
        elif collision:
            return -100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2
