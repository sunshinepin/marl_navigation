import math
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
TIME_DELTA = 0.1
MAX_STEPS = 500
STAY_REWARD = 5.0  # 每步留在目标点的奖励
COLLISION_DIST = 0.35
COLLISION_PENALTY = -100.0  # 碰撞惩罚

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
    def __init__(self, launchfile, environment_dim, car_name, car_x, car_y, car_z):
        self.environment_dim = environment_dim
        self.car_name = car_name
        self.car_x = car_x
        self.car_y = car_y
        self.car_z = car_z
        self.odom_x = 0
        self.odom_y = 0
        self.goal_x = 1
        self.goal_y = 0.0
        self.upper = 5.0
        self.lower = -5.0
        self.velodyne_data = np.ones(environment_dim) * 10
        self.last_odom = None
        self.steps = 0
        self.target_reached = False

        self.set_self_state = ModelState()
        self.set_self_state.model_name = self.car_name
        self.set_self_state.pose.position.x = self.car_x
        self.set_self_state.pose.position.y = self.car_y
        self.set_self_state.pose.position.z = self.car_z
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / environment_dim]]
        for m in range(environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / environment_dim]
            )
        self.gaps[-1][-1] += 0.03

        print("Roscore launched!")
        rospy.init_node("gym", anonymous=True)

        self.vel_pub = rospy.Publisher(f"/{self.car_name}/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher(f"{self.car_name}/goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher(f"{self.car_name}/linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher(f"{self.car_name}/angular_velocity", MarkerArray, queue_size=1)
        self.velodyne = rospy.Subscriber(
            f"/{self.car_name}/velodyne/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1)
        self.odom = rospy.Subscriber(
            f"/{self.car_name}/odom_gazebo", Odometry, self.odom_callback, queue_size=1)
        self.trajectory_file = f"trajectory_{self.car_name}.txt"
        self.clear_trajectory_file()
        self.trajectory_pub = rospy.Publisher(f"{self.car_name}/trajectory", Marker, queue_size=10)
        self.trajectory = Marker()
        self.trajectory.header.frame_id = "world"
        self.trajectory.type = Marker.LINE_STRIP
        self.trajectory.action = Marker.ADD
        self.trajectory.scale.x = 0.05
        self.trajectory.color.a = 1.0
        self.trajectory.color.r = 0.0
        self.trajectory.color.g = 0.0
        self.trajectory.color.b = 1.0
        self.trajectory.points = []

    def clear_trajectory_file(self):
        with open(self.trajectory_file, "w") as file:
            file.write("")

    def record_trajectory(self):
        with open(self.trajectory_file, "a") as file:
            file.write(f"{self.odom_x},{self.odom_y}\n")

    def velodyne_callback(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(self.environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)
                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break

    def odom_callback(self, od_data):
        self.last_odom = od_data

    def record_trajectory(self):
        point = Point()
        point.x = self.odom_x
        point.y = self.odom_y
        point.z = 0
        self.trajectory.points.append(point)
        self.trajectory.header.stamp = rospy.Time.now()
        self.trajectory_pub.publish(self.trajectory)

    def step(self, action):
        self.steps += 1
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

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

        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
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
            self.target_reached = True

        if self.steps >= MAX_STEPS or collision:
            done = True
        else:
            done = False

        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        reward = self.get_reward(self.target_reached, collision, action, distance)

        self.record_trajectory()

        return state, reward, done, self.target_reached

    def reset(self):
        self.steps = 0
        self.target_reached = False
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        self.set_self_state.pose.position.x = self.car_x
        self.set_self_state.pose.position.y = self.car_y
        self.set_self_state.pose.position.z = self.car_z
        self.set_self_state.pose.orientation.z = math.sin(angle / 2.0)
        self.set_self_state.pose.orientation.w = math.cos(angle / 2.0)

        self.set_state.publish(self.set_self_state)
        self.odom_x = self.set_self_state.pose.position.x
        self.odom_y = self.set_self_state.pose.position.y

        self.goal_x = np.random.uniform(self.lower, self.upper)
        self.goal_y = np.random.uniform(self.lower, self.upper)
        while not check_pos(self.goal_x, self.goal_y):
            self.goal_x = np.random.uniform(self.lower, self.upper)
            self.goal_y = np.random.uniform(self.lower, self.upper)

        self.publish_markers([0.0, 0.0])

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
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
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

        self.trajectory.points = []
        self.record_trajectory()

        return state

    def publish_markers(self, action):
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "world"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = GOAL_REACHED_DIST * 2
        marker.scale.y = GOAL_REACHED_DIST * 2
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 1.0 if self.car_name == "car1" else 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0 if self.car_name == "car1" else 1.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0
        markerArray.markers.append(marker)
        self.publisher.publish(markerArray)

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
        self.publisher2.publish(markerArray2)

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
        self.publisher3.publish(markerArray3)

    @staticmethod
    def observe_collision(laser_data):
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, distance):
        reward = action[0] / 2 - abs(action[1]) / 2 - (1 - distance) / 2
        if target:
            reward += STAY_REWARD
        if collision:
            reward += COLLISION_PENALTY
        return reward
