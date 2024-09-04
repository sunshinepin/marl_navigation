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
COLLISION_DIST = 0.35
COLLISION_PENALTY = -100.0  # 碰撞惩罚
GOAL_REWARD = 100.0  # 达到目标的奖励
MIN_GOAL_DISTANCE = 1.0  # 目标点之间的最小距离

def check_pos(x, y):
    goal_ok = True

    if 1.5 < x < 4.5 and 1.5 < y < 4.5:
        goal_ok = False
    if -5 < x < -0.5 and 4.5 > y > 1.5:
        goal_ok = False
    if -5.5 < x < -2.5 and 5 > y > 0.5:
        goal_ok = False
    if 0.5 < x < 5 and -5.5 < y < -2.5:
        goal_ok = False
    if 2.5 < x < 5.5 and -5 < y < -0.5:
        goal_ok = False
    if -4.5 < x < -1.5 and -4.5 < y < -1.5:
        goal_ok = False
    if -7.5 < x < -5.5 and 5.5 < y < 7.5:
        goal_ok = False
    if -5.5 < x < -4.5 and 3.0 < y < 4.0:
        goal_ok = False
    if -5.5 < x < -4.5 and -7.0 < y < -6.0:
        goal_ok = False
    if 4.5 < x < 5.5 and 5.0 < y < 6.0:
        goal_ok = False
    if 5.5 < x < 6.5 and -6.5 < y < -5.5:
        goal_ok = False

    if x > 6.5 or x < -6.5 or y > 6.5 or y < -6.5:
        goal_ok = False

    return goal_ok


def generate_unique_goal(existing_goals, lower, upper):
    while True:
        goal_x = np.random.uniform(lower, upper)
        goal_y = np.random.uniform(lower, upper)
        if check_pos(goal_x, goal_y) and all(
            np.linalg.norm([goal_x - gx, goal_y - gy]) > MIN_GOAL_DISTANCE for gx, gy in existing_goals
        ):
            return goal_x, goal_y

class GazeboEnv:
    def __init__(self, launchfile, environment_dim, car_names, car_positions):
        self.environment_dim = environment_dim
        self.car_names = car_names
        self.car_positions = car_positions
        self.odom_positions = [[0, 0] for _ in range(len(car_names))]
        self.goal_positions = [[1, 0.0] for _ in range(len(car_names))]
        self.upper = 5.0
        self.lower = -5.0
        self.velodyne_data = [np.ones(environment_dim) * 10 for _ in range(len(car_names))]
        self.last_odom = [None for _ in range(len(car_names))]
        self.steps = 0
        self.target_reached = [False for _ in range(len(car_names))]
        self.goal_reward_given = [False for _ in range(len(car_names))]

        self.set_self_states = []
        for i, car_name in enumerate(car_names):
            state = ModelState()
            state.model_name = car_name
            state.pose.position.x = car_positions[i][0]
            state.pose.position.y = car_positions[i][1]
            state.pose.position.z = car_positions[i][2]
            state.pose.orientation.x = 0.0
            state.pose.orientation.y = 0.0
            state.pose.orientation.z = 0.0
            state.pose.orientation.w = 1.0
            self.set_self_states.append(state)

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / environment_dim]]
        for m in range(environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / environment_dim]
            )
        self.gaps[-1][-1] += 0.03

        print("Roscore launched!")
        rospy.init_node("gym", anonymous=True)

        self.vel_pubs = []
        self.set_states = []
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publishers = []
        self.publishers2 = []
        self.publishers3 = []
        self.velodynes = []
        self.odom = []
        self.trajectory_files = []
        self.trajectory_pubs = []
        self.trajectories = []

        for i, car_name in enumerate(car_names):
            self.vel_pubs.append(rospy.Publisher(f"/{car_name}/cmd_vel", Twist, queue_size=1))
            self.set_states.append(rospy.Publisher(
                "gazebo/set_model_state", ModelState, queue_size=10
            ))
            self.publishers.append(rospy.Publisher(f"{car_name}/goal_point", MarkerArray, queue_size=3))
            self.publishers2.append(rospy.Publisher(f"{car_name}/linear_velocity", MarkerArray, queue_size=1))
            self.publishers3.append(rospy.Publisher(f"{car_name}/angular_velocity", MarkerArray, queue_size=1))
            self.velodynes.append(rospy.Subscriber(
                f"/{car_name}/velodyne/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1, callback_args=i))
            self.odom.append(rospy.Subscriber(
                f"/{car_name}/odom_gazebo", Odometry, self.odom_callback, queue_size=1, callback_args=i))
            self.trajectory_files.append(f"trajectory_{car_name}.txt")
            self.clear_trajectory_file(i)
            self.trajectory_pubs.append(rospy.Publisher(f"{car_name}/trajectory", Marker, queue_size=10))
            trajectory = Marker()
            trajectory.header.frame_id = "world"
            trajectory.type = Marker.LINE_STRIP
            trajectory.action = Marker.ADD
            trajectory.scale.x = 0.05
            trajectory.color.a = 1.0
            trajectory.color.r = 0.0
            trajectory.color.g = 0.0
            trajectory.color.b = 1.0
            trajectory.points = []
            self.trajectories.append(trajectory)

    def clear_trajectory_file(self, index):
        with open(self.trajectory_files[index], "w") as file:
            file.write("")

    def record_trajectory(self, index):
        with open(self.trajectory_files[index], "a") as file:
            file.write(f"{self.odom_positions[index][0]},{self.odom_positions[index][1]}\n")

    def velodyne_callback(self, v, index):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data[index] = np.ones(self.environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)
                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[index][j] = min(self.velodyne_data[index][j], dist)
                        break

    def odom_callback(self, od_data, index):
        self.last_odom[index] = od_data

    def step(self, actions):
        self.steps += 1
        rewards = []
        dones = []
        target_reached = []

        for i, action in enumerate(actions):
            if not self.target_reached[i]:
                vel_cmd = Twist()
                vel_cmd.linear.x = action[0]
                vel_cmd.angular.z = action[1]
                self.vel_pubs[i].publish(vel_cmd)
                self.publish_markers(i, action)
            else:
                vel_cmd = Twist()
                vel_cmd.linear.x = 0
                vel_cmd.angular.z = 0
                self.vel_pubs[i].publish(vel_cmd)

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

        states = []
        for i in range(len(self.car_names)):
            done, collision, min_laser = self.observe_collision(self.velodyne_data[i])
            v_state = []
            v_state[:] = self.velodyne_data[i][:]
            laser_state = [v_state]

            self.odom_positions[i][0] = self.last_odom[i].pose.pose.position.x
            self.odom_positions[i][1] = self.last_odom[i].pose.pose.position.y
            quaternion = Quaternion(
                self.last_odom[i].pose.pose.orientation.w,
                self.last_odom[i].pose.pose.orientation.x,
                self.last_odom[i].pose.pose.orientation.y,
                self.last_odom[i].pose.pose.orientation.z,
            )
            euler = quaternion.to_euler(degrees=False)
            angle = round(euler[2], 4)

            distance = np.linalg.norm(
                [self.odom_positions[i][0] - self.goal_positions[i][0],
                 self.odom_positions[i][1] - self.goal_positions[i][1]]
            )
            skew_x = self.goal_positions[i][0] - self.odom_positions[i][0]
            skew_y = self.goal_positions[i][1] - self.odom_positions[i][1]
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
                self.target_reached[i] = True

            if self.steps >= MAX_STEPS:
                done = True
            else:
                done = False

            if collision:
                self.reset_car_position(i)

            robot_state = [distance, theta, action[0], action[1]]
            state = np.append(laser_state, robot_state)
            reward = self.get_reward(self.target_reached[i], collision, action, distance)

            self.record_trajectory(i)

            rewards.append(reward)
            dones.append(done)
            target_reached.append(self.target_reached[i])
            states.append(state)

        return states, rewards, dones, target_reached

    def reset(self):
        self.steps = 0
        self.target_reached = [False for _ in range(len(self.car_names))]
        self.goal_reward_given = [False for _ in range(len(self.car_names))]
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        states = []
        existing_goals = []

        for i in range(len(self.car_names)):
            angle = np.random.uniform(-np.pi, np.pi)
            self.set_self_states[i].pose.position.x = self.car_positions[i][0]
            self.set_self_states[i].pose.position.y = self.car_positions[i][1]
            self.set_self_states[i].pose.position.z = self.car_positions[i][2]
            self.set_self_states[i].pose.orientation.z = math.sin(angle / 2.0)
            self.set_self_states[i].pose.orientation.w = math.cos(angle / 2.0)

            self.set_states[i].publish(self.set_self_states[i])
            self.odom_positions[i][0] = self.set_self_states[i].pose.position.x
            self.odom_positions[i][1] = self.set_self_states[i].pose.position.y

            self.goal_positions[i][0], self.goal_positions[i][1] = generate_unique_goal(existing_goals, self.lower, self.upper)
            existing_goals.append((self.goal_positions[i][0], self.goal_positions[i][1]))

            self.publish_markers(i, [0.0, 0.0])

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
            v_state[:] = self.velodyne_data[i][:]
            laser_state = [v_state]

            distance = np.linalg.norm([self.odom_positions[i][0] - self.goal_positions[i][0], self.odom_positions[i][1] - self.goal_positions[i][1]])
            skew_x = self.goal_positions[i][0] - self.odom_positions[i][0]
            skew_y = self.goal_positions[i][1] - self.odom_positions[i][1]
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

            self.trajectories[i].points = []
            self.record_trajectory(i)

            states.append(state)

        return states

    def reset_car_position(self, index):
        angle = np.random.uniform(-np.pi, np.pi)
        self.set_self_states[index].pose.position.x = self.car_positions[index][0]
        self.set_self_states[index].pose.position.y = self.car_positions[index][1]
        self.set_self_states[index].pose.position.z = self.car_positions[index][2]
        self.set_self_states[index].pose.orientation.z = math.sin(angle / 2.0)
        self.set_self_states[index].pose.orientation.w = math.cos(angle / 2.0)

        self.set_states[index].publish(self.set_self_states[index])
        self.odom_positions[index][0] = self.set_self_states[index].pose.position.x
        self.odom_positions[index][1] = self.set_self_states[index].pose.position.y

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

    def publish_markers(self, index, action):
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "world"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = GOAL_REACHED_DIST * 2
        marker.scale.y = GOAL_REACHED_DIST * 2
        marker.scale.z = 0.01
        marker.color.a = 1.0
        if self.car_names[index] == "car1":
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        elif self.car_names[index] == "car2":
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
        elif self.car_names[index] == "car3":
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_positions[index][0]
        marker.pose.position.y = self.goal_positions[index][1]
        marker.pose.position.z = 0
        markerArray.markers.append(marker)
        self.publishers[index].publish(markerArray)

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
        self.publishers2[index].publish(markerArray2)

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
        self.publishers3[index].publish(markerArray3)

    @staticmethod
    def observe_collision(laser_data):
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    def get_reward(self, target, collision, action, distance):
        reward = action[0] / 2 - abs(action[1]) / 2 - (1 - distance) / 2
        if target:
            reward += GOAL_REWARD  # 只在第一次到达目标时给予奖励
        if collision:
            reward += COLLISION_PENALTY
        return reward
