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

# Constants
GOAL_REACHED_DIST = 0.3
TIME_DELTA = 0.1
MAX_STEPS = 1500
COLLISION_DIST = 0.3
COLLISION_PENALTY = -100.0
GOAL_REWARD = 100.0
STAY_REWARD = 1.0
MIN_GOAL_DISTANCE = 1.0


def check_pos(x, y):
    goal_ok = True
    a = 0.65
    if (-7.49 - a) < x < (-7.49 + a) and (7.44 - a) < y < (7.44 + a):
        goal_ok = False

    if (-4.49 - a) < x < (-4.49 + a) and (7.47 - a) < y < (7.47 + a):
        goal_ok = False
    if (-1.53 - a) < x < (-1.53 + a) and (7.48 - a) < y < (7.48 + a):
        goal_ok = False
    if (1.56 - a) < x < (1.56 + a) and (7.45 - a) < y < (7.45 + a):
        goal_ok = False
    if (4.5 - a) < x < (4.5 + a) and (7.5 - a) < y < (7.5 + a):
        goal_ok = False
    if (7.49 - a) < x < (7.49 + a) and (7.5 - a) < y < (7.5 + a):
        goal_ok = False
    if (-7.49 - a) < x < (-7.49 + a) and (4.5 - a) < y < (4.5 + a):
        goal_ok = False
    if (-2.4 - a) < x < (-2.4 + a) and (5.45 - a) < y < (5.45 + a):
        goal_ok = False
    if (2.5 - a) < x < (2.5 + a) and (4.5 - a) < y < (4.5 + a):
        goal_ok = False
    if (4.49 - a) < x < (4.49 + a) and (5.52 - a) < y < (5.52 + a):
        goal_ok = False
    if (7.5 - a) < x < (7.5 + a) and (4.5 - a) < y < (4.5 + a):
        goal_ok = False
    if (-4.37 - a) < x < (-4.37 + a) and (3.49 - a) < y < (3.49 + a):
        goal_ok = False
    if (-7.49 - a) < x < (-7.49 + a) and (1.51 - a) < y < (1.51 + a):
        goal_ok = False
    if (-3.57 - a) < x < (-3.57 + a) and (0.49 - a) < y < (0.49 + a):
        goal_ok = False
    if (0.53 - a) < x < (0.53 + a) and (1.52 - a) < y < (1.52 + a):
        goal_ok = False
    if (4.5 - a) < x < (4.5 + a) and (1.49 - a) < y < (1.49 + a):
        goal_ok = False
    if (7.58 - a) < x < (7.58 + a) and (1.5 - a) < y < (1.5 + a):
        goal_ok = False
    if (-7.5 - a) < x < (-7.5 + a) and (-1.45 - a) < y < (-1.45 + a):
        goal_ok = False
    if (-7.5 - a) < x < (-7.5 + a) and (-4.5 - a) < y < (-4.5 + a):
        goal_ok = False
    if (-5.47 - a) < x < (-5.47 + a) and (-3.49 - a) < y < (-3.49 + a):
        goal_ok = False
    if (-2.46 - a) < x < (-2.46 + a) and (-3.44 - a) < y < (-3.44 + a):
        goal_ok = False
    if (-0.53 - a) < x < (-0.53 + a) and (-1.56 - a) < y < (-1.56 + a):
        goal_ok = False
    if (0.5 - a) < x < (0.5 + a) and (-4.5 - a) < y < (-4.5 + a):
        goal_ok = False
    if (2.44 - a) < x < (2.44 + a) and (-1.52 - a) < y < (-1.52 + a):
        goal_ok = False
    if (4.46 - a) < x < (4.46 + a) and (-4.45 - a) < y < (-4.45 + a):
        goal_ok = False
    if (7.5 - a) < x < (7.5 + a) and (-1.51 - a) < y < (-1.51 + a):
        goal_ok = False
    if (7.54 - a) < x < (7.54 + a) and (-4.58 - a) < y < (-4.58 + a):
        goal_ok = False
    if (-7.47 - a) < x < (-7.47 + a) and (-7.51 - a) < y < (-7.51 + a):
        goal_ok = False
    if (-4.5 - a) < x < (-4.5 + a) and (-7.49 - a) < y < (-7.49 + a):
        goal_ok = False
    if (-1.49 - a) < x < (-1.49 + a) and (-7.48 - a) < y < (-7.48 + a):
        goal_ok = False
    if (1.49 - a) < x < (1.49 + a) and (-7.5 - a) < y < (-7.5 + a):
        goal_ok = False
    if (4.5 - a) < x < (4.5 + a) and (-7.5 - a) < y < (-7.5 + a):
        goal_ok = False
    if (7.5 - a) < x < (7.5 + a) and (-7.5 - a) < y < (-7.5 + a):
        goal_ok = False

    if x > 9.5 or x < -9.5 or y > 9.5 or y < -9.5:
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
    def __init__(self, launchfile, environment_dim, car_names, car_positions, car_orientations):
        self.environment_dim = environment_dim
        self.car_names = car_names
        self.car_positions = car_positions
        self.car_orientations = car_orientations
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
            angle = car_orientations[i]
            state.pose.orientation.x = 0.0
            state.pose.orientation.y = 0.0
            state.pose.orientation.z = math.sin(angle / 2.0)
            state.pose.orientation.w = math.cos(angle / 2.0)
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
                f"/{car_name}/velodyne/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1,
                callback_args=i))
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

    def record_trajectory(self, index):
        point = Point()
        point.x = self.odom_positions[index][0]
        point.y = self.odom_positions[index][1]
        point.z = 0
        self.trajectories[index].points.append(point)
        self.trajectories[index].header.stamp = rospy.Time.now()
        self.trajectory_pubs[index].publish(self.trajectories[index])

    def step(self, actions):
        self.steps += 1
        rewards = []
        dones = []
        target_reached = []

        for i, (agent, action) in enumerate(actions.items()):
            if not self.target_reached[i]:
                try:
                    vel_cmd = Twist()
                    vel_cmd.linear.x = float(action[0])  # Ensure it's a float
                    vel_cmd.angular.z = float(action[1])  # Ensure it's a float
                    self.vel_pubs[i].publish(vel_cmd)
                    self.publish_markers(i, action)
                except ValueError as e:
                    print(f"Error converting action to float for agent {i}: {action}")
                    raise e
            else:
                vel_cmd = Twist()
                vel_cmd.linear.x = 0.0
                vel_cmd.angular.z = 0.0
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

        states = {}
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
            reward = self.get_reward(self.target_reached[i], collision, action, min_laser, distance, i)

            self.record_trajectory(i)

            rewards.append(reward)
            dones.append(done)
            target_reached.append(self.target_reached[i])
            states[f"agent_{i}"] = state

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

        states = {}
        existing_goals = []

        for i in range(len(self.car_names)):
            # Set initial orientation and position
            angle = self.car_orientations[i]
            self.set_self_states[i].pose.position.x = self.car_positions[i][0]
            self.set_self_states[i].pose.position.y = self.car_positions[i][1]
            self.set_self_states[i].pose.position.z = self.car_positions[i][2]
            self.set_self_states[i].pose.orientation.z = math.sin(angle / 2.0)
            self.set_self_states[i].pose.orientation.w = math.cos(angle / 2.0)

            self.set_states[i].publish(self.set_self_states[i])
            self.odom_positions[i][0] = self.set_self_states[i].pose.position.x
            self.odom_positions[i][1] = self.set_self_states[i].pose.position.y

            self.goal_positions[i][0], self.goal_positions[i][1] = generate_unique_goal(existing_goals, self.lower,
                                                                                        self.upper)
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

            distance = np.linalg.norm([self.odom_positions[i][0] - self.goal_positions[i][0],
                                       self.odom_positions[i][1] - self.goal_positions[i][1]])
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

            states[f"agent_{i}"] = state

        return states

    def reset_car_position(self, index):
        angle = self.car_orientations[index]
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
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException as e:
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
        marker2.scale.x = abs(float(action[0]))
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
        marker3.scale.x = abs(float(action[1]))
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

    def get_reward(self, target, collision, action, min_laser, distance, car_index):
        reward = 0.0
        if target:
            if not self.goal_reward_given[car_index]:
                time_step_reward = MAX_STEPS - self.steps
                reward += GOAL_REWARD + time_step_reward/2
                self.goal_reward_given[car_index] = True
        if collision:
            reward += COLLISION_PENALTY
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            reward += action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2

        return reward


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time

# 定义神经网络模型
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class TD3(object):
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )


# 设置模型参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0
max_ep = 800
file_name = "TD3_velodyne"

# 固定的目标点
# target_points = [
#     [9.0, -9.0, 0.0, 0.0],  # Car1 目标点
#     [-9.0, 9.0, 0.0, 0.0],    # Car2 目标点
#     [9.0, 9.0, 0.0, 0.0]    # Car3 目标点
# ]
target_points = [
    [-9.0, -9.0, 0.0, 0.0],  # Car1 目标点
    [9.0, 9.0, 0.0, 0.0],    # Car2 目标点
    [9.0, -9.0, 0.0, 0.0]    # Car3 目标点
]

# 每辆车的起始点
start_points = [[0.0, 9.0, 0.01], [0.0, -9.0, 0.01], [-9.0, 0.0, 0.01]]
car_orientations = [-1.57, 1.57, 0]
car_names = ["car1", "car2", "car3"]

# 初始化环境和模型
environment_dim = 20
robot_dim = 4
env = GazeboEnv("td3.launch", environment_dim, car_names, start_points, car_orientations)

torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2

# 为每辆车创建独立的网络
networks = [TD3(state_dim, action_dim) for _ in range(3)]
for i, network in enumerate(networks):
    try:
        # network.load(file_name + f"_car{i+1}", "./pytorch_models")
        network.load(file_name , "./pytorch_models")

    except:
        raise ValueError(f"Could not load the stored model parameters for car{i+1}")

# 设置每辆车的起点和固定的目标点
for i in range(3):
    env.set_self_states[i].pose.position.x = start_points[i][0]
    env.set_self_states[i].pose.position.y = start_points[i][1]
    env.set_self_states[i].pose.position.z = start_points[i][2]
    angle = car_orientations[i]
    env.set_self_states[i].pose.orientation.z = math.sin(angle / 2.0)
    env.set_self_states[i].pose.orientation.w = math.cos(angle / 2.0)
    env.set_states[i].publish(env.set_self_states[i])

    # 设置固定的目标点
    env.goal_positions[i][0] = target_points[i][0]
    env.goal_positions[i][1] = target_points[i][1]

# 确保目标点在 reset 后不被覆盖
states = env.reset()

# 再次设置固定的目标点，确保它们不会被覆盖
for i in range(3):
    env.goal_positions[i][0] = target_points[i][0]
    env.goal_positions[i][1] = target_points[i][1]

# 初始化评估指标
trajectories = [[] for _ in range(3)]
total_rewards = [0 for _ in range(3)]
reached_goal_steps = [-1 for _ in range(3)]  # -1 表示没有达到目标点

time.sleep(5)

# 开始测试
done = [False, False, False]
episode_timesteps = [0, 0, 0]

while not all(done):
    actions = {}
    for i in range(3):
        action = networks[i].get_action(np.array(states[f"agent_{i}"]))
        a_in = [(action[0] + 1) / 2, action[1]]
        actions[f"agent_{i}"] = a_in

    # 执行动作并获取新状态、奖励和完成标志
    next_states, rewards, dones, targets = env.step(actions)

    for i in range(3):
        done[i] = True if episode_timesteps[i] + 1 == max_ep else dones[i]

        # 更新状态、奖励和轨迹
        if not done[i]:
            states[f"agent_{i}"] = next_states[f"agent_{i}"]
            episode_timesteps[i] += 1

            # 记录轨迹点
            trajectories[i].append(env.odom_positions[i])
            total_rewards[i] += rewards[i]

            # 检查是否达到了目标点
            if targets[i] and reached_goal_steps[i] == -1:
                reached_goal_steps[i] = episode_timesteps[i]

# 计算轨迹长度和其他指标
trajectory_lengths = []
for i in range(3):
    length = 0
    for j in range(1, len(trajectories[i])):
        length += np.linalg.norm(np.array(trajectories[i][j]) - np.array(trajectories[i][j - 1]))
    trajectory_lengths.append(length)

# 打印评估结果
for i in range(3):
    print(f"Car {i+1} Evaluation:")
    print(f"  - Trajectory Length: {trajectory_lengths[i]:.2f} meters")
    print(f"  - Total Reward: {total_rewards[i]:.2f}")
    print(f"  - Steps to Reach Goal: {reached_goal_steps[i] if reached_goal_steps[i] != -1 else 'Not reached'}")
    print(f"  - Total Steps: {episode_timesteps[i]}")
    print()

# 如果需要返回结果，可以将结果返回
# return trajectory_lengths, total_rewards, reached_goal_steps, episode_timesteps




