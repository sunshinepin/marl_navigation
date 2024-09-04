import math
import time
import numpy as np
import threading
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray

# 常量定义
GOAL_REACHED_DIST = 0.3
TIME_DELTA = 0.1
MAX_STEPS = 500
COLLISION_DIST = 0.3
COLLISION_PENALTY = -100.0  # 碰撞惩罚
GOAL_REWARD = 100.0  # 达到目标的奖励
STAY_REWARD = 1.0  # 保持在目标点的奖励
MIN_GOAL_DISTANCE = 1.0  # 目标点之间的最小距离


def check_pos(x, y):
    goal_ok = True
    # 检查位置是否在禁区内
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
    def __init__(self, launchfile, environment_dim, car_names, car_positions, car_orientations):
        print("GazeboEnv initialized")
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

    def get_state_dim(self):
        # 假设状态的维度是环境维度加上每个机器人状态
        return self.environment_dim + 4

    def get_action_dim(self):
        # 假设动作的维度为2
        return 2

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
        print("Entering step function")  # Debug output
        self.steps += 1
        rewards = []
        dones = []
        target_reached = []

        all_done = True  # Initialize to True, will be set to False if any car hasn't reached its goal

        for i, action in enumerate(actions):
            if not self.target_reached[i]:  # Only act if the target hasn't been reached
                try:
                    vel_cmd = Twist()
                    vel_cmd.linear.x = float(action[0])  # Ensure it's a float
                    vel_cmd.angular.z = float(action[1])  # Ensure it's a float
                    self.vel_pubs[i].publish(vel_cmd)
                    self.publish_markers(i, action)
                except ValueError as e:
                    raise e
            else:  # Stop the car if it has reached its goal
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
                vel_cmd = Twist()
                vel_cmd.linear.x = 0.0
                vel_cmd.angular.z = 0.0
                self.vel_pubs[i].publish(vel_cmd)
                print(f"Car {self.car_names[i]} reached its goal.")  # Debug output
            else:
                all_done = False  # If any car hasn't reached its goal, continue the episode

            if collision:
                print(f"Collision detected for car {self.car_names[i]}, resetting position.")  # Debug output
                self.reset_car_position(i)
                done = True  # If a collision happens, end the episode for this car
            for i, action in enumerate(actions):
                print(f"Processing car {i + 1} with action: {action}")  # 调试输出
                if not self.target_reached[i]:  # 只有目标未达成时才行动
                    try:
                        vel_cmd = Twist()
                        vel_cmd.linear.x = float(action[0])  # 确保是浮点数
                        vel_cmd.angular.z = float(action[1])  # 确保是浮点数
                        self.vel_pubs[i].publish(vel_cmd)
                        self.publish_markers(i, action)
                    except ValueError as e:
                        raise e
                else:  # 如果目标已达成，停止车辆
                    vel_cmd = Twist()
                    vel_cmd.linear.x = 0.0
                    vel_cmd.angular.z = 0.0
                    self.vel_pubs[i].publish(vel_cmd)
                    print(f"Car {self.car_names[i]} has reached its goal and stopped.")  # 调试输出

            robot_state = [distance, theta, action[0], action[1]]
            self.state = np.append(laser_state, robot_state)
            reward = self.get_reward(i, self.target_reached[i], collision, action, distance)

            self.record_trajectory(i)

            rewards.append(reward)
            dones.append(done)
            target_reached.append(self.target_reached[i])
            states.append(self.state)

        # Final decision to end the episode
        if all_done:
            print("All cars have reached their goals.")  # Debug output
        elif self.steps >= MAX_STEPS:
            print("Maximum steps reached.")  # Debug output

        done = all_done or self.steps >= MAX_STEPS  # Only end the episode if all cars are done or max steps reached

        return states, rewards, [done] * len(self.car_names), target_reached

    def reset(self):
        self.steps = 0
        self.target_reached = [False for _ in range(len(self.car_names))]
        self.goal_reward_given = [False for _ in range(len(self.car_names))]  # Reset reward flag for all agents
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
        # Reset to initial position and orientation
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
        self.publishers2[index].publish(markerArray2)

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
        self.publishers3[index].publish(markerArray3)

    @staticmethod
    def observe_collision(laser_data):
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    def get_reward(self, index, target, collision, action, distance):
        # Calculate the base reward considering the action and distance to the goal
        reward = action[0] / 2 - abs(action[1]) / 2 - (1 - distance) / 2

        if target:
            if not self.goal_reward_given[index]:  # 修改这里
                reward += GOAL_REWARD  # Give the full reward the first time the goal is reached
                self.goal_reward_given[index] = True  # 修改这里，标记该机器人已经获得奖励
            reward += STAY_REWARD  # Give a smaller reward for staying in the goal area

        if collision:
            reward += COLLISION_PENALTY  # Apply the collision penalty

        return reward

class DistributedGazeboEnv(GazeboEnv):
    def __init__(self, launchfile, environment_dim, car_names, car_positions, car_orientations):
        super().__init__(launchfile, environment_dim, car_names, car_positions, car_orientations)
        self.threads = []
        self.lock = threading.Lock()

    def run_car(self, car_index, actions, rewards, dones, target_reached, states):
        with self.lock:
            action = actions[car_index]

            if not self.target_reached[car_index]:
                try:
                    vel_cmd = Twist()
                    vel_cmd.linear.x = float(action[0])
                    vel_cmd.angular.z = float(action[1])
                    self.vel_pubs[car_index].publish(vel_cmd)
                    self.publish_markers(car_index, action)
                except ValueError as e:
                    raise e
            else:
                vel_cmd = Twist()
                vel_cmd.linear.x = 0.0
                vel_cmd.angular.z = 0.0
                self.vel_pubs[car_index].publish(vel_cmd)

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

            done, collision, min_laser = self.observe_collision(self.velodyne_data[car_index])

            self.odom_positions[car_index][0] = self.last_odom[car_index].pose.pose.position.x
            self.odom_positions[car_index][1] = self.last_odom[car_index].pose.pose.position.y
            quaternion = Quaternion(
                self.last_odom[car_index].pose.pose.orientation.w,
                self.last_odom[car_index].pose.pose.orientation.x,
                self.last_odom[car_index].pose.pose.orientation.y,
                self.last_odom[car_index].pose.pose.orientation.z,
            )
            euler = quaternion.to_euler(degrees=False)
            angle = round(euler[2], 4)

            distance = np.linalg.norm(
                [self.odom_positions[car_index][0] - self.goal_positions[car_index][0],
                 self.odom_positions[car_index][1] - self.goal_positions[car_index][1]]
            )

            skew_x = self.goal_positions[car_index][0] - self.odom_positions[car_index][0]
            skew_y = self.goal_positions[car_index][1] - self.odom_positions[car_index][1]
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
                self.target_reached[car_index] = True
                vel_cmd = Twist()
                vel_cmd.linear.x = 0.0
                vel_cmd.angular.z = 0.0
                self.vel_pubs[car_index].publish(vel_cmd)
                print(f"Car {self.car_names[car_index]} reached its goal.")
            else:
                all_done = False

            if collision:
                print(f"Collision detected for car {self.car_names[car_index]}, resetting position.")
                self.reset_car_position(car_index)
                done = True

            robot_state = [distance, theta, action[0], action[1]]
            laser_state = [self.velodyne_data[car_index][:]]
            state = np.append(laser_state, robot_state)

            reward = self.get_reward(car_index, self.target_reached[car_index], collision, action, distance)

            self.record_trajectory(car_index)

            with self.lock:
                rewards[car_index] = reward
                dones[car_index] = done
                target_reached[car_index] = self.target_reached[car_index]
                states[car_index] = state

    def step(self, actions):
        self.steps += 1
        rewards = [0.0] * len(self.car_names)
        dones = [False] * len(self.car_names)
        target_reached = [False] * len(self.car_names)
        states = [None] * len(self.car_names)

        self.threads = []
        for i in range(len(self.car_names)):
            thread = threading.Thread(target=self.run_car, args=(i, actions, rewards, dones, target_reached, states))
            thread.start()
            self.threads.append(thread)

        for thread in self.threads:
            thread.join()

        done = all(dones) or self.steps >= MAX_STEPS
        return states, rewards, [done] * len(self.car_names), target_reached
