#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
from gym import spaces
import numpy as np
import rospy
import threading
import time
import math
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty
from squaternion import Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from xuance.environment import RawMultiAgentEnv
import yaml
from gym.spaces import Box
#这版是有20+4+6的观察空间
# 常量定义 
GOAL_REACHED_DIST = 0.35
TIME_DELTA = 0.1
MAX_STEPS = 500
COLLISION_DIST = 0.35
COLLISION_PENALTY = -100.0  # 碰撞惩罚
GOAL_REWARD = 100.0  # 达到目标的奖励
STAY_REWARD = 1.0  # 停留在目标区域的奖励
MIN_GOAL_DISTANCE = 1.0  # 目标之间的最小距离


def check_pos(x, y):
    a =0.5
    goal_ok = True
    # 检查位置是否在禁区内
    if (1.5 - a) < x < (4.5 + a) and (1.5 - a) < y < (4.5 + a):
        goal_ok = False
    if (-5 - a) < x < (-0.5 + a) and (1.5 - a) < y < (4.5 + a):
        goal_ok = False
    if (-5.5 - a) < x < (-2.5 + a) and (0.5 - a) < y < (5 + a):
        goal_ok = False
    if (0.5 - a) < x < (5 + a) and (-5.5 - a) < y < (-2.5 + a):
        goal_ok = False
    if (2.5 - a) < x < (5.5 + a) and (-5 - a) < y < (-0.5 + a):
        goal_ok = False
    if (-4.5 - a) < x < (-1.5 + a) and (-4.5 - a) < y < (-1.5 + a):
        goal_ok = False
    if (-7.5 - a) < x < (-5.5 + a) and (5.5 - a) < y < (7.5 + a):
        goal_ok = False
    if (-5.5 - a) < x < (-4.5 + a) and (3.0 - a) < y < (4.0 + a):
        goal_ok = False
    if (-5.5 - a) < x < (-4.5 + a) and (-7.0 - a) < y < (-6.0 + a):
        goal_ok = False
    if (4.5 - a) < x < (5.5 + a) and (5.0 - a) < y < (6.0 + a):
        goal_ok = False
    if (5.5 - a) < x < (6.5 + a) and (-6.5 - a) < y < (-5.5 + a):
        goal_ok = False

    if x > 6.5 or x < -6.5 or y > 6.5 or y < -6.5:
        goal_ok = False

    return goal_ok

def generate_unique_goal(existing_positions, lower, upper, reference_positions):
    while True:
        x = np.random.uniform(lower, upper)
        y = np.random.uniform(lower, upper)
        # 只取 reference_positions 的前两个值（x, y）
        ref_positions_2d = [(pos[0], pos[1]) for pos in reference_positions]
        if check_pos(x, y) and all(
            np.linalg.norm([x - px, y - py]) > MIN_GOAL_DISTANCE 
            for px, py in existing_positions + ref_positions_2d
        ):
            return x, y

class MyNewMultiAgentEnv(RawMultiAgentEnv):
    def __init__(self, env_config):
        super(MyNewMultiAgentEnv, self).__init__()
        # 环境配置参数
        self.env_id = env_config.env_id
        self.num_agents = len(env_config.car_names)
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.state_space = Box(low=-np.inf, high=np.inf, shape=(24 * self.num_agents,), dtype=np.float32)
        # 定义动作空间和观测空间
        self.action_space = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            for agent in self.agents
        }
        self.observation_space = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(24 + 3 * (self.num_agents - 1),), dtype=np.float32)
            for agent in self.agents
        }
        self.max_episode_steps = env_config.max_episode_steps
        self._current_step = 0
        self.alive = [True for _ in range(self.num_agents)] 
        self.prev_positions = [[0.0, 0.0] for _ in range(self.num_agents)]  # 记录上一步位置
        self.direction_reward_scale = 1.0  # 方向奖励的系数，可调整
        self.direction_penalty_scale = 2.0  # 方向惩罚的系数，可调整
        
        # 初始化Gazebo环境变量
        self.start_positions = env_config.car_positions
        self.start_orientations = env_config.car_orientations
        self.goal_positions = [[0, 0] for _ in range(self.num_agents)]
        self.upper = 5.0
        self.lower = -5.0
        self.velodyne_data = [np.ones(20) * 10 for _ in range(self.num_agents)]
        self.last_odom = [None for _ in range(self.num_agents)]
        self.steps = 0
        self.target_reached = [False for _ in range(self.num_agents)]
        self.goal_reward_given = [False for _ in range(self.num_agents)]

        self.odom_positions = [[0, 0] for _ in range(self.num_agents)]

        # 初始化模型状态
        self.set_self_states = []
        for i in range(self.num_agents):
            state = ModelState()
            state.model_name = env_config.car_names[i]
            state.pose.position.x = self.start_positions[i][0]
            state.pose.position.y = self.start_positions[i][1]
            state.pose.position.z = self.start_positions[i][2]
            angle = self.start_orientations[i]
            q = Quaternion.from_euler(0, 0, angle)
            state.pose.orientation.x = q.x
            state.pose.orientation.y = q.y
            state.pose.orientation.z = q.z
            state.pose.orientation.w = q.w
            self.set_self_states.append(state)

        # 初始化ROS节点
        rospy.init_node("gym_env", anonymous=True)

        # 初始化发布者和订阅者
        self.vel_pubs = []
        self.set_states = []
        self.goal_marker_pubs = []  # 新增：目标点Marker发布者
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.velodynes = []
        self.odom = []
        self.trajectory_files = []
        self.trajectory_pubs = []
        self.trajectories = []

        for i in range(self.num_agents):
            car_name = env_config.car_names[i]
            self.vel_pubs.append(rospy.Publisher(f"/{car_name}/cmd_vel", Twist, queue_size=1))
            self.set_states.append(rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10))
            self.velodynes.append(rospy.Subscriber(
                f"/{car_name}/velodyne/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1, callback_args=i))
            self.odom.append(rospy.Subscriber(
                f"/{car_name}/odom_gazebo", Odometry, self.odom_callback, queue_size=1, callback_args=i))
            self.trajectory_files.append(f"trajectory_{car_name}.txt")
            self.clear_trajectory_file(i)
            self.trajectory_pubs.append(rospy.Publisher(f"{car_name}/trajectory", Marker, queue_size=10))
            
            self.goal_marker_pubs.append(rospy.Publisher(f"{car_name}/goal_marker", MarkerArray, queue_size=10))

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

        self.ros_thread = threading.Thread(target=self.ros_spin)
        self.ros_thread.start()

        self.reset()
    
    def ros_spin(self):
        rospy.spin()
    
    def get_env_info(self):
        return {
            'state_space': self.observation_space,
            'action_space': self.action_space,
            'agents': self.agents,
            'num_agents': self.num_agents,
            'max_episode_steps': self.max_episode_steps
        }
    def avail_actions(self):
        return None
    
    def agent_mask(self):
        """Returns boolean mask variables indicating which agents are currently alive."""
        return {agent: self.alive[i] for i, agent in enumerate(self.agents)}
    
    def check_agent_collision(self):
        """检查智能体间的碰撞"""
        collisions = []
        for i in range(self.num_agents):
            if not self.alive[i]:
                collisions.append(False)
                continue
            collision = False
            for j in range(self.num_agents):
                if i != j and self.alive[j]:
                    dist = np.linalg.norm(
                        np.array(self.odom_positions[i]) - np.array(self.odom_positions[j])
                    )
                    if dist < COLLISION_DIST:  # 智能体间碰撞阈值与障碍物一致
                        collision = True
                        break
            collisions.append(collision)
        return collisions
    def state(self):
        global_state = []
        for i in range(self.num_agents):
            laser_state = self.velodyne_data[i]
            position = self.odom_positions[i]
            angle = self.start_orientations[i]
            goal_position = self.goal_positions[i]
            distance_to_goal = np.linalg.norm([position[0] - goal_position[0], position[1] - goal_position[1]])
            
            skew_x = goal_position[0] - position[0]
            skew_y = goal_position[1] - position[1]
            dot = skew_x * 1 + skew_y * 0
            mag1 = np.linalg.norm([skew_x, skew_y])
            mag2 = 1.0
            beta = np.arccos(dot / (mag1 * mag2)) if mag1 != 0 else 0.0
            
            if skew_y < 0:
                beta = -beta if skew_x >= 0 else beta
            
            theta = beta - angle
            theta = (theta + np.pi) % (2 * np.pi) - np.pi
            
            agent_state = np.concatenate([laser_state, [distance_to_goal, theta]])
            global_state.append(agent_state)
        
        global_state = np.concatenate(global_state)
        return global_state


    def clear_trajectory_file(self, index):
        with open(self.trajectory_files[index], "w") as file:
            file.write("")

    def record_trajectory(self, index):
        # 添加调试日志以检查属性
        # rospy.loginfo(f"Current environment attributes: {self.__dict__}")
        if not hasattr(self, 'odom_positions'):
            rospy.logerr(f"odom_positions attribute is missing for index {index}")
            return
        if index >= len(self.odom_positions):
            rospy.logerr(f"Index {index} out of range for odom_positions")
            return
        with open(self.trajectory_files[index], "a") as file:
            file.write(f"{self.odom_positions[index][0]},{self.odom_positions[index][1]}\n")
        # 更新轨迹（可选，若不需要可移除）
        point = Point()
        point.x = self.odom_positions[index][0]
        point.y = self.odom_positions[index][1]
        point.z = 0
        self.trajectories[index].points.append(point)
        self.trajectories[index].header.stamp = rospy.Time.now()
        self.trajectory_pubs[index].publish(self.trajectories[index])

    def velodyne_callback(self, msg, index):
        """
        处理激光雷达数据回调。
        """
        data = list(pc2.read_points(msg, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data[index] = np.ones(20) * 10
        for point in data:
            if point[2] > -0.2:
                x, y, z = point
                dist = np.linalg.norm([x, y, z])
                angle = np.arctan2(y, x)
                # 简化分段处理，根据需要调整
                gap_size = 2 * np.pi / 20
                bin_idx = int((angle + np.pi) / gap_size) % 20
                self.velodyne_data[index][bin_idx] = min(self.velodyne_data[index][bin_idx], dist)

    def odom_callback(self, od_data, index):
        """
        处理里程计数据回调。
        """
        self.last_odom[index] = od_data

    def step(self, action_dict):
        self._current_step += 1
        observation = {}
        rewards = {}
        truncated = False if self._current_step < self.max_episode_steps else True
        dones = {}
        infos = {}

        for i, agent in enumerate(self.agents):
            if self.alive[i]:
                action = action_dict[agent]
                action[0] = np.clip(action[0], self.action_space[agent].low[0], self.action_space[agent].high[0])
                action[1] = np.clip(action[1], self.action_space[agent].low[1], self.action_space[agent].high[1])
                vel_cmd = Twist()
                vel_cmd.linear.x = (action[0] + 1) / 2
                vel_cmd.angular.z = action[1]
                self.vel_pubs[i].publish(vel_cmd)
            else:
                vel_cmd = Twist()
                vel_cmd.linear.x = 0.0
                vel_cmd.angular.z = 0.0
                self.vel_pubs[i].publish(vel_cmd)

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

        agent_collisions = self.check_agent_collision()
        all_dead = True

        for i, agent in enumerate(self.agents):
            if self.alive[i]:
                done, collision, min_laser = self.observe_collision(self.velodyne_data[i])
                self.prev_positions[i] = self.odom_positions[i].copy()
                if self.last_odom[i] is not None:
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
                else:
                    angle = self.start_orientations[i]

                distance = np.linalg.norm([
                    self.odom_positions[i][0] - self.goal_positions[i][0],
                    self.odom_positions[i][1] - self.goal_positions[i][1]
                ])
                skew_x = self.goal_positions[i][0] - self.odom_positions[i][0]
                skew_y = self.goal_positions[i][1] - self.odom_positions[i][1]
                dot = skew_x * 1 + skew_y * 0
                mag1 = np.linalg.norm([skew_x, skew_y])
                mag2 = 1.0
                beta = np.arccos(dot / (mag1 * mag2)) if mag1 != 0 else 0.0
                if skew_y < 0:
                    beta = -beta if skew_x >= 0 else beta
                theta = beta - angle
                theta = (theta + np.pi) % (2 * np.pi) - np.pi

                if distance < GOAL_REACHED_DIST and not self.target_reached[i]:
                    self.target_reached[i] = True
                    self.alive[i] = False
                    rospy.loginfo(f"Agent {i} reached the goal and stopped.")
                if collision or agent_collisions[i]:
                    self.reset_car_position(i)
                    rospy.loginfo(f"Agent {i} collided and was reset.")
                    dones[agent] = False
                else:
                    dones[agent] = not self.alive[i]
                    all_dead = False

                laser_state = self.velodyne_data[i]
                robot_state = [distance, theta, action_dict[agent][0], action_dict[agent][1]]
                relative_positions = []
                for j in range(self.num_agents):
                    if j != i:  # 只处理其他智能体
                        if self.alive[j]:
                            rel_x = self.odom_positions[j][0] - self.odom_positions[i][0]
                            rel_y = self.odom_positions[j][1] - self.odom_positions[i][1]
                            rel_dist = np.linalg.norm([rel_x, rel_y])
                            relative_positions.extend([rel_x, rel_y, rel_dist])
                        else:
                            relative_positions.extend([0.0, 0.0, 0.0])
                observation[agent] = np.concatenate([laser_state, robot_state, relative_positions])
                

                reward = self.get_reward(
                    self.target_reached[i],
                    collision or agent_collisions[i],
                    action_dict[agent],
                    min_laser,
                    distance,
                    i
                )
                rewards[agent] = reward
                infos[agent] = {"alive": self.alive[i], "reached_goal": self.target_reached[i]}
                self.record_trajectory(i)
            else:
                observation[agent] = np.zeros(24 + 3 * (self.num_agents - 1))
                rewards[agent] = 0.0
                dones[agent] = True
                infos[agent] = {"alive": False, "reached_goal": self.target_reached[i]}

        truncated = truncated or all_dead
        return observation, rewards, dones, truncated, infos
    
    def observe_collision(self, laser_data):
        """
        检查是否发生碰撞。
        返回 (是否完成, 是否碰撞, 最小激光距离)
        """
        min_laser = np.min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser
    
    def get_reward(self, target, collision, action, min_laser, distance, car_index):
        reward = 0.0
        current_pos = self.odom_positions[car_index]
        goal_pos = self.goal_positions[car_index]

        # 方向奖励/惩罚
        goal_direction = np.array(goal_pos) - np.array(current_pos)
        goal_direction /= (np.linalg.norm(goal_direction) + 1e-6)
        displacement = np.array(current_pos) - np.array(self.prev_positions[car_index])
        displacement_norm = np.linalg.norm(displacement)
        if displacement_norm > 0.01:
            displacement_direction = displacement / displacement_norm
            dot_product = np.dot(displacement_direction, goal_direction)
            if dot_product > 0:
                # reward += self.direction_reward_scale * dot_product * displacement_norm
                reward += self.direction_reward_scale * dot_product
            # elif dot_product < 0:
            #     # reward -= self.direction_penalty_scale * abs(dot_product) * displacement_norm
            #     reward -= self.direction_penalty_scale * abs(dot_product)

        # 新增：邻近惩罚
        min_agent_dist = float('inf')
        for j in range(self.num_agents):
            if j != car_index and self.alive[j]:
                dist = np.linalg.norm(np.array(current_pos) - np.array(self.odom_positions[j]))
                min_agent_dist = min(min_agent_dist, dist)
        if min_agent_dist < 1.0:  # 安全距离阈值，可调整
            reward -= 10.0 * (1.0 - min_agent_dist)  # 接近时惩罚

        # 现有奖励逻辑
        if target and not self.goal_reward_given[car_index]:
            reward += GOAL_REWARD
            self.goal_reward_given[car_index] = True
        elif target:
            reward += STAY_REWARD
        if collision:
            reward += COLLISION_PENALTY
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            reward += action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2

        if all(self.target_reached):
            reward += 50.0  # 协作奖励
        return reward

    def reset(self):
        self.steps = 0
        self._current_step = 0
        self.target_reached = [False for _ in range(self.num_agents)]
        self.goal_reward_given = [False for _ in range(self.num_agents)]
        self.alive = [True for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            self.prev_positions[i] = self.odom_positions[i].copy()

        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("调用 /gazebo/reset_world 服务失败")

        try:
            self.unpause()
        except rospy.ServiceException as e:
            rospy.logerr("调用 /gazebo/unpause_physics 服务失败")
        rospy.sleep(TIME_DELTA * 2)

        existing_goals = []
        for i in range(self.num_agents):
            start_positions_2d = [(pos[0], pos[1]) for pos in self.start_positions]
            gx, gy = generate_unique_goal(existing_goals, self.lower, self.upper, start_positions_2d)
            self.goal_positions[i] = [gx, gy]
            existing_goals.append((gx, gy))
            self.publish_goal_marker(i)

        existing_starts = []
        for i in range(self.num_agents):
            sx, sy = generate_unique_goal(existing_starts, self.lower, self.upper, existing_goals)
            self.start_positions[i] = [sx, sy, 0.01]
            self.odom_positions[i] = [sx, sy]  # 同步初始位置
            self.prev_positions[i] = [sx, sy]
            existing_starts.append((sx, sy))

            self.start_orientations[i] = np.random.uniform(0, 2 * np.pi)
            angle = self.start_orientations[i]
            q = Quaternion.from_euler(0, 0, angle)
            self.set_self_states[i].pose.position.x = sx
            self.set_self_states[i].pose.position.y = sy
            self.set_self_states[i].pose.position.z = 0.01
            self.set_self_states[i].pose.orientation.x = q.x
            self.set_self_states[i].pose.orientation.y = q.y
            self.set_self_states[i].pose.orientation.z = q.z
            self.set_self_states[i].pose.orientation.w = q.w
            self.set_states[i].publish(self.set_self_states[i])

        rospy.sleep(1.0)  # 增加等待时间，确保 Gazebo 更新里程计
        try:
            self.pause()
        except rospy.ServiceException as e:
            rospy.logerr("调用 /gazebo/pause_physics 服务失败")

        # 可选：验证里程计数据
        for i in range(self.num_agents):
            if self.last_odom[i] is not None:
                self.odom_positions[i][0] = self.last_odom[i].pose.pose.position.x
                self.odom_positions[i][1] = self.last_odom[i].pose.pose.position.y
            else:
                rospy.logwarn(f"Agent {i} odom data not available yet, using start position")

        # 打印调试信息
        print(f"After reset, odom_positions: {self.odom_positions}")

        observations = {}
        for i, agent in enumerate(self.agents):
            laser_state = self.velodyne_data[i]
            distance = np.linalg.norm([
                self.start_positions[i][0] - self.goal_positions[i][0],
                self.start_positions[i][1] - self.goal_positions[i][1]
            ])
            skew_x = self.goal_positions[i][0] - self.start_positions[i][0]
            skew_y = self.goal_positions[i][1] - self.start_positions[i][1]
            dot = skew_x * 1 + skew_y * 0
            mag1 = np.linalg.norm([skew_x, skew_y])
            mag2 = 1.0
            beta = np.arccos(dot / (mag1 * mag2)) if mag1 != 0 else 0.0
            if skew_y < 0:
                beta = -beta if skew_x >= 0 else beta
            theta = beta - self.start_orientations[i]
            theta = (theta + np.pi) % (2 * np.pi) - np.pi

            robot_state = [distance, theta, 0.0, 0.0]
            relative_positions = []
            for j in range(self.num_agents):
                if j != i:  # 只处理其他智能体
                    if self.alive[j]:
                        rel_x = self.start_positions[j][0] - self.start_positions[i][0]
                        rel_y = self.start_positions[j][1] - self.start_positions[i][1]
                        rel_dist = np.linalg.norm([rel_x, rel_y])
                        relative_positions.extend([rel_x, rel_y, rel_dist])
                    else:
                        relative_positions.extend([0.0, 0.0, 0.0])
            observations[agent] = np.concatenate([laser_state, robot_state, relative_positions])

            self.trajectories[i].points = []
            self.record_trajectory(i)

        return observations, {}

    def reset_car_position(self, index):
        """
        在碰撞时随机重置指定智能体的位置和朝向。
        """
        # 生成随机位置，确保与其他位置和目标保持距离
        existing_starts = [(self.odom_positions[i][0], self.odom_positions[i][1]) for i in range(self.num_agents) if i != index]
        existing_goals = [(self.goal_positions[i][0], self.goal_positions[i][1]) for i in range(self.num_agents)]
        sx, sy = generate_unique_goal(existing_starts, self.lower, self.upper, existing_goals)
        self.start_positions[index] = [sx, sy, 0.01]  # 更新起始位置

        # 随机生成朝向
        angle = np.random.uniform(0, 2 * np.pi)
        self.start_orientations[index] = angle  # 更新起始朝向
        q = Quaternion.from_euler(0, 0, angle)

        # 设置模型状态
        self.set_self_states[index].pose.position.x = sx
        self.set_self_states[index].pose.position.y = sy
        self.set_self_states[index].pose.position.z = 0.01
        self.set_self_states[index].pose.orientation.x = q.x
        self.set_self_states[index].pose.orientation.y = q.y
        self.set_self_states[index].pose.orientation.z = q.z
        self.set_self_states[index].pose.orientation.w = q.w

        # 发布模型状态
        self.set_states[index].publish(self.set_self_states[index])
        self.odom_positions[index] = [sx, sy]  # 更新odom位置
        # rospy.loginfo(f"Agent {self.agents[index]} reset to random position: ({sx}, {sy}) and orientation: {angle:.2f}")

        # 允许仿真更新
        try:
            self.unpause()
        except rospy.ServiceException as e:
            rospy.logerr("调用 /gazebo/unpause_physics 服务失败")

        rospy.sleep(TIME_DELTA)

        # 暂停仿真
        try:
            self.pause()
        except rospy.ServiceException as e:
            rospy.logerr("调用 /gazebo/pause_physics 服务失败")

    def render(self, *args, **kwargs):
        """
        可选：实现渲染功能（如使用Rviz进行可视化）。
        """
        return np.ones([64, 64, 64])

    def close(self):
        """
        关闭环境，清理资源。
        """
        rospy.signal_shutdown("训练完成")
        self.ros_thread.join()

    def publish_goal_marker(self, index):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = f"goal_{self.agents[index]}"
        marker.id = index
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.goal_positions[index][0]
        marker.pose.position.y = self.goal_positions[index][1]
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0  # 不透明度

        # 根据 index 设置固定颜色
        if index == 0:  # agent_0: 红色
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        elif index == 1:  # agent_1: 绿色
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        elif index == 2:  # agent_2: 蓝色
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
        else:
            # 如果有意外 index，默认灰色（仅作为保护，未使用）
            marker.color.r = 0.5
            marker.color.g = 0.5
            marker.color.b = 0.5

        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.goal_marker_pubs[index].publish(marker_array)
