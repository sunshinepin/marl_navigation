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

# 常量定义 这一版是回合内碰撞不重置位置,而是死亡
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
    """
    生成一个唯一的位置，确保与现有位置和参考位置之间的距离大于 MIN_GOAL_DISTANCE。
    existing_positions: 已生成的位置列表（如已有目标或已有起始位置）
    reference_positions: 参考位置列表（如目标位置生成时参考起始位置，反之亦然）
    """
    while True:
        x = np.random.uniform(lower, upper)
        y = np.random.uniform(lower, upper)
        if check_pos(x, y) and all(
            np.linalg.norm([x - px, y - py]) > MIN_GOAL_DISTANCE 
            for px, py in existing_positions + reference_positions
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
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
            for agent in self.agents
        }
        self.max_episode_steps = env_config.max_episode_steps
        self._current_step = 0
        self.alive = [True for _ in range(self.num_agents)] 
        self.prev_positions = [[0.0, 0.0] for _ in range(self.num_agents)]  # 记录上一步位置
        self.direction_reward_scale = 1.0  # 方向奖励的系数，可调整
        self.direction_penalty_scale = 1.0  # 方向惩罚的系数，可调整
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
        """
        执行动作，更新环境状态，并返回新的观测、奖励、完成标志和信息
        """
        self._current_step += 1
        observation = {}
        rewards = {}
        truncated = False if self._current_step < self.max_episode_steps else True
        dones = {}
        infos = {}

        # 应用动作（只对存活的智能体）
        for i, agent in enumerate(self.agents):
            if self.alive[i]:  # 只对存活的智能体应用动作
                action = action_dict[agent]
                action[0] = np.clip(action[0], self.action_space[agent].low[0], self.action_space[agent].high[0])
                action[1] = np.clip(action[1], self.action_space[agent].low[1], self.action_space[agent].high[1])

                vel_cmd = Twist()
                vel_cmd.linear.x = (action[0]+1)/2
                vel_cmd.angular.z = action[1]
                self.vel_pubs[i].publish(vel_cmd)
            else:
                # 对死亡的智能体发布零速度
                vel_cmd = Twist()
                vel_cmd.linear.x = 0.0
                vel_cmd.angular.z = 0.0
                self.vel_pubs[i].publish(vel_cmd)

        # 仿真步骤
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

        # 收集观测和计算奖励
        all_dead = True
        for i, agent in enumerate(self.agents):
            if self.alive[i]:  # 只处理存活的智能体
                done, collision, min_laser = self.observe_collision(self.velodyne_data[i])

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
                    angle = 0.0

                distance = np.linalg.norm([
                    self.odom_positions[i][0] - self.goal_positions[i][0],
                    self.odom_positions[i][1] - self.goal_positions[i][1]
                ])
                # 计算角度差
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

                # 检查是否到达目标
                if distance < GOAL_REACHED_DIST and not self.target_reached[i]:
                    self.target_reached[i] = True
                    self.alive[i] = False  # 到达目标后标记死亡
                    rospy.loginfo(f"Agent {i} reached the goal and died.")

                # 处理碰撞
                if collision:
                    self.alive[i] = False  # 碰撞后标记死亡
                    rospy.loginfo(f"Agent {i} died due to collision")

                all_dead = False  # 只要有一个存活就不是全死

            # 设置终止条件
            done_episode = self._current_step >= self.max_episode_steps
            dones[agent] = not self.alive[i]  # 每个智能体的done状态取决于是否存活
            truncated = truncated or all_dead  # 回合结束条件：达到最大步数或全死

            # 计算观测
            laser_state = self.velodyne_data[i]
            robot_state = [distance, theta, action_dict[agent][0], action_dict[agent][1]] if self.alive[i] else [0.0, 0.0, 0.0, 0.0]
            observation[agent] = np.concatenate([laser_state, robot_state])

            # 计算奖励
            reward = self.get_reward(
                self.target_reached[i], 
                collision if self.alive[i] else False, 
                action_dict[agent] if self.alive[i] else [0.0, 0.0], 
                min_laser if self.alive[i] else 0.0, 
                distance if self.alive[i] else 0.0, 
                i
            )
            rewards[agent] = reward
            infos[agent] = {"alive": self.alive[i], "reached_goal": self.target_reached[i]}
            self.record_trajectory(i)

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

        # 当前位置
        current_pos = self.odom_positions[car_index]
        # 目标位置
        goal_pos = self.goal_positions[car_index]
        
        # 计算目标方向（从当前位置到目标的向量）
        goal_direction = np.array(goal_pos) - np.array(current_pos)
        goal_direction = goal_direction / (np.linalg.norm(goal_direction) + 1e-6)  # 归一化，避免除以0
        
        # 计算位移方向（当前位置减去上一步位置）
        displacement = np.array(current_pos) - np.array(self.prev_positions[car_index])
        displacement_norm = np.linalg.norm(displacement)
        
        # 如果有位移，检查是否朝目标方向移动
        if displacement_norm > 0.01:  # 增加最小移动阈值，避免噪声
            displacement_direction = displacement / displacement_norm  # 归一化位移
            # 计算点积
            dot_product = np.dot(displacement_direction, goal_direction)
            
            # 动态奖励/惩罚：根据点积和位移幅度
            if dot_product > 0:  # 朝目标方向移动
                reward += self.direction_reward_scale * dot_product * displacement_norm  # 奖励与方向和距离成正比
            elif dot_product < 0:  # 反向移动
                reward -= self.direction_penalty_scale * abs(dot_product) * displacement_norm  # 惩罚与反方向和距离成正比

        # 更新上一步位置
        self.prev_positions[car_index] = current_pos.copy()

        # 保持原有奖励逻辑
        if target and not self.goal_reward_given[car_index]:
            reward += GOAL_REWARD  # 到达目标奖励
            self.goal_reward_given[car_index] = True
        elif target:  # 死亡后仍因到达目标给小奖励
            reward += 1.0

        if collision:
            reward += COLLISION_PENALTY  # 碰撞惩罚
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            reward += action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2 

        # 添加全局奖励：如果所有智能体都到达目标
        if all(self.target_reached):
            reward += 50.0  # 额外协同奖励

        return reward

    def reset(self):
        """
        重置环境到初始状态，并返回初始观测。
        每次重置时随机生成起始位置和朝向，并在RViz中显示目标点。
        """
        self.steps = 0
        self._current_step = 0
        self.target_reached = [False for _ in range(self.num_agents)]
        self.goal_reward_given = [False for _ in range(self.num_agents)]
        self.alive = [True for _ in range(self.num_agents)]  # 重置时所有智能体复活
        # 重置上一步位置
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

        # 生成目标位置并发布Marker
        existing_goals = []
        for i in range(self.num_agents):
            gx, gy = generate_unique_goal(existing_goals, self.lower, self.upper, [])
            self.goal_positions[i] = [gx, gy]
            existing_goals.append((gx, gy))
            # rospy.loginfo(f"Agent {self.agents[i]} assigned goal position: ({gx}, {gy})")
            self.publish_goal_marker(i)  # 发布目标点Marker

        # 生成随机起始位置和朝向
        existing_starts = []
        for i in range(self.num_agents):
            sx, sy = generate_unique_goal(existing_starts, self.lower, self.upper, existing_goals)
            self.start_positions[i] = [sx, sy, 0.01]
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
            # rospy.loginfo(f"Agent {self.agents[i]} assigned random start position: ({sx}, {sy}) and orientation: {angle:.2f}")

        rospy.sleep(TIME_DELTA)

        try:
            self.pause()
        except rospy.ServiceException as e:
            rospy.logerr("调用 /gazebo/pause_physics 服务失败")

        # 生成初始观测
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
            observations[agent] = np.concatenate([laser_state, robot_state])

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
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        
        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.goal_marker_pubs[index].publish(marker_array)
        # rospy.loginfo(f"Published goal marker array for {self.agents[index]} at ({self.goal_positions[index][0]}, {self.goal_positions[index][1]})")
