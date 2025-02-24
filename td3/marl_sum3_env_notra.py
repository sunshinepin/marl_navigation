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
from xuance.environment import RawMultiAgentEnv
import yaml
from visualization_msgs.msg import Marker

# 常量定义
GOAL_REACHED_DIST = 0.3
TIME_DELTA = 0.1
MAX_STEPS = 500
COLLISION_DIST = 0.35
COLLISION_PENALTY = -100.0  # 碰撞惩罚
GOAL_REWARD = 100.0  # 达到目标的奖励
MIN_GOAL_DISTANCE = 1.0  # 目标之间的最小距离

def check_pos(x, y):
    goal_ok = True

    if x > 9.0 or x < -9.0 or y > 9.0 or y < -9.0:
        goal_ok = False

    return goal_ok

def generate_unique_goal(existing_goals, lower, upper, start_positions):
    while True:
        goal_x = np.random.uniform(lower, upper)
        goal_y = np.random.uniform(lower, upper)
        if check_pos(goal_x, goal_y) and all(
                np.linalg.norm([goal_x - gx, goal_y - gy]) > MIN_GOAL_DISTANCE for gx, gy in existing_goals
        ) and all(
                np.linalg.norm([goal_x - sx, goal_y - sy]) > MIN_GOAL_DISTANCE for sx, sy in start_positions
        ):
            return goal_x, goal_y

class MyNewMultiAgentEnv(RawMultiAgentEnv):
    def __init__(self, env_config):
        super(MyNewMultiAgentEnv, self).__init__()
        # 环境配置参数
        self.env_id = env_config.env_id
        self.num_agents = len(env_config.car_names)
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        
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

        # 初始化Gazebo环境变量
        self.start_positions = env_config.car_positions  # 从 YAML 读取起始位置
        self.start_orientations = env_config.car_orientations  # 从 YAML 读取初始角度
        self.goal_positions = [[0, 0] for _ in range(self.num_agents)]  # 初始化为空，稍后重置
        self.upper = 5.0
        self.lower = -5.0
        self.velodyne_data = [np.ones(20) * 10 for _ in range(self.num_agents)]
        self.last_odom = [None for _ in range(self.num_agents)]
        self.steps = 0
        self.target_reached = [False for _ in range(self.num_agents)]
        self.goal_reward_given = [False for _ in range(self.num_agents)]

        # 初始化odom_positions
        self.odom_positions = [[0, 0] for _ in range(self.num_agents)]

        # 初始化模型状态
        self.set_self_states = []
        for i in range(self.num_agents):
            state = ModelState()
            state.model_name = env_config.car_names[i]
            # 设置起始位置
            state.pose.position.x = self.start_positions[i][0]
            state.pose.position.y = self.start_positions[i][1]
            state.pose.position.z = self.start_positions[i][2]
            # 设置起始朝向
            angle = self.start_orientations[i]
            q = Quaternion.from_euler(0, 0, angle)  # 使用 squaternion 将角度转换为四元数
            state.pose.orientation.x = q.x
            state.pose.orientation.y = q.y
            state.pose.orientation.z = q.z
            state.pose.orientation.w = q.w
            self.set_self_states.append(state)

        # 初始化ROS节点和发布者/订阅者
        rospy.init_node("gym_env", anonymous=True)
        
        self.vel_pubs = []
        self.set_states = []
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.velodynes = []
        self.odom = []
        
        for i in range(self.num_agents):
            car_name = env_config.car_names[i]
            self.vel_pubs.append(rospy.Publisher(f"/{car_name}/cmd_vel", Twist, queue_size=1))
            self.set_states.append(rospy.Publisher(
                "/gazebo/set_model_state", ModelState, queue_size=10
            ))
            self.velodynes.append(rospy.Subscriber(
                f"/{car_name}/velodyne/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1,
                callback_args=i))
            self.odom.append(rospy.Subscriber(
                f"/{car_name}/odom_gazebo", Odometry, self.odom_callback, queue_size=1, callback_args=i))
        
        # 启动ROS回调线程
        self.ros_thread = threading.Thread(target=self.ros_spin)
        self.ros_thread.start()
        
        # 初始化环境
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
        return {agent: True for agent in self.agents}

    def state(self):
        """Returns the global state of the environment."""
        return self.observation_space.sample()

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
                gap_size = 2 * np.pi / 8
                bin_idx = int((angle + np.pi) / gap_size) % 8
                self.velodyne_data[index][bin_idx] = min(self.velodyne_data[index][bin_idx], dist)

    def odom_callback(self, od_data, index):
        """
        处理里程计数据回调。
        """
        self.last_odom[index] = od_data

    def step(self, action_dict):
        """
        执行动作，更新环境状态，并返回新的观测、奖励、完成标志和信息。
        action_dict: dict，键为agent名称，值为动作
        """
        self._current_step += 1
        observation = {agent: self.observation_space[agent].sample() for agent in self.agents}
        rewards = {agent: np.random.random() for agent in self.agents}
        self.steps += 1
        dones = {}
        terminated = {agent: False for agent in self.agents}
        truncated = False if self._current_step < self.max_episode_steps else True
        infos = {}

        # 应用动作
        for i, agent in enumerate(self.agents):
            action = action_dict[agent]
            if not self.target_reached[i]:
                try:
                    vel_cmd = Twist()
                    vel_cmd.linear.x = float(action[0])
                    vel_cmd.angular.z = float(action[1])
                    self.vel_pubs[i].publish(vel_cmd)
                except ValueError as e:
                    rospy.logerr(f"智能体 {agent} 的动作转换错误: {action}")
                    vel_cmd = Twist()
                    vel_cmd.linear.x = 0.0
                    vel_cmd.angular.z = 0.0
                    self.vel_pubs[i].publish(vel_cmd)
            else:
                # 如果目标已达，停止智能体
                vel_cmd = Twist()
                vel_cmd.linear.x = 0.0
                vel_cmd.angular.z = 0.0
                self.vel_pubs[i].publish(vel_cmd)

        # 取消暂停仿真
        try:
            self.unpause()
        except rospy.ServiceException as e:
            rospy.logerr("调用 /gazebo/unpause_physics 服务失败")

        # 等待物理更新
        rospy.sleep(TIME_DELTA)

        # 暂停仿真
        try:
            self.pause()
        except rospy.ServiceException as e:
            rospy.logerr("调用 /gazebo/pause_physics 服务失败")

        # 收集观测、计算奖励并检查完成标志
        observations = {}
        for i, agent in enumerate(self.agents):
            done, collision, min_laser = self.observe_collision(self.velodyne_data[i])

            # 更新里程计
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
                angle = 0.0  # 默认角度

            # 计算与目标的距离和角度
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

            # 根据 y 偏转调整 beta
            if skew_y < 0:
                beta = -beta if skew_x >= 0 else beta

            theta = beta - angle
            theta = (theta + np.pi) % (2 * np.pi) - np.pi  # 归一化到 [-pi, pi]

            # 检查是否到达目标
            if distance < GOAL_REACHED_DIST:
                self.target_reached[i] = True

            # 检查是否超过最大步数
            done_episode = self.steps >= self.max_episode_steps
            dones[agent] = done_episode or done
            if done_episode:
                rospy.loginfo("达到最大步数")

            # 处理碰撞
            if collision:
                self.reset_car_position(i)
                dones[agent] = True  # 可选：碰撞后结束智能体回合

            # 构建观测
            laser_state = self.velodyne_data[i]
            robot_state = [distance, theta, action_dict[agent][0], action_dict[agent][1]]
            observation = np.concatenate([laser_state, robot_state])
            observations[agent] = observation
            rospy.loginfo(f"Agent {agent} observation: {observation}")

            # 计算奖励
            reward = self.get_reward(
                self.target_reached[i],
                collision,
                action_dict[agent],
                min_laser,
                distance,
                i
            )
            rewards[agent] = reward

            # 信息字典
            infos[agent] = {}

        # 判断是否所有智能体都完成
        done = all(dones.values())
        return observations, rewards, dones, truncated, infos

    def reset(self):
        """
        重置环境到初始状态，并返回初始观测。
        """
        observation = {agent: self.observation_space[agent].sample() for agent in self.agents}
        info = {}
        self._current_step = 0
        self.steps = 0
        self.target_reached = [False for _ in range(self.num_agents)]
        self.goal_reward_given = [False for _ in range(self.num_agents)]
        
        # 重置仿真
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("调用 /gazebo/reset_world 服务失败")
        
        # 等待仿真重置
        rospy.sleep(TIME_DELTA)
        
        observations = {}
        existing_goals = []
        
        # **首先生成所有智能体的唯一目标位置**
        start_positions_2d = [[pos[0], pos[1]] for pos in self.start_positions]
        for i, agent in enumerate(self.agents):
            # 生成唯一的目标位置
            gx, gy = generate_unique_goal(existing_goals, self.lower, self.upper, start_positions_2d)
            self.goal_positions[i][0] = gx
            self.goal_positions[i][1] = gy
            existing_goals.append((gx, gy))
            rospy.loginfo(f"Agent {agent} assigned goal position: ({gx}, {gy})")
            
            # 发布目标位置Marker
            goal_marker = Marker()
            goal_marker.header.frame_id = "world"
            goal_marker.type = Marker.CYLINDER
            goal_marker.action = Marker.ADD
            goal_marker.scale.x = 0.6  # 直径
            goal_marker.scale.y = 0.6
            goal_marker.scale.z = 0.01
            goal_marker.color.a = 1.0
            goal_marker.color.r = 1.0
            goal_marker.color.g = 0.0
            goal_marker.color.b = 0.0
            goal_marker.pose.position.x = gx
            goal_marker.pose.position.y = gy
            goal_marker.pose.position.z = 0
            goal_marker.pose.orientation.w = 1.0
            self.set_goal_marker(i, goal_marker)
        
        # **然后设置每个智能体的模型状态到起始位置**
        for i, agent in enumerate(self.agents):
            # 获取起始位置和朝向
            start_x, start_y, start_z = self.start_positions[i]
            angle = self.start_orientations[i]
            q = Quaternion.from_euler(0, 0, angle)
            
            # 设置模型状态
            self.set_self_states[i].pose.position.x = start_x
            self.set_self_states[i].pose.position.y = start_y
            self.set_self_states[i].pose.position.z = start_z
            self.set_self_states[i].pose.orientation.x = q.x
            self.set_self_states[i].pose.orientation.y = q.y
            self.set_self_states[i].pose.orientation.z = q.z
            self.set_self_states[i].pose.orientation.w = q.w
            
            # 发布模型状态
            self.set_states[i].publish(self.set_self_states[i])
            rospy.loginfo(f"Agent {agent} reset to start position: ({start_x}, {start_y})")
            
            # 更新 odom_positions 为起始位置
            self.odom_positions[i][0] = start_x
            self.odom_positions[i][1] = start_y
        
        # **允许仿真更新**
        try:
            self.unpause()
        except rospy.ServiceException as e:
            rospy.logerr("调用 /gazebo/unpause_physics 服务失败")
        
        rospy.sleep(TIME_DELTA)
        
        # **暂停仿真**
        try:
            self.pause()
        except rospy.ServiceException as e:
            rospy.logerr("调用 /gazebo/pause_physics 服务失败")
        
        # **构建初始观测**
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
            
            # 根据 y 偏转调整 beta
            if skew_y < 0:
                beta = -beta if skew_x >= 0 else beta
            
            theta = beta - self.start_orientations[i]
            theta = (theta + np.pi) % (2 * np.pi) - np.pi  # 归一化到 [-pi, pi]
            
            robot_state = [distance, theta, 0.0, 0.0]
            observation = np.concatenate([laser_state, robot_state])
            observations[agent] = observation
            rospy.loginfo(f"Agent {agent} initial observation: {observation}")
            
            # 初始化轨迹
            self.trajectories[i].points = []
            self.record_trajectory(i)
        return observations, info

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
        """
        计算奖励。
        """
        reward = 0.0
        
        # 如果达到了目标
        if target:
            if not self.goal_reward_given[car_index]:
                reward += GOAL_REWARD
                self.goal_reward_given[car_index] = True
        
        # 如果发生碰撞
        if collision:
            reward += COLLISION_PENALTY
        else:
            # 奖励设计示例：鼓励前进，惩罚大转弯和靠近障碍物
            reward += 0.5 * action[0] - 0.1 * abs(action[1])
            reward -= 0.05 * (1.0 / (min_laser + 1e-6))
        
        return reward

    def set_goal_marker(self, index, marker):
        """
        发布目标位置Marker。
        """
        car_name = self.agents[index].replace("agent_", "car")
        self.trajectory_pubs[index].publish(marker)

    def reset_car_position(self, index):
        """
        重置指定智能体的位置和朝向到起始状态。
        """
        angle = self.start_orientations[index]
        q = Quaternion.from_euler(0, 0, angle)
        
        # 重新设置位置和朝向
        self.set_self_states[index].pose.position.x = self.start_positions[index][0]
        self.set_self_states[index].pose.position.y = self.start_positions[index][1]
        self.set_self_states[index].pose.position.z = self.start_positions[index][2]
        self.set_self_states[index].pose.orientation.x = q.x
        self.set_self_states[index].pose.orientation.y = q.y
        self.set_self_states[index].pose.orientation.z = q.z
        self.set_self_states[index].pose.orientation.w = q.w
        
        # 发布模型状态
        self.set_states[index].publish(self.set_self_states[index])
        self.odom_positions[index][0] = self.start_positions[index][0]
        self.odom_positions[index][1] = self.start_positions[index][1]
        rospy.loginfo(f"Agent {self.agents[index]} reset to start position: ({self.start_positions[index][0]}, {self.start_positions[index][1]})")
        
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
