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
from visualization_msgs.msg import Marker
from xuance.environment import RawMultiAgentEnv
import yaml
from gym.spaces import Box
#没有随机初始化起始位置，
# 常量定义
GOAL_REACHED_DIST = 0.3
TIME_DELTA = 0.1
MAX_STEPS = 500
COLLISION_DIST = 0.35
COLLISION_PENALTY = -100.0  # 碰撞惩罚
GOAL_REWARD = 100.0  # 达到目标的奖励
STAY_REWARD = 1.0  # 停留在目标区域的奖励
MIN_GOAL_DISTANCE = 1.0  # 目标之间的最小距离

def check_pos(x, y):
    goal_ok = True
    if x > 9.5 or x < -9.5 or y > 9.5 or y < -9.5:
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
        # rospy.loginfo(f"odom_positions initialized: {self.odom_positions}")

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
        # rospy.loginfo(f"Model states initialized: {self.set_self_states}")

        # 初始化ROS节点和发布者/订阅者
        rospy.init_node("gym_env", anonymous=True)
        
        self.vel_pubs = []
        self.set_states = []
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
            self.set_states.append(rospy.Publisher(
                "/gazebo/set_model_state", ModelState, queue_size=10
            ))
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
        global_state = []
        
        # 为每个智能体收集状态信息
        for i in range(self.num_agents):
            # 激光雷达数据
            laser_state = self.velodyne_data[i]
            
            # 位置和目标信息
            position = self.odom_positions[i]
            goal_position = self.goal_positions[i]
            distance_to_goal = np.linalg.norm([position[0] - goal_position[0], position[1] - goal_position[1]])
            
            # 获取实时朝向
            if self.last_odom[i] is not None:
                q = self.last_odom[i].pose.pose.orientation
                quaternion = Quaternion(q.w, q.x, q.y, q.z)
                angle = quaternion.to_euler()[2]  # yaw 角
            else:
                angle = self.start_orientations[i]  # 回退到初始朝向
            
            # 计算目标角度偏差
            skew_x = goal_position[0] - position[0]
            skew_y = goal_position[1] - position[1]
            beta = np.arctan2(skew_y, skew_x)  # 简化角度计算，使用 atan2
            theta = beta - angle
            theta = (theta + np.pi) % (2 * np.pi) - np.pi  # 归一化到 [-pi, pi]
            
            # 添加最近的动作（假设步进后动作保留，或设为 0）
            # 这里假设有一个 self.last_actions 记录上一步动作
            if hasattr(self, 'last_actions') and self.last_actions[i] is not None:
                linear_vel, angular_vel = self.last_actions[i]
            else:
                linear_vel, angular_vel = 0.0, 0.0  # 默认值
            
            # 构建智能体状态（与 observation_space 一致）
            agent_state = np.concatenate([laser_state, [distance_to_goal, theta, linear_vel, angular_vel]])
            global_state.append(agent_state)
        
        # 拼接成全局状态
        return np.concatenate(global_state)


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
        执行动作，更新环境状态，并返回新的观测、奖励、完成标志和信息。
        action_dict: dict，键为agent名称，值为动作
        """
        self._current_step += 1
        observation = {}
        rewards = {}
        truncated = False if self._current_step < self.max_episode_steps else True
        dones = {}
        infos = {}

        # 应用动作
        for i, agent in enumerate(self.agents):
            action = action_dict[agent]

            # 确保动作符合该智能体的动作空间
            action[0] = np.clip(action[0], self.action_space[agent].low[0], self.action_space[agent].high[0])
            action[1] = np.clip(action[1], self.action_space[agent].low[1], self.action_space[agent].high[1])

            # 如果目标未达，发布控制命令
            if not self.target_reached[i]:
                vel_cmd = Twist()
                vel_cmd.linear.x = float(action[0])
                vel_cmd.angular.z = float(action[1])
                self.vel_pubs[i].publish(vel_cmd)
            else:
                # 如果目标已达，停止智能体
                vel_cmd = Twist()
                vel_cmd.linear.x = 0.0
                vel_cmd.angular.z = 0.0
                self.vel_pubs[i].publish(vel_cmd)

        # 等待一段时间
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

        # 收集观测、计算奖励并检查完成标志
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
            if distance < GOAL_REACHED_DIST and not self.target_reached[i]:
                self.target_reached[i] = True
                rospy.loginfo(f"Agent {i} reached the goal.")

            # 检查是否超过最大步数
            done_episode = self.steps >= self.max_episode_steps
            dones[agent] = done_episode or done
            if done_episode:
                rospy.loginfo("达到最大步数")

            # 处理碰撞
            if collision:
                self.reset_car_position(i)
                dones[agent] = True  # 碰撞后结束智能体回合

            # 构建观测
            laser_state = self.velodyne_data[i]
            robot_state = [distance, theta, action_dict[agent][0], action_dict[agent][1]]
            observation[agent] = np.concatenate([laser_state, robot_state])

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

            # 记录轨迹
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

    def reset(self):
        """
        重置环境到初始状态，并返回初始观测。
        每次重置时随机生成起始位置和目标位置。
        """
        self.steps = 0
        self._current_step = 0
        self.target_reached = [False for _ in range(self.num_agents)]
        self.goal_reward_given = [False for _ in range(self.num_agents)]

        # 重置仿真
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("调用 /gazebo/reset_world 服务失败")

        # 允许仿真更新
        try:
            self.unpause()
        except rospy.ServiceException as e:
            rospy.logerr("调用 /gazebo/unpause_physics 服务失败")
        rospy.sleep(TIME_DELTA * 2)  # 增加等待时间以确保仿真稳定

        # 先生成所有目标位置
        existing_goals = []
        for i in range(self.num_agents):
            gx, gy = generate_unique_goal(existing_goals, self.lower, self.upper, [])
            self.goal_positions[i] = [gx, gy]
            existing_goals.append((gx, gy))
            rospy.loginfo(f"Agent {self.agents[i]} assigned goal position: ({gx}, {gy})")

        # 再生成所有随机起始位置，确保与目标位置保持距离
        existing_starts = []
        for i in range(self.num_agents):
            sx, sy = generate_unique_goal(existing_starts, self.lower, self.upper, existing_goals)
            self.start_positions[i] = [sx, sy, 0.01]  # z 设为 0.01 与环境一致
            existing_starts.append((sx, sy))
            rospy.loginfo(f"Agent {self.agents[i]} assigned random start position: ({sx}, {sy})")

            # 设置模型状态到随机起始位置
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

        # 等待仿真更新
        rospy.sleep(TIME_DELTA)

        # 暂停仿真
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
            theta = (theta + np.pi) % (2 * np.pi) - np.pi  # 归一化到 [-pi, pi]

            robot_state = [distance, theta, 0.0, 0.0]
            observations[agent] = np.concatenate([laser_state, robot_state])

            # 重置轨迹
            self.trajectories[i].points = []
            self.record_trajectory(i)

        return observations, {}


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
        # rospy.loginfo(f"Agent {self.agents[index]} reset to start position: ({self.start_positions[index][0]}, {self.start_positions[index][1]})")
        
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
