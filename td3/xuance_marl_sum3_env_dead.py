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
    a = 0.5
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
        self.direction_reward = 5.0  # 朝目标方向移动的奖励值，可调整

        # 从配置中获取固定的起始位置、朝向和目标位置
        self.start_positions = env_config.car_positions  # 期望格式: [[x1, y1, z1], [x2, y2, z2], ...]
        self.start_orientations = env_config.car_orientations  # 期望格式: [angle1, angle2, ...]
        self.goal_positions = env_config.goal_positions  # 期望格式: [[gx1, gy1], [gx2, gy2], ...]
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
        self.goal_marker_pubs = []
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
        if not hasattr(self, 'odom_positions'):
            rospy.logerr(f"odom_positions attribute is missing for index {index}")
            return
        if index >= len(self.odom_positions):
            rospy.logerr(f"Index {index} out of range for odom_positions")
            return
        with open(self.trajectory_files[index], "a") as file:
            file.write(f"{self.odom_positions[index][0]},{self.odom_positions[index][1]}\n")
        point = Point()
        point.x = self.odom_positions[index][0]
        point.y = self.odom_positions[index][1]
        point.z = 0
        self.trajectories[index].points.append(point)
        self.trajectories[index].header.stamp = rospy.Time.now()
        self.trajectory_pubs[index].publish(self.trajectories[index])

    def velodyne_callback(self, msg, index):
        data = list(pc2.read_points(msg, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data[index] = np.ones(20) * 10
        for point in data:
            if point[2] > -0.2:
                x, y, z = point
                dist = np.linalg.norm([x, y, z])
                angle = np.arctan2(y, x)
                gap_size = 2 * np.pi / 20
                bin_idx = int((angle + np.pi) / gap_size) % 20
                self.velodyne_data[index][bin_idx] = min(self.velodyne_data[index][bin_idx], dist)

    def odom_callback(self, od_data, index):
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

        all_dead = True
        for i, agent in enumerate(self.agents):
            if self.alive[i]:
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
                    rospy.loginfo(f"Agent {i} reached the goal and died.")

                if collision:
                    self.alive[i] = False
                    rospy.loginfo(f"Agent {i} died due to collision")

                all_dead = False

            done_episode = self._current_step >= self.max_episode_steps
            dones[agent] = not self.alive[i]
            truncated = truncated or all_dead

            laser_state = self.velodyne_data[i]
            robot_state = [distance, theta, action_dict[agent][0], action_dict[agent][1]] if self.alive[i] else [0.0, 0.0, 0.0, 0.0]
            observation[agent] = np.concatenate([laser_state, robot_state])

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
        min_laser = np.min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser
    
    def get_reward(self, target, collision, action, min_laser, distance, car_index):
        reward = 0.0

        current_pos = self.odom_positions[car_index]
        goal_pos = self.goal_positions[car_index]
        
        goal_direction = np.array(goal_pos) - np.array(current_pos)
        goal_direction = goal_direction / (np.linalg.norm(goal_direction) + 1e-6)
        
        displacement = np.array(current_pos) - np.array(self.prev_positions[car_index])
        displacement_norm = np.linalg.norm(displacement)
        
        if displacement_norm > 0:
            displacement_direction = displacement / displacement_norm
            dot_product = np.dot(displacement_direction, goal_direction)
            if dot_product > 0:
                reward += self.direction_reward

        self.prev_positions[car_index] = current_pos.copy()

        if target and not self.goal_reward_given[car_index]:
            reward += GOAL_REWARD
            self.goal_reward_given[car_index] = True
        elif target:
            reward += 1.0

        if collision:
            reward += COLLISION_PENALTY
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            reward += action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2 

        if all(self.target_reached):
            reward += 50.0

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

        # 使用配置中的固定目标位置并发布Marker
        for i in range(self.num_agents):
            gx, gy = self.goal_positions[i]
            if not check_pos(gx, gy):
                rospy.logwarn(f"Goal position for {self.agents[i]} at ({gx}, {gy}) is in a forbidden area!")
            rospy.loginfo(f"Agent {self.agents[i]} assigned goal position: ({gx}, {gy})")
            self.publish_goal_marker(i)

        # 使用配置中的固定起始位置和朝向
        for i in range(self.num_agents):
            sx, sy, sz = self.start_positions[i]
            angle = self.start_orientations[i]
            q = Quaternion.from_euler(0, 0, angle)
            self.set_self_states[i].pose.position.x = sx
            self.set_self_states[i].pose.position.y = sy
            self.set_self_states[i].pose.position.z = 0.1
            self.set_self_states[i].pose.orientation.x = q.x
            self.set_self_states[i].pose.orientation.y = q.y
            self.set_self_states[i].pose.orientation.z = q.z
            self.set_self_states[i].pose.orientation.w = q.w
            self.set_states[i].publish(self.set_self_states[i])
            self.odom_positions[i] = [sx, sy]  # 初始化odom位置
            rospy.loginfo(f"Agent {self.agents[i]} set to start position: ({sx}, {sy}, {sz}) and orientation: {angle:.2f}")

        rospy.sleep(TIME_DELTA)

        try:
            self.pause()
        except rospy.ServiceException as e:
            rospy.logerr("调用 /gazebo/pause_physics 服务失败")

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

    # def reset_car_position(self, index):
    #     # 在碰撞时恢复到初始位置，而不是随机位置
    #     sx, sy, sz = self.start_positions[index]
    #     angle = self.start_orientations[index]
    #     q = Quaternion.from_euler(0, 0, angle)

    #     self.set_self_states[index].pose.position.x = sx
    #     self.set_self_states[index].pose.position.y = sy
    #     self.set_self_states[index].pose.position.z = sz
    #     self.set_self_states[index].pose.orientation.x = q.x
    #     self.set_self_states[index].pose.orientation.y = q.y
    #     self.set_self_states[index].pose.orientation.z = q.z
    #     self.set_self_states[index].pose.orientation.w = q.w

    #     self.set_states[index].publish(self.set_self_states[index])
    #     self.odom_positions[index] = [sx, sy]
    #     rospy.loginfo(f"Agent {self.agents[index]} reset to initial position: ({sx}, {sy}, {sz}) and orientation: {angle:.2f}")

    #     try:
    #         self.unpause()
    #     except rospy.ServiceException as e:
    #         rospy.logerr("调用 /gazebo/unpause_physics 服务失败")

    #     rospy.sleep(TIME_DELTA)

    #     try:
    #         self.pause()
    #     except rospy.ServiceException as e:
    #         rospy.logerr("调用 /gazebo/pause_physics 服务失败")

    def render(self, *args, **kwargs):
        return np.ones([64, 64, 64])

    def close(self):
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
        rospy.loginfo(f"Published goal marker for {self.agents[index]} at ({self.goal_positions[index][0]}, {self.goal_positions[index][1]})")