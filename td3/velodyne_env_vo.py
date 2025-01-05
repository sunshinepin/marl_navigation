import math
import os
import random
import subprocess
import time
from os import path

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

GOAL_REACHED_DIST = 0.35
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
    """所有 Gazebo 环境的超类。"""

    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim
        self.odom_x = 0
        self.odom_y = 0

        self.goal_x = 1
        self.goal_y = 0.0

        self.upper = 5.0
        self.lower = -5.0
        self.velodyne_data = np.ones(self.environment_dim) * 10
        self.last_odom = None

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "car1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03

        print("Roscore launched!")
        rospy.init_node("gym", anonymous=True)

        self.vel_pub = rospy.Publisher("/car1/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        self.publisher_vo = rospy.Publisher("vo_markers", MarkerArray, queue_size=3)
        self.velodyne = rospy.Subscriber(
            "/car1/velodyne/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        )
        self.odom = rospy.Subscriber(
            "/car1/odom_gazebo", Odometry, self.odom_callback, queue_size=1
        )

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

    def compute_velocity_obstacles(self, laser_data, max_linear=1.0, max_angular=1.0, time_horizon=1.0):
        """
        计算所有障碍物的速度障碍物（VO）。
        返回所有VO的集合。
        """
        VO = []
        for i, distance in enumerate(laser_data):
            if distance < COLLISION_DIST:
                continue  # 已经处于碰撞距离，无需计算
            angle = self.gaps[i][0] + (self.gaps[i][1] - self.gaps[i][0]) / 2
            # 计算障碍物相对于机器人的位置
            obs_x = distance * math.cos(angle)
            obs_y = distance * math.sin(angle)
            # 假设障碍物为静态障碍物，相对速度为零
            rel_velocity = (0.0, 0.0)
            # 计算VO
            vo = self.calculate_vo(obs_x, obs_y, rel_velocity, max_linear, max_angular, time_horizon)
            if vo:
                VO.append(vo)
        return VO

    def calculate_vo(self, obs_x, obs_y, rel_velocity, max_linear, max_angular, time_horizon):
        """
        计算单个障碍物的速度障碍物（VO）。
        返回VO的圆心和半径。
        """
        # 计算障碍物相对于机器人的距离
        distance = math.hypot(obs_x, obs_y)
        if distance == 0:
            return None  # 避免除以零
        # 计算预测时间内的安全距离
        safety_distance = 0.5  # 可根据需求调整
        # 计算碰撞可能发生的最小时间
        collision_time = distance / (max_linear + 1e-5)  # 避免除以零
        collision_time = min(collision_time, time_horizon)
        # 计算安全半径
        radius = safety_distance + max_linear * collision_time
        # VO的圆心为障碍物位置
        vo_center = (obs_x + rel_velocity[0] * collision_time, obs_y + rel_velocity[1] * collision_time)
        return (vo_center, radius)

    def is_velocity_feasible(self, action_linear, action_angular, VO, max_linear=1.0, max_angular=1.0):
        """
        判断给定的线速度和角速度是否在可行速度空间内。
        """
        # 计算机器人动作的速度向量
        # 简化处理：将角速度转换为线速度影响
        # 这里假设机器人朝向不改变，可以进一步优化
        velocity_vector = (action_linear, action_angular)  # 线速度和角速度组合

        for vo in VO:
            vo_center, vo_radius = vo
            # 计算动作速度向量与VO中心的距离
            distance = math.hypot(velocity_vector[0] - vo_center[0], velocity_vector[1] - vo_center[1])
            if distance < vo_radius:
                return False  # 动作速度在VO内，不可行
        return True  # 动作速度在可行空间内

    def get_reward(self, target, collision, action, min_laser, VO, distance_before, distance_after, max_linear=1.0, max_angular=1.0):
        if target:
            return 100.0  # 达到目标，给予高额奖励
        elif collision:
            return -100.0  # 碰撞，给予高额惩罚
        else:
            action_linear = action[0]
            action_angular = action[1]
            # 判断动作是否在可行速度空间内
            is_feasible = self.is_velocity_feasible(action_linear, action_angular, VO, max_linear, max_angular)
            
            # 奖励与惩罚
            reward = 0.0
            
            if is_feasible:
                reward += 0.5  # 可行动作给予正奖励
            else:
                reward -= 0.5  # 不可行动作给予负奖励
            
            # 奖励智能体缩短与目标的距离
            distance_change = distance_before - distance_after
            reward += distance_change * 8  # 根据需要调整权重
            
            # 惩罚智能体远离目标
            if distance_change < 0:
                reward -= 0.6  # 根据需要调整权重
            
            # 引入方向奖励
            if action_linear > 0:
                reward += action_linear * 2  # 鼓励向前移动
            else:
                reward -= abs(action_linear) * 2  # 惩罚向后移动
            
            # 额外的奖励形状调整，鼓励保持直线和减少转向
            reward += (action_linear / max_linear) - (abs(action_angular) / max_angular)
            
            # 根据最小激光距离进一步调整
            if min_laser < COLLISION_DIST:
                reward -= (COLLISION_DIST - min_laser) / COLLISION_DIST * 5  # 根据需要调整权重
            
            return reward

    def step(self, action):
        target = False

        # 记录动作前的距离
        distance_before = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])

        # 发布机器人动作
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics 服务调用失败")

        # 传播状态 TIME_DELTA 秒
        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics 服务调用失败")

        # 读取 Velodyne 激光雷达数据
        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        v_state = self.velodyne_data.copy()
        laser_state = [v_state]

        # 从里程计数据计算机器人朝向
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

        # 计算机器人到目标的距离
        distance_after = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])

        # 计算机器人朝向与目标方向的相对角度
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(skew_x**2 + skew_y**2)
        mag2 = 1.0  # 机器人前进方向的单位向量
        beta = math.acos(dot / (mag1 * mag2)) if mag1 != 0 else 0.0
        if skew_y < 0:
            beta = -beta
        theta = beta - angle
        theta = (theta + np.pi) % (2 * np.pi) - np.pi  # 归一化到 [-π, π]

        # 检测是否到达目标
        if distance_after < GOAL_REACHED_DIST:
            target = True
            done = True

        # 计算速度障碍物
        VO = self.compute_velocity_obstacles(self.velodyne_data, max_linear=1.0, max_angular=1.0, time_horizon=1.0)

        # 可视化VO
        self.visualize_vo(VO)

        # 使用更新后的 get_reward 函数计算奖励
        reward = self.get_reward(target, collision, action, min_laser, VO, distance_before, distance_after, max_linear=1.0, max_angular=1.0)

        robot_state = [distance_after, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        return state, reward, done, target

    def reset(self):
        # 重置环境状态并返回初始观察
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state

        x = 0
        y = 0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = check_pos(x, y)
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        # 设置一个随机目标点
        self.change_goal()
        # 随机散布箱子
        self.random_box()
        # 如果训练不需要随机箱子就去除
        self.publish_markers([0.0, 0.0])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics 服务调用失败")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics 服务调用失败")

        # 计算初始状态下的速度障碍物
        VO = self.compute_velocity_obstacles(self.velodyne_data, max_linear=1.0, max_angular=1.0, time_horizon=1.0)

        # 可视化VO
        self.visualize_vo(VO)

        v_state = self.velodyne_data.copy()
        laser_state = [v_state]

        # 计算机器人到目标的距离
        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])

        # 计算机器人朝向与目标方向的相对角度
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(skew_x**2 + skew_y**2)
        mag2 = 1.0  # 机器人前进方向的单位向量
        beta = math.acos(dot / (mag1 * mag2)) if mag1 != 0 else 0.0
        if skew_y < 0:
            beta = -beta
        theta = beta - 0.0  # 初始角度为0
        theta = (theta + np.pi) % (2 * np.pi) - np.pi  # 归一化到 [-π, π]

        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)

        return state

    def change_goal(self):
        # 设置新的目标点，并检查其位置不在障碍物上
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        goal_ok = False

        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.lower, self.upper)
            self.goal_y = self.odom_y + random.uniform(self.lower, self.upper)
            goal_ok = check_pos(self.goal_x, self.goal_y)

    def random_box(self):
        # 每次重置时随机更改箱子的位置，随机化训练环境
        for i in range(4):
            name = "cardboard_box_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
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

    def publish_markers(self, action):
        # 在 Rviz 中发布可视化数据
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

    def visualize_vo(self, VO):
        """
        在 Rviz 中可视化速度障碍物（VO）。
        """
        markerArray = MarkerArray()
        for idx, vo in enumerate(VO):
            vo_center, vo_radius = vo
            marker = Marker()
            marker.header.frame_id = "world"
            marker.id = idx
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = vo_radius * 2
            marker.scale.y = vo_radius * 2
            marker.scale.z = 0.1
            marker.color.a = 0.3
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.pose.position.x = vo_center[0]
            marker.pose.position.y = vo_center[1]
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            markerArray.markers.append(marker)
        self.publisher_vo.publish(markerArray)

    @staticmethod
    def observe_collision(laser_data):
        # 从激光雷达数据检测碰撞
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    def reset(self):
        # 重置环境状态并返回初始观察
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state

        x = 0
        y = 0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = check_pos(x, y)
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        # 设置一个随机目标点
        self.change_goal()
        # 随机散布箱子
        self.random_box()
        # 如果训练不需要随机箱子就去除
        self.publish_markers([0.0, 0.0])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics 服务调用失败")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics 服务调用失败")

        # 计算初始状态下的速度障碍物
        VO = self.compute_velocity_obstacles(self.velodyne_data, max_linear=1.0, max_angular=1.0, time_horizon=1.0)

        # 可视化VO
        self.visualize_vo(VO)

        v_state = self.velodyne_data.copy()
        laser_state = [v_state]

        # 计算机器人到目标的距离
        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])

        # 计算机器人朝向与目标方向的相对角度
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(skew_x**2 + skew_y**2)
        mag2 = 1.0  # 机器人前进方向的单位向量
        beta = math.acos(dot / (mag1 * mag2)) if mag1 != 0 else 0.0
        if skew_y < 0:
            beta = -beta
        theta = beta - 0.0  # 初始角度为0
        theta = (theta + np.pi) % (2 * np.pi) - np.pi  # 归一化到 [-π, π]

        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)

        return state

    def compute_feasible_velocities(self, laser_data, max_linear=1.0, max_angular=1.0, time_horizon=1.0):
        """
        计算所有障碍物的速度障碍物（VO）。
        返回可行的速度集合（线速度, 角速度）。
        """
        VO = self.compute_velocity_obstacles(laser_data, max_linear, max_angular, time_horizon)
        feasible_velocities = self.get_feasible_velocity_space(VO, resolution=0.1, max_linear=max_linear, max_angular=max_angular)
        return feasible_velocities

    def get_feasible_velocity_space(self, VO, resolution=0.1, max_linear=1.0, max_angular=1.0):
        """
        生成可行速度空间网格，用于可视化或其他用途。
        """
        feasible_velocities = []
        for linear in np.arange(-max_linear, max_linear + resolution, resolution):
            for angular in np.arange(-max_angular, max_angular + resolution, resolution):
                if self.is_velocity_feasible(linear, angular, VO, max_linear, max_angular):
                    feasible_velocities.append((linear, angular))
        return feasible_velocities
