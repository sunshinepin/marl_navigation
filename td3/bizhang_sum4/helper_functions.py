import numpy as np
import math

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.3
COLLISION_PENALTY = -100.0
GOAL_REWARD = 100.0
MIN_GOAL_DISTANCE = 1.0

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

def observe_collision(laser_data):
    min_laser = min(laser_data)
    if min_laser < COLLISION_DIST:
        return True, True, min_laser
    return False, False, min_laser

def get_reward(target, collision, action, min_laser, distance):
    r3 = lambda x: 1 - x if x < 1 else 0.0
    reward = action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2
    if target:
        reward += GOAL_REWARD  # 只在第一次到达目标时给予奖励
    if collision:
        reward += COLLISION_PENALTY
    return reward
