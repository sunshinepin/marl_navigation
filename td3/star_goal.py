import numpy as np
import os  # 导入 os 模块

# 定义 generate_unique_goal 函数
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

# 定义 check_pos 函数
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

# 定义常量
MIN_GOAL_DISTANCE = 1.0  # 目标之间的最小距离

# 定义生成起始点和目标点的函数
def generate_start_goal_pairs(num_pairs=100):
    start_goal_list = []
    for _ in range(num_pairs):
        # 生成三个车的起始点和目标点
        starts = []
        goals = []
        existing_starts = []
        existing_goals = []
        
        for _ in range(3):  # 三个车
            # 生成起始点
            sx, sy = generate_unique_goal(existing_starts, -5.0, 5.0, existing_goals)
            angle = np.random.uniform(0, 2 * np.pi)
            starts.append([sx, sy, angle, 0.0])
            existing_starts.append((sx, sy))
            
            # 生成目标点
            gx, gy = generate_unique_goal(existing_goals, -5.0, 5.0, existing_starts)
            goals.append([gx, gy, 0.0, 0.0])
            existing_goals.append((gx, gy))
        
        # 将起始点和目标点合并为一个列表
        start_goal_pair = []
        for i in range(3):
            start_goal_pair.extend(starts[i][:3])  # 起始点的x, y, theta
            start_goal_pair.extend(goals[i][:2])   # 目标点的x, y
        start_goal_list.append(start_goal_pair)
    
    return start_goal_list

# 定义保存数据的函数
def save_start_goal_pairs(start_goal_list, file_path):
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # 保存数据到文件
    with open(file_path, 'w') as f:
        for pair in start_goal_list:
            line = ' '.join(map(str, pair)) + '\n'
            f.write(line)

# 生成100组起始点和目标点
start_goal_list = generate_start_goal_pairs(100)

# 保存到文件
log_path = "/home/xzh/xzh/madrl-navigation/data_processing/matd3_test/world1/start_goal.txt"
save_start_goal_pairs(start_goal_list, log_path)

print(f"100组起始点和目标点已保存到 {log_path}")