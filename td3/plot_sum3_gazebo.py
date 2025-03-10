import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import hypot

# 地图文件路径
file_map = '/home/xzh/xzh/drl-robot-navigation/catkin_ws/src/multi_robot_scenario/maps/world1.png'

# 基础路径
# base_path = '/home/xzh/xzh/madrl-navigation/data_processing/mtd3_test/world1'
base_path = '/home/xzh/xzh/madrl-navigation/data_processing/matd3_test/world1'


# mtd3_sum3指定车辆的轨迹编号
# car_paths = {
#     'Car1': [3,2, 8],  # 绘制 Car1 的第 5 条和第 9 条轨迹
#     'Car2': [7,3, 6],  # 绘制 Car2 的第 2 条和第 4 条轨迹
#     'Car3': [7,2, 3]   # 绘制 Car3 的第 1 条和第 3 条轨迹
# }
car_paths = {
    'Car1': [1,2, 3],  # 绘制 Car1 的第 5 条和第 9 条轨迹
    'Car2': [1,2, 3],  # 绘制 Car2 的第 2 条和第 4 条轨迹
    'Car3': [1,2, 3]   # 绘制 Car3 的第 1 条和第 3 条轨迹
}

# 检查地图文件是否存在
if not os.path.exists(file_map):
    raise FileNotFoundError(f"Map file not found: {file_map}")

# 加载地图并预处理
img = cv2.imread(file_map, cv2.IMREAD_GRAYSCALE)
resolution = 0.05
height, width = img.shape

# 给地图四周添加边界
img[:, [0, -1]] = 0
img[[0, -1], :] = 0

# 提取障碍物的坐标点
point = [
    [(j - (width - 1) / 2) * resolution, -(i - (height - 1) / 2) * resolution]
    for i in range(height) for j in range(width) if img[i, j] < 127
]

# 配置Matplotlib绘图参数
plt.figure(figsize=(10, 10), dpi=80)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

# 绘制障碍物散点图
point_array = np.array(point)
plt.scatter(point_array[:, 0], point_array[:, 1], c='k', s=1)

# 定义每辆车的颜色
car_colors = {
    'Car1': 'red',
    'Car2': 'blue',
    'Car3': 'green'
}

# 绘制每辆车的轨迹
for car, trajectory_ids in car_paths.items():
    car_color = car_colors[car]
    for trajectory_id in trajectory_ids:
        file_path = os.path.join(base_path, f'path_{car.lower()}_{trajectory_id}.txt')
        if not os.path.exists(file_path):
            print(f"Warning: Path file not found - {file_path}")
            continue

        # 加载轨迹文件，跳过标题行
        path = np.loadtxt(file_path, delimiter=',', usecols=(0, 1), skiprows=1)
        if path.shape[0] < 2:
            print(f"Insufficient data in {file_path}. Skipping this trajectory.")
            continue

        # 计算路径长度和平均步长
        total_length = sum(
            hypot(path[i + 1, 0] - path[i, 0], path[i + 1, 1] - path[i, 1])
            for i in range(path.shape[0] - 1)
        )
        average_step_length = total_length / path.shape[0]

        # 打印路径信息
        print(f'{car} Trajectory {trajectory_id} ({file_path}):')
        print(f'  Total length: {total_length:.2f} meters')
        print(f'  Average step length: {average_step_length:.4f} meters\n')

        # 绘制轨迹
        x, y = path[:, 0], path[:, 1]
        plt.plot(x, y, color=car_color, alpha=0.6, linewidth=2)

        # 标记起点和终点
        plt.scatter(x[0], y[0], color=car_color, s=100, marker='o', 
                   edgecolors='k')  # 起点
        plt.scatter(x[-1], y[-1], color=car_color, s=100, marker='s', 
                   edgecolors='k')  # 终点

# 添加图例（每辆车只需要一个图例）
for car, color in car_colors.items():
    plt.plot([], [], color=color, label=car, linewidth=2)  # 添加一个空线段用于图例

# 添加图例和标签
plt.legend(loc='upper right', fontsize=12)
plt.xlabel('x/m', fontsize=20)
plt.ylabel('y/m', fontsize=20)
plt.title('Trajectories of Selected Cars', fontsize=24)

# 设置坐标轴范围（可选，可根据需要调整）
plt.axis('equal')

# 显示图像
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()