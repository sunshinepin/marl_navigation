import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 定义障碍物的边界框
obstacles = [
    (1.5, 4.5, 1.5, 4.5),  # (x_min, x_max, y_min, y_max)
    (-5.0, -0.5, 1.5, 4.5),
    (-5.5, -2.5, 0.5, 5.0),
    (0.5, 5.0, -5.5, -2.5),
    (2.5, 5.5, -5.0, -0.5),
    (-4.5, -1.5, -4.5, -1.5),
    (-7.5, -5.5, 5.5, 7.5),
    (-5 - 0.5, -5 + 0.5, 3.5 - 0.5, 3.5 + 0.5),  # 新增障碍物
    (-5 - 0.5, -5 + 0.5, -6.5 - 0.5, -6.5 + 0.5),  # 新增障碍物
    (5 - 0.5, 5 + 0.5, 5.5 - 0.5, 5.5 + 0.5),  # 新增障碍物
    (6 - 0.5, 6 + 0.5, -6 - 0.5, -6 + 0.5)  # 新增障碍物
]

# 定义边界
boundary = (-6.5, 6.5, -6.5, 6.5)

# 创建图形和坐标轴
fig, ax = plt.subplots()
ax.set_xlim(boundary[0], boundary[1])
ax.set_ylim(boundary[2], boundary[3])

# 添加障碍物
for obs in obstacles:
    rect = patches.Rectangle((obs[0], obs[2]), obs[1] - obs[0], obs[3] - obs[2], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

# 添加边界
rect = patches.Rectangle((boundary[0], boundary[2]), boundary[1] - boundary[0], boundary[3] - boundary[2], linewidth=2, edgecolor='b', facecolor='none')
ax.add_patch(rect)

# 添加标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Obstacle Positions')

# 显示图形
plt.grid(True)
plt.show()
