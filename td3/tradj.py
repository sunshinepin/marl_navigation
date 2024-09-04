import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 读取背景图
img = mpimg.imread('background.png')

# 读取car1的轨迹数据
x_coords_car1 = []
y_coords_car1 = []
with open('trajectory_car1.txt', 'r') as file:
    for line in file:
        x, y = map(float, line.strip().split(','))
        x_coords_car1.append(x)
        y_coords_car1.append(y)

# 读取car2的轨迹数据
x_coords_car2 = []
y_coords_car2 = []
with open('trajectory_car2.txt', 'r') as file:
    for line in file:
        x, y = map(float, line.strip().split(','))
        x_coords_car2.append(x)
        y_coords_car2.append(y)

# 绘制背景图
plt.imshow(img, extent=[-5, 5, -5, 5])  # 根据场景大小调整extent参数

# 绘制car1的轨迹
plt.plot(x_coords_car1, y_coords_car1, 'r-', linewidth=2, label='Car1 Trajectory')

# 绘制car2的轨迹
plt.plot(x_coords_car2, y_coords_car2, 'b-', linewidth=2, label='Car2 Trajectory')

# 设置图例
plt.legend()

# 设置图表标题和轴标签
plt.title('Robot Trajectories')
plt.xlabel('X')
plt.ylabel('Y')

# 显示图表
plt.show()
