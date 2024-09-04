import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# 读取数据文件
file_path = '/home/xzh/xzh/drl-robot-navigation/td3/bizhang_sum3/training_log_726.txt'  # 替换为你的文件路径
episodes = []
car1_rewards = []
car2_rewards = []
car3_rewards = []

with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if "Car1 Episode" in line:
            parts = line.split()
            episode = int(parts[2][:-1])
            reward1 = float(parts[4])
            episodes.append(episode)
            car1_rewards.append(reward1)
        elif "Car2 Episode" in line:
            parts = line.split()
            reward2 = float(parts[4])
            car2_rewards.append(reward2)
        elif "Car3 Episode" in line:
            parts = line.split()
            reward3 = float(parts[4])
            car3_rewards.append(reward3)

# 将数据转换为numpy数组
episodes = np.array(episodes)
car1_rewards = np.array(car1_rewards)
car2_rewards = np.array(car2_rewards)
car3_rewards = np.array(car3_rewards)

# 生成平滑曲线
def smooth_data(x, y, smooth_factor=300):
    x_smooth = np.linspace(x.min(), x.max(), smooth_factor)
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_smooth)
    return x_smooth, y_smooth

# 调整平滑度
smooth_factor = 2000  # 增加平滑因子

x_smooth, car1_smooth = smooth_data(episodes, car1_rewards, smooth_factor)
_, car2_smooth = smooth_data(episodes, car2_rewards, smooth_factor)
_, car3_smooth = smooth_data(episodes, car3_rewards, smooth_factor)

# 多项式拟合
poly_degree = 5  # 选择多项式的阶数
p1 = np.poly1d(np.polyfit(episodes, car1_rewards, poly_degree))
p2 = np.poly1d(np.polyfit(episodes, car2_rewards, poly_degree))
p3 = np.poly1d(np.polyfit(episodes, car3_rewards, poly_degree))

# 创建图表
plt.figure(figsize=(10, 6))

# 画出每辆车的平滑奖励变化曲线
# plt.plot(x_smooth, car1_smooth, label='Car1 Smooth')
plt.plot(x_smooth, car2_smooth, label='Car2 Smooth')
plt.plot(x_smooth, car3_smooth, label='Car3 Smooth')

# 画出每辆车的拟合曲线
# plt.plot(episodes, p1(episodes), '--', label='Car1 Fit', color='blue')
plt.plot(episodes, p2(episodes), '--', label='Car2 Fit', color='orange')
plt.plot(episodes, p3(episodes), '--', label='Car3 Fit', color='green')

# 设置图表标题和标签
plt.title('Reward per Episode for Each Car')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()

# 显示图表
plt.grid(True)
plt.show()
