import re
import matplotlib.pyplot as plt

# 定义输入和输出文件路径
input_file_path = '/home/xzh/xzh/drl-robot-navigation/td3/bizhang_sum3/726.txt'  # 替换为你的输入文件路径
output_file_path = '/home/xzh/xzh/drl-robot-navigation/td3/bizhang_sum3/extracted_data.txt'  # 这是将要保存提取数据的文件路径

# 定义提取数据的正则表达式模式
pattern = re.compile(r"Epoch (\d+): ([\d\.]+), ([\d\.]+)")

# 打开输入文件并读取数据
with open(input_file_path, 'r') as file:
    data = file.read()

# 使用正则表达式提取数据
matches = pattern.findall(data)

# 打开输出文件并保存提取的数据
with open(output_file_path, 'w') as file:
    for match in matches:
        file.write(f"{match[0]},{match[1]},{match[2]}\n")

print(f"数据已提取并保存到 {output_file_path}")

# 读取提取的数据
def read_data(file_path):
    epochs = []
    average_rewards = []
    other_metrics = []

    with open(file_path, 'r') as file:
        for line in file:
            epoch, avg_reward, other_metric = line.strip().split(',')
            epochs.append(int(epoch))
            average_rewards.append(float(avg_reward))
            other_metrics.append(float(other_metric))

    return epochs, average_rewards, other_metrics

epochs, average_rewards, other_metrics = read_data(output_file_path)

# 绘制双轴图表
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Average Reward', color=color)
ax1.plot(epochs, average_rewards, label='Average Reward', marker='o', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # 创建第二个轴
color = 'tab:red'
ax2.set_ylabel('Other Metric', color=color)
ax2.plot(epochs, other_metrics, label='Other Metric', marker='x', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # 调整布局
plt.title('Average Reward and Other Metric over Epochs')
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

plt.show()
