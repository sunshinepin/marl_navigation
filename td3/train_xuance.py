# train_xuance.py

import argparse
from xuance.common import get_configs
from xuance.environment import make_envs
from xuance.torch.agents import IPPO_Agents
import env_registration  # 导入并执行注册脚本

# 注册自定义环境
env_registration.register_env()

# 读取配置文件
configs_dict = get_configs(file_dir="/home/xzh/xzh/madrl-navigation/td3/sum3_agent_env.yaml")  # 替换为你的配置文件路径
configs = argparse.Namespace(**configs_dict)

# 创建环境
envs = make_envs(configs)  # 创建并行环境

# 创建智能体
Agent = IPPO_Agents(config=configs, envs=envs)  # 选择适当的智能体类型

# 训练模型
Agent.train(configs.running_steps // configs.parallels)  # 训练步骤数根据并行环境数量调整

# 保存模型
Agent.save_model("final_train_model.pth")  # 保存路径可根据需要调整

# 完成训练
Agent.finish()
