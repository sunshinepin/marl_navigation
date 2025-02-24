#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from copy import deepcopy
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs, REGISTRY_MULTI_AGENT_ENV
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import MADDPG_Agents
from xuance_marl_sum3_env import MyNewMultiAgentEnv  # 你的自定义环境

def parse_args():
    parser = argparse.ArgumentParser("MADDPG for MyNewMultiAgentEnv")
    parser.add_argument("--env-id", type=str, default="MyNewMultiAgentEnv")  # 你的环境ID
    parser.add_argument("--test", type=int, default=0)  # 是否测试模式，0为否，1为是
    parser.add_argument("--benchmark", type=int, default=1)  # 是否跑基准测试，0为否，1为是
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    parser = parse_args()
    
    # 加载配置文件
    configs_dict = get_configs(file_dir="/home/xzh/xzh/madrl-navigation/td3/sum3_maddpg.yaml")
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    # 注册你的自定义环境
    REGISTRY_MULTI_AGENT_ENV[configs.env_name] = MyNewMultiAgentEnv

    # 设置随机种子
    set_seed(configs.seed)

    # 创建环境
    envs = make_envs(configs)

    # 初始化MADDPG智能体
    Agent = MADDPG_Agents(config=configs, envs=envs)

    # 打印训练信息
    train_information = {
        "Deep learning toolbox": configs.dl_toolbox,
        "Calculating device": configs.device,
        "Algorithm": configs.agent,
        "Environment": configs.env_name,
        "Scenario": configs.env_id
    }
    for k, v in train_information.items():
        print(f"{k}: {v}")

    # 根据模式执行不同操作
    if configs.benchmark:
        # Benchmark模式：训练并定期评估，保存最佳模型
        def env_fn():
            configs_test = deepcopy(configs)
            configs_test.parallels = configs_test.test_episode
            return make_envs(configs_test)

        train_steps = configs.running_steps // configs.parallels
        eval_interval = configs.eval_interval // configs.parallels
        test_episode = configs.test_episode
        num_epoch = int(train_steps / eval_interval)

        # 初始测试
        test_scores = Agent.test(env_fn, test_episode)
        Agent.save_model(model_name="best_model.pth")
        best_scores_info = {
            "mean": np.mean(test_scores),
            "std": np.std(test_scores),
            "step": Agent.current_step
        }

        # 训练循环
        for i_epoch in range(num_epoch):
            print(f"Epoch: {i_epoch}/{num_epoch}:")
            Agent.train(eval_interval)
            test_scores = Agent.test(env_fn, test_episode)

            # 如果当前得分更好，更新最佳模型
            if np.mean(test_scores) > best_scores_info["mean"]:
                best_scores_info = {
                    "mean": np.mean(test_scores),
                    "std": np.std(test_scores),
                    "step": Agent.current_step
                }
                Agent.save_model(model_name="best_model.pth")
        
        # 输出最佳模型结果
        print(f"Best Model Score: {best_scores_info['mean']:.2f}, std={best_scores_info['std']:.2f}")

    else:
        if configs.test:
            # 测试模式：加载模型并测试
            def env_fn():
                configs.parallels = configs.test_episode
                return make_envs(configs)

            Agent.load_model(path=Agent.model_dir_load)  # 需要确保model_dir_load在配置文件中定义
            scores = Agent.test(env_fn, configs.test_episode)
            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print("Finish testing.")
        else:
            # 训练模式：完整训练并保存最终模型
            Agent.train(configs.running_steps // configs.parallels)
            Agent.save_model("final_train_model.pth")
            print("Finish training!")

    # 结束
    Agent.finish()