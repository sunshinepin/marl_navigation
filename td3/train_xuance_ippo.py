#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from copy import deepcopy
from xuance.common import get_configs
from xuance.environment import make_envs, REGISTRY_MULTI_AGENT_ENV
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import IPPO_Agents
from xuance_marl_sum3_env import MyNewMultiAgentEnv  # 你的自定义环境

def parse_args():
    parser = argparse.ArgumentParser("IPPO Training for SUM3 Environment")
    parser.add_argument("--config", type=str, default="/home/xzh/xzh/madrl-navigation/td3/sum3_ippo.yaml",
                        help="Path to the configuration YAML file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--test", action="store_true", help="Run in test mode instead of training.")
    parser.add_argument("--benchmark", action="store_true", help="Run in benchmark mode with evaluation.")
    return parser.parse_args()

def main(args):
    # 加载配置文件
    configs_dict = get_configs(file_dir=args.config)
    configs = argparse.Namespace(**configs_dict)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 注册自定义环境
    REGISTRY_MULTI_AGENT_ENV[configs.env_name] = MyNewMultiAgentEnv
    
    # 创建环境
    envs = make_envs(configs)
    
    # 创建智能体
    Agent = IPPO_Agents(config=configs, envs=envs)
    
    # 打印训练信息
    train_info = {
        "Algorithm": configs.agent,
        "Environment": configs.env_name,
        "Device": configs.device,
        "Seed": args.seed
    }
    print("Training Information:")
    for k, v in train_info.items():
        print(f"  {k}: {v}")
    
    # 根据模式执行
    if args.benchmark:
        # 基准测试模式：训练并定期评估，保存最佳模型
        train_steps = configs.running_steps // configs.parallels
        eval_interval = configs.eval_interval // configs.parallels
        num_epoch = int(train_steps / eval_interval)
        test_episode = configs.test_episode
        best_scores_info = {"mean": -float('inf'), "std": 0, "step": 0}
        
        def env_fn():
            configs_test = deepcopy(configs)
            configs_test.parallels = test_episode
            return make_envs(configs_test)
        
        for i_epoch in range(num_epoch):
            print(f"\nEpoch: {i_epoch + 1}/{num_epoch}")
            Agent.train(eval_interval)
            test_scores = Agent.test(env_fn, test_episode)
            mean_score = np.mean(test_scores)
            print(f"Test Scores - Mean: {mean_score:.2f}, Std: {np.std(test_scores):.2f}")
            
            if mean_score > best_scores_info["mean"]:
                best_scores_info = {
                    "mean": mean_score,
                    "std": np.std(test_scores),
                    "step": Agent.current_step
                }
                Agent.save_model("sum3_best_model.pth")
                print("Saved new best model!")
        
        print(f"\nBenchmark Finished!")
        print(f"Best Model Score: {best_scores_info['mean']:.2f}, Std: {best_scores_info['std']:.2f}, "
              f"Step: {best_scores_info['step']}")
    
    elif args.test:
        # 测试模式：加载模型并评估
        def env_fn():
            configs.parallels = configs.test_episode
            return make_envs(configs)
        
        Agent.load_model(path=Agent.model_dir_load)  # 确保 model_dir_load 已正确设置
        scores = Agent.test(env_fn, configs.test_episode)
        print(f"\nTest Results:")
        print(f"Mean Score: {np.mean(scores):.2f}, Std: {np.std(scores):.2f}")
        print("Testing finished!")
    
    else:
        # 普通训练模式
        Agent.train(configs.running_steps // configs.parallels)
        Agent.save_model("sum3_train_model.pth")
        print("\nTraining finished!")
    
    # 清理资源
    Agent.finish()

if __name__ == "__main__":
    args = parse_args()
    main(args)
    