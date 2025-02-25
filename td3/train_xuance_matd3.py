#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from copy import deepcopy
from xuance.common import get_configs
from xuance.environment import make_envs, REGISTRY_MULTI_AGENT_ENV
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import MATD3_Agents
from xuance_marl_sum3_env import MyNewMultiAgentEnv  # 您的自定义环境

def parse_args():
    parser = argparse.ArgumentParser("MATD3 训练 SUM3 环境")
    parser.add_argument("--config", type=str, default="/home/xzh/xzh/madrl-navigation/td3/sum3_matd3.yaml",
                        help="配置文件 YAML 的路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，用于确保实验可重复")
    parser.add_argument("--test", action="store_true", help="运行测试模式而不是训练模式")
    parser.add_argument("--benchmark", action="store_true", help="运行基准测试模式并进行评估")
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
    
    # 创建 MATD3 智能体
    Agent = MATD3_Agents(config=configs, envs=envs)
    
    # 打印训练信息
    train_info = {
        "算法": configs.agent,
        "环境": configs.env_name,
        "设备": configs.device,
        "随机种子": args.seed
    }
    print("训练信息：")
    for k, v in train_info.items():
        print(f"  {k}: {v}")
    
    # 根据模式执行
    if args.benchmark:
        # 基准测试模式：训练并定期评估，保存最佳模型
        train_steps = configs.running_steps // configs.parallels  # 总训练步数
        eval_interval = configs.eval_interval // configs.parallels  # 评估间隔
        num_epoch = int(train_steps / eval_interval)  # 总轮数
        test_episode = configs.test_episode  # 测试回合数
        best_scores_info = {"mean": -float('inf'), "std": 0, "step": 0}  # 最佳得分记录
        
        def env_fn():
            configs_test = deepcopy(configs)
            configs_test.parallels = test_episode
            return make_envs(configs_test)
        
        # 初始测试
        test_scores = Agent.test(env_fn, test_episode)
        best_scores_info = {"mean": np.mean(test_scores), "std": np.std(test_scores), "step": Agent.current_step}
        Agent.save_model("sum3_initial_model.pth")
        print(f"初始测试得分 - 平均值: {best_scores_info['mean']:.2f}, 标准差: {best_scores_info['std']:.2f}")
        
        for i_epoch in range(num_epoch):
            print(f"\n轮次: {i_epoch + 1}/{num_epoch}")
            Agent.train(eval_interval)
            test_scores = Agent.test(env_fn, test_episode)
            mean_score = np.mean(test_scores)
            print(f"测试得分 - 平均值: {mean_score:.2f}, 标准差: {np.std(test_scores):.2f}")
            
            if mean_score > best_scores_info["mean"]:
                best_scores_info = {
                    "mean": mean_score,
                    "std": np.std(test_scores),
                    "step": Agent.current_step
                }
                Agent.save_model("sum3_best_model.pth")
                print("保存新的最佳模型！")
        
        print(f"\n基准测试完成！")
        print(f"最佳模型得分: {best_scores_info['mean']:.2f}, 标准差: {best_scores_info['std']:.2f}, "
              f"步数: {best_scores_info['step']}")
    
    elif args.test:
        # 测试模式：加载模型并评估
        def env_fn():
            configs.parallels = configs.test_episode
            return make_envs(configs)
        
        Agent.load_model(path=Agent.model_dir_load)  # 确保配置文件中指定了 model_dir_load
        scores = Agent.test(env_fn, configs.test_episode)
        print(f"\n测试结果：")
        print(f"平均得分: {np.mean(scores):.2f}, 标准差: {np.std(scores):.2f}")
        print("测试完成！")
    
    else:
        # 训练模式：训练并保存最终模型
        Agent.train(configs.running_steps // configs.parallels)
        Agent.save_model("sum3_train_model.pth")
        print("\n训练完成！")
    
    # 清理资源
    Agent.finish()

if __name__ == "__main__":
    args = parse_args()
    main(args)