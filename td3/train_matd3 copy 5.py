#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import MultiAgentReplayBuffer  
from xuance_marl_sum3_env import MyNewMultiAgentEnv
from types import SimpleNamespace
import random 

# 环境配置 这版去除adapt_layer，共享经验吃
env_config_dict = {
    "env_id": "multi_car_env",
    "car_names": ["car1", "car2", "car3"],
    "car_positions": [[0.0, 5.0, 0.01], [0.0, -5.0, 0.01], [-5.0, 0.0, 0.01]],
    "car_orientations": [0, 0, 0],
    "max_episode_steps": 500
}

env_config = SimpleNamespace(**env_config_dict)

# 常量定义
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0
eval_freq = 6e3
max_ep = 500
eval_ep = 10
max_timesteps = 5e6
expl_noise = 1
expl_decay_steps = 500000
expl_min = 0.1
batch_size = 40
discount = 0.99999
tau = 0.005
policy_noise = 0.2
noise_clip = 0.5
policy_freq = 2
buffer_size = 1e6
file_name = "TD3_multi_agent"
save_model = True
load_model = True
random_near_obstacle = True

# 创建存储文件夹
if not os.path.exists("./results"):
    os.makedirs("./results")
if save_model and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

# 创建环境
env = MyNewMultiAgentEnv(env_config)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = 24 + 3 * (env.num_agents - 1)  # 新状态维度，应为 30
action_dim = 2
max_action = 1
num_agents = env.num_agents
old_state_dim = 24

# 修改后的 Actor 网络（移除 adapt_layer）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, old_state_dim=24):
        super(Actor, self).__init__()
        # 移除 adapt_layer，直接使用 state_dim
        self.layer_1 = nn.Linear(state_dim, 800)  # 直接从 state_dim=30 输入到 800
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        # 移除 adapt_layer 的调用，直接从输入 s 开始
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a

# 修改后的 Critic 网络（移除 adapt_layer）
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, old_state_dim=24):
        super(Critic, self).__init__()
        # 移除 adapt_layer，直接使用 state_dim
        self.layer_1 = nn.Linear(state_dim, 800)  # 直接从 state_dim=30 输入到 800
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)
        self.layer_4 = nn.Linear(state_dim, 800)  # 直接从 state_dim=30 输入到 800
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        # 移除 adapt_layer 的调用，直接从输入 s 开始
        s1 = F.relu(self.layer_1(s))
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(s))
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2

# TD3 类
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, old_state_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, old_state_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim, old_state_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, old_state_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.writer = SummaryWriter()
        self.iter_count = 0

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=1, tau=0.005,
              policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        agents = [f"agent_{i}" for i in range(num_agents)]
        for it in range(iterations):
            batch = replay_buffer.sample_batch(batch_size)
            # 假设所有智能体共享策略，随机选择一个智能体的经验训练
            agent_id = random.choice(agents)
            state = torch.Tensor(batch[f'obs_{agent_id}']).to(device)
            next_state = torch.Tensor(batch[f'next_obs_{agent_id}']).to(device)
            action = torch.Tensor(batch[f'actions_{agent_id}']).to(device)
            reward = torch.Tensor(batch[f'rewards_{agent_id}']).to(device)
            done = torch.Tensor(batch[f'dones_{agent_id}']).to(device)

            next_action = self.actor_target(next_state)
            noise = torch.Tensor(action).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            current_Q1, current_Q2 = self.critic(state, action)
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            av_loss += loss
        self.iter_count += 1
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")

    def load(self, filename, directory):
        old_actor_dict = torch.load(f"{directory}/{filename}_actor.pth")
        old_critic_dict = torch.load(f"{directory}/{filename}_critic.pth")
        self.actor.load_state_dict(old_actor_dict, strict=False)
        self.critic.load_state_dict(old_critic_dict, strict=False)
        for name, param in self.actor.named_parameters():
            if "adapt_layer" not in name:
                param.requires_grad = False
        for name, param in self.critic.named_parameters():
            if "adapt_layer" not in name:
                param.requires_grad = False

# 评估函数
def evaluate(network, env, epoch, eval_episodes=10):
    avg_rewards = {agent: 0.0 for agent in env.agents}
    collisions = {agent: 0 for agent in env.agents}
    for _ in range(eval_episodes):
        obs, _ = env.reset()
        dones = {agent: False for agent in env.agents}
        step_count = 0
        while not all(dones.values()) and step_count < env.max_episode_steps:
            actions = {agent: network.get_action(np.array(obs[agent])) 
                      for agent in env.agents if not dones[agent]}
            for agent in env.agents:
                if dones[agent]:
                    actions[agent] = np.zeros(action_dim)
            next_obs, rewards, dones, truncated, infos = env.step(actions)
            for agent in env.agents:
                avg_rewards[agent] += rewards[agent]
                if rewards[agent] < -90:
                    collisions[agent] += 1
            obs = next_obs
            step_count += 1
            if truncated:
                break
    for agent in env.agents:
        avg_rewards[agent] /= eval_episodes
        collisions[agent] /= eval_episodes
    print("..............................................")
    for agent in env.agents:
        print(f"Agent {agent} - Avg Reward: {avg_rewards[agent]:.2f}, Avg Collisions: {collisions[agent]:.2f}, Epoch: {epoch}")
    print("..............................................")
    return avg_rewards

# 初始化网络和缓冲区
network = TD3(state_dim, action_dim, max_action)
replay_buffer = MultiAgentReplayBuffer(buffer_size, num_agents=num_agents, random_seed=seed)

if load_model:
    try:
        network.load(file_name, "./pytorch_models")
        print("Loaded pre-trained model successfully.")
    except Exception as e:
        print(f"Could not load model: {e}. Initializing with random parameters.")

# 训练循环
evaluations = []
timestep = 0
timesteps_since_eval = 0
episode_num = 0
epoch = 1

while timestep < max_timesteps:
    obs, _ = env.reset()
    done = {agent: False for agent in env.agents}
    episode_reward = 0
    episode_timesteps = 0
    count_rand_actions = {agent: 0 for agent in env.agents}
    random_action = {agent: [] for agent in env.agents}

    while not all(done.values()):
        if expl_noise > expl_min:
            expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

        action_dict = {}
        for agent in env.agents:
            if not done[agent]:
                state = obs[agent]
                action = network.get_action(np.array(state))
                action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(-max_action, max_action)
                if random_near_obstacle:
                    if (np.random.uniform(0, 1) > 0.85 and min(state[4:16]) < 0.6 and
                            count_rand_actions[agent] < 1):
                        count_rand_actions[agent] = np.random.randint(8, 15)
                        random_action[agent] = np.random.uniform(-1, 1, 2)
                    if count_rand_actions[agent] > 0:
                        count_rand_actions[agent] -= 1
                        action = random_action[agent]
                        action[0] = -1
                action_dict[agent] = action
            else:
                action_dict[agent] = np.zeros(action_dim)

        next_obs, rewards, dones, truncated, infos = env.step(action_dict)
        episode_reward += sum(rewards.values()) / num_agents

        # 添加多智能体经验
        replay_buffer.add(obs, action_dict, rewards, dones, next_obs)

        obs = next_obs
        done = dones
        episode_timesteps += 1
        timestep += 1
        timesteps_since_eval += 1

        if truncated or all(done.values()):
            network.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
            if timesteps_since_eval >= eval_freq:
                print("Validating")
                timesteps_since_eval = 0
                evaluations.append(evaluate(network, env, epoch, eval_ep))
                network.save(file_name, "./pytorch_models")
                np.save(f"./results/{file_name}", evaluations)
                epoch += 1
            episode_num += 1
            break

# 最终评估和保存
evaluations.append(evaluate(network, env, epoch, eval_ep))
if save_model:
    network.save(file_name, "./pytorch_models")
np.save(f"./results/{file_name}", evaluations)
env.close()