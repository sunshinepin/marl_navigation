#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import MultiAgentReplayBuffer
from xuance_marl_sum3_env import MyNewMultiAgentEnv
from types import SimpleNamespace

# 评估函数（多智能体版本）Critic (batch_size, 74) (global_state + action)
def evaluate(network, env, epoch, eval_episodes=10):
    avg_rewards = {agent: 0.0 for agent in env.agents}
    collisions = {agent: 0 for agent in env.agents}
    for _ in range(eval_episodes):
        obs, _ = env.reset()
        dones = {agent: False for agent in env.agents}
        step_count = 0
        while not all(dones.values()) and step_count < env.max_episode_steps:
            actions = {agent: [(raw_action[0] + 1) / 2, raw_action[1]] 
                       for agent, raw_action in 
                       {agent: network.get_action(np.array(obs[agent]), agent) 
                        for agent in env.agents if env.alive[env.agents.index(agent)]}.items()}
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

# Actor网络（每个智能体独立）
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(obs_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, obs):
        obs = F.relu(self.layer_1(obs))
        obs = F.relu(self.layer_2(obs))
        return self.tanh(self.layer_3(obs))

# Critic网络（输入全局状态和单个智能体动作）
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):  # state_dim 为全局状态维度，action_dim 为单个动作维度
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 800)  # 72 + 2 = 74
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, 1)
        self.layer_4 = nn.Linear(state_dim + action_dim, 800)
        self.layer_5 = nn.Linear(800, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        s1 = F.relu(self.layer_1(sa))
        s1 = F.relu(self.layer_2(s1))
        q1 = self.layer_3(s1)
        s2 = F.relu(self.layer_4(sa))
        s2 = F.relu(self.layer_5(s2))
        q2 = self.layer_6(s2)
        return q1, q2

# MATD3类
class MATD3(object):
    def __init__(self, env_info, max_action):
        self.agents = env_info['agents']
        self.num_agents = env_info['num_agents']
        obs_dim = env_info['state_space'][self.agents[0]].shape[0]  # 24
        action_dim = env_info['action_space'][self.agents[0]].shape[0]  # 2
        state_dim = obs_dim * self.num_agents  # 72

        self.actors = {agent: Actor(obs_dim, action_dim).to(device) for agent in self.agents}
        self.actor_targets = {agent: Actor(obs_dim, action_dim).to(device) for agent in self.agents}
        self.actor_optimizers = {agent: torch.optim.Adam(self.actors[agent].parameters()) for agent in self.agents}
        
        self.critics = {agent: Critic(state_dim, action_dim).to(device) for agent in self.agents}  # 输入全局状态和单个动作
        self.critic_targets = {agent: Critic(state_dim, action_dim).to(device) for agent in self.agents}
        self.critic_optimizers = {agent: torch.optim.Adam(self.critics[agent].parameters()) for agent in self.agents}

        for agent in self.agents:
            self.actor_targets[agent].load_state_dict(self.actors[agent].state_dict())
            self.critic_targets[agent].load_state_dict(self.critics[agent].state_dict())

        self.max_action = max_action
        self.writer = SummaryWriter()
        self.iter_count = 0

    def get_action(self, obs, agent, noise_scale=0.0):
        obs = torch.Tensor(obs.reshape(1, -1)).to(device)
        action = self.actors[agent](obs).cpu().data.numpy().flatten()
        if noise_scale > 0:
            action += np.random.normal(0, noise_scale, size=action.shape)
            action = np.clip(action, -self.max_action, self.max_action)
        return action

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99999, tau=0.005, 
              policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        for it in range(iterations):
            batch = replay_buffer.sample_batch(batch_size)
            states = {agent: torch.Tensor(batch[f'obs_{agent}']).to(device) for agent in self.agents}
            actions = {agent: torch.Tensor(batch[f'actions_{agent}']).to(device) for agent in self.agents}
            rewards = {agent: torch.Tensor(batch[f'rewards_{agent}']).to(device) for agent in self.agents}
            dones = {agent: torch.Tensor(batch[f'dones_{agent}']).to(device) for agent in self.agents}
            next_states = {agent: torch.Tensor(batch[f'next_obs_{agent}']).to(device) for agent in self.agents}

            global_state = torch.cat([states[agent] for agent in self.agents], dim=1)
            next_global_state = torch.cat([next_states[agent] for agent in self.agents], dim=1)

            # 计算目标 Q 值和更新 Critic
            for agent in self.agents:
                next_action = self.actor_targets[agent](next_states[agent])
                noise = torch.Tensor(next_action).data.normal_(0, policy_noise).to(device)
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

                target_Q1, target_Q2 = self.critic_targets[agent](next_global_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = rewards[agent] + ((1 - dones[agent]) * discount * target_Q).detach()

                current_Q1, current_Q2 = self.critics[agent](global_state, actions[agent])
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

                self.critic_optimizers[agent].zero_grad()
                critic_loss.backward()
                self.critic_optimizers[agent].step()

                # 更新 Actor
                if it % policy_freq == 0:
                    actor_action = self.actors[agent](states[agent])
                    q1, _ = self.critics[agent](global_state, actor_action)
                    actor_loss = -q1.mean()

                    self.actor_optimizers[agent].zero_grad()
                    actor_loss.backward()
                    self.actor_optimizers[agent].step()

                    # 软更新目标网络
                    for param, target_param in zip(self.actors[agent].parameters(), self.actor_targets[agent].parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    for param, target_param in zip(self.critics[agent].parameters(), self.critic_targets[agent].parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                self.writer.add_scalar(f"Critic_Loss/{agent}", critic_loss.item(), self.iter_count)

        self.iter_count += 1

    def save(self, filename, directory):
        for agent in self.agents:
            torch.save(self.actors[agent].state_dict(), f"{directory}/{filename}_{agent}_actor.pth")
            torch.save(self.critics[agent].state_dict(), f"{directory}/{filename}_{agent}_critic.pth")

    def load(self, filename, directory):
        for agent in self.agents:
            self.actors[agent].load_state_dict(torch.load(f"{directory}/{filename}_{agent}_actor.pth"))
            self.critics[agent].load_state_dict(torch.load(f"{directory}/{filename}_{agent}_critic.pth"))

# 主训练循环
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    eval_freq = 5e3
    max_timesteps = 5e6
    expl_noise = 1.0
    expl_decay_steps = 500000
    expl_min = 0.1
    batch_size = 40
    discount = 0.99999
    tau = 0.005
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2
    buffer_size = 1e6
    file_name = "MATD3_multi_car"
    save_model = True
    load_model = False

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if save_model and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    env_config_dict = {
        "env_id": "multi_car_env",
        "car_names": ["car1", "car2", "car3"],
        "car_positions": [[0.0, 5.0, 0.01], [0.0, -5.0, 0.01], [-5.0, 0.0, 0.01]],
        "car_orientations": [0, 0, 0],
        "max_episode_steps": 500
    }
    env_config = SimpleNamespace(**env_config_dict)
    print(f"Creating environment with config: {env_config.__dict__}")
    try:
        env = MyNewMultiAgentEnv(env_config)
        print(f"Environment initialized with {env.num_agents} agents")
    except Exception as e:
        print(f"Failed to initialize environment: {e}")
        exit(1)
    time.sleep(5)
    env_info = env.get_env_info()
    print(f"env_info: {env_info}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    max_action = 1.0

    network = MATD3(env_info, max_action)
    replay_buffer = MultiAgentReplayBuffer(buffer_size, env_info['num_agents'], seed)
    if load_model:
        try:
            network.load(file_name, "./pytorch_models")
        except:
            print("Could not load model, initializing with random parameters")

    evaluations = []
    timestep = 0
    timesteps_since_eval = 0
    episode_num = 0
    epoch = 1

    while timestep < max_timesteps:
        obs, _ = env.reset()
        dones = {agent: False for agent in env.agents}
        episode_rewards = {agent: 0.0 for agent in env.agents}
        episode_timesteps = 0

        while not all(dones.values()) and episode_timesteps < env.max_episode_steps:
            if expl_noise > expl_min:
                expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)
            
            actions = {}
            for agent in env.agents:
                if env.alive[env.agents.index(agent)]:
                    raw_action = network.get_action(obs[agent], agent, expl_noise)
                    raw_action = np.clip(raw_action, -max_action, max_action)
                    actions[agent] = [(raw_action[0] + 1) / 2, raw_action[1]]

            next_obs, rewards, dones, truncated, infos = env.step(actions)
            done_bool = {agent: 0 if episode_timesteps + 1 == env.max_episode_steps else int(dones[agent]) 
                         for agent in env.agents}

            obs_dict = {agent: obs[agent] for agent in env.agents}
            actions_dict = {agent: actions.get(agent, np.zeros(2)) for agent in env.agents}
            rewards_dict = {agent: rewards[agent] for agent in env.agents}
            dones_dict = {agent: done_bool[agent] for agent in env.agents}
            next_obs_dict = {agent: next_obs[agent] for agent in env.agents}

            replay_buffer.add(obs_dict, actions_dict, rewards_dict, dones_dict, next_obs_dict)

            obs = next_obs
            for agent in env.agents:
                episode_rewards[agent] += rewards[agent]
            episode_timesteps += 1
            timestep += 1
            timesteps_since_eval += 1

            if truncated or all(dones.values()):
                break

        network.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

        if timesteps_since_eval >= eval_freq:
            print("Validating")
            timesteps_since_eval %= eval_freq
            avg_rewards = evaluate(network, env, epoch)
            evaluations.append(avg_rewards)
            network.save(file_name, "./pytorch_models")
            np.save(f"./results/{file_name}", evaluations)
            for agent in env.agents:
                print(f"Episode {episode_num}, Agent {agent} Reward: {episode_rewards[agent]:.2f}")
                network.writer.add_scalar(f"Reward/{agent}", episode_rewards[agent], timestep)
            epoch += 1
            episode_num += 1

    evaluations.append(evaluate(network, env, epoch))
    if save_model:
        network.save(file_name, "./pytorch_models")
    np.save(f"./results/{file_name}", evaluations)
    env.close()