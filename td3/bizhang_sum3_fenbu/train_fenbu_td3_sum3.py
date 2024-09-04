import torch
import numpy as np
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from gazebo_env import GazeboEnv
# from replay_buffer import ReplayBuffer
import random
from collections import deque

import numpy as np
import ray

# 设置设备为 CPU
device = torch.device("cpu")
TIME_DELTA = 0.1
class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
        t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
# Actor 网络定义
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a

# Critic 网络定义
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

    def forward(self, s, a):
        s1 = F.relu(self.layer_1(s))
        s1 = F.relu(self.layer_2_s(s1) + self.layer_2_a(a))
        q1 = self.layer_3(s1)
        return q1

# 参数服务器类（直接作为普通类）
class ParameterServer:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = Adam(self.critic.parameters())

        self.tau = 0.005  # 软更新的系数

    def get_params(self):
        return self.actor.state_dict(), self.critic.state_dict()

    def update_params(self, actor_grad, critic_grad):
        try:
            # 更新Actor参数
            self.actor_optimizer.zero_grad()
            for param, grad in zip(self.actor.parameters(), actor_grad):
                param.grad = grad
            self.actor_optimizer.step()

            # 更新Critic参数
            self.critic_optimizer.zero_grad()
            for param, grad in zip(self.critic.parameters(), critic_grad):
                param.grad = grad
            self.critic_optimizer.step()

            # 软更新目标网络
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        except Exception as e:
            print(f"Exception in updating params: {e}")

    def set_params(self, actor_params, critic_params):
        self.actor.load_state_dict(actor_params)
        self.critic.load_state_dict(critic_params)

    def save_model(self, actor_path="actor.pth", critic_path="critic.pth"):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"Models saved to {actor_path} and {critic_path}")

    def load_model(self, actor_path="actor.pth", critic_path="critic.pth"):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        print(f"Models loaded from {actor_path} and {critic_path}")


# Worker 类（不再使用多线程）
class Worker:
    def __init__(self, replay_buffer, param_server, state_dim, action_dim, max_action, environment_dim, car_name, car_position, car_orientation, gamma=0.99, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.env = GazeboEnv("td3.launch", environment_dim, [car_name], [car_position], [car_orientation])
        self.replay_buffer = replay_buffer
        self.param_server = param_server
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        # 本地Actor和Critic
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = Adam(self.critic.parameters())

        # 计数器
        self.total_it = 0

    def run(self, max_episodes, batch_size):
        for episode in range(max_episodes):
            try:
                state_dict = self.env.reset()
                state = np.array(list(state_dict.values()), dtype=np.float32)
                done = False
                while not done:
                    state_tensor = torch.Tensor(state).to(device)
                    action = self.actor(state_tensor).cpu().data.numpy()

                    print(f"Episode {episode}, Step {self.total_it}: Action taken: {action}")

                    # 执行动作并等待反馈
                    next_state_list, reward, done, _ = self.env.step(action)
                    next_state = np.array(next_state_list, dtype=np.float32)
                    print(f"Next state: {next_state}, Reward: {reward}, Done: {done}")
                    self.replay_buffer.add(state, action, reward, done, next_state)
                    state = next_state

                    if self.replay_buffer.size() > batch_size:
                        self.update_model(batch_size)

                # 定期从参数服务器同步模型参数
                actor_params, critic_params = self.param_server.get_params()
                self.actor.load_state_dict(actor_params)
                self.critic.load_state_dict(critic_params)

            except Exception as e:
                print(f"Exception in episode {episode}: {e}")

    def update_model(self, batch_size):
        self.total_it += 1
        try:
            states, actions, rewards, dones, next_states = self.replay_buffer.sample_batch(batch_size)

            state = torch.Tensor(np.copy(states)).to(device)
            action = torch.Tensor(np.copy(actions)).to(device)
            reward = torch.Tensor(np.copy(rewards)).to(device)
            done = torch.Tensor(np.copy(dones)).to(device)
            next_state = torch.Tensor(np.copy(next_states)).to(device)

            # Select action according to policy and add clipped noise
            noise = torch.Tensor(np.copy(actions)).data.normal_(0, self.policy_noise).to(device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            critic_grads = torch.autograd.grad(critic_loss, self.critic.parameters(), retain_graph=True)

            # Delayed policy updates
            if self.total_it % self.policy_freq == 0:
                # Compute actor loss
                actor_loss = -self.critic(state, self.actor(state)).mean()
                actor_grads = torch.autograd.grad(actor_loss, self.actor.parameters(), retain_graph=True)

                # 传递 Critic 和 Actor 的梯度给参数服务器
                self.param_server.update_params(actor_grads, critic_grads)

            # 传递 Critic 的梯度给参数服务器（如果 policy update 没有触发）
            else:
                self.param_server.update_params([], critic_grads)

        except Exception as e:
            print(f"Exception in update_model: {e}")

# 主程序启动
if __name__ == "__main__":
    print("Initializing single-threaded training")

    # 定义环境参数
    environment_dim = 20  # 假设环境的状态维度
    robot_dim = 4  # 机器人状态维度
    state_dim = environment_dim + robot_dim  # 所以 state_dim 是 24
    action_dim = 2  # 假设动作维度
    max_action = 1.0  # 动作的最大值，假设为1.0
    car_name = "car1"
    car_position = [0.0, 5.0, 0.01]
    car_orientation = -1.57

    # 初始化全局模型和经验回放池
    param_server = ParameterServer(state_dim, action_dim)
    replay_buffer = ReplayBuffer(int(1e6))

    # 初始化单个Worker
    worker = Worker(replay_buffer, param_server, state_dim, action_dim, max_action, environment_dim, car_name, car_position, car_orientation)

    # 启动训练
    max_episodes = 1000
    batch_size = 100
    worker.run(max_episodes, batch_size)

    print("Training finished")
