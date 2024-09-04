import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from gazebo_env import GazeboEnv

# 定义评估函数
def evaluate(env, network, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()
        done = False
        while not done and count < 501:
            action = network.get_action(np.array(state))
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done, _ = env.step(a_in)
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    print("..............................................")
    print(
        f"Average Reward over {eval_episodes} Evaluation Episodes, Epoch {epoch}: {avg_reward}, {avg_col}"
    )
    print("..............................................")
    return avg_reward


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


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)
        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        s1 = F.relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)
        s2 = F.relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2


class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.max_action = max_action
        self.writer = SummaryWriter()
        self.iter_count = 0

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=1, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        for it in range(iterations):
            batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)
            next_action = self.actor_target(next_state)
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
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
        self.writer.add_scalar("Average Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth"))

# 设置参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0
eval_freq = 5e3
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
file_name1 = "TD3_velodyne_car1"
file_name2 = "TD3_velodyne_car2"
save_model = True
load_model = True
random_near_obstacle = True

# 创建文件夹
if not os.path.exists("../results"):
    os.makedirs("../results")
if save_model and not os.path.exists("../pytorch_models"):
    os.makedirs("../pytorch_models")

# 创建环境
env = GazeboEnv("td3.launch", 20, ["car1", "car2"], [[-5, 5, 0.01], [-5, -5, 0.01]])
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = 24
action_dim = 2
max_action = 1

# 创建网络
network1 = TD3(state_dim, action_dim, max_action)
network2 = TD3(state_dim, action_dim, max_action)
# 创建经验回放缓冲区
replay_buffer1 = ReplayBuffer(buffer_size, seed)
replay_buffer2 = ReplayBuffer(buffer_size, seed)
if load_model:
    try:
        network1.load(file_name1, "./pytorch_models")
        network2.load(file_name2, "./pytorch_models")
    except:
        print("Could not load the stored model parameters, initializing training with random parameters")

# 创建评估数据存储
evaluations1 = []
evaluations2 = []

timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True
epoch = 1

# 开始训练循环
while timestep < max_timesteps:

    # 在回合结束时
    if done:
        if timestep != 0:
            network1.train(
                replay_buffer1,
                episode_timesteps,
                batch_size,
                discount,
                tau,
                policy_noise,
                noise_clip,
                policy_freq,
            )
            network2.train(
                replay_buffer2,
                episode_timesteps,
                batch_size,
                discount,
                tau,
                policy_noise,
                noise_clip,
                policy_freq,
            )

        if timesteps_since_eval >= eval_freq:
            print("Validating")
            timesteps_since_eval %= eval_freq
            evaluations1.append(
                evaluate(env, network1, epoch, eval_episodes=eval_ep)
            )
            evaluations2.append(
                evaluate(env, network2, epoch, eval_episodes=eval_ep)
            )
            network1.save(file_name1, directory="./pytorch_models")
            network2.save(file_name2, directory="./pytorch_models")
            np.save("./results/%s" % (file_name1), evaluations1)
            np.save("./results/%s" % (file_name2), evaluations2)
            epoch += 1

        states = env.reset()
        done = False

        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # 添加一些探索噪声
    if expl_noise > expl_min:
        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

    actions = []
    action1 = network1.get_action(np.array(states[0]))
    action1 = (action1 + np.random.normal(0, expl_noise, size=action_dim)).clip(
        -max_action, max_action
    )
    actions.append(action1)

    action2 = network2.get_action(np.array(states[1]))
    action2 = (action2 + np.random.normal(0, expl_noise, size=action_dim)).clip(
        -max_action, max_action
    )
    actions.append(action2)

    # 执行动作并获取新的状态和奖励
    next_states, rewards, dones, target_reached = env.step(actions)

    done_bool = 0 if episode_timesteps + 1 == max_ep else int(dones[0])
    done = 1 if episode_timesteps + 1 == max_ep else int(dones[0])
    episode_reward += rewards[0]

    replay_buffer1.add(states[0], actions[0], rewards[0], done_bool, next_states[0])
    replay_buffer2.add(states[1], actions[1], rewards[1], done_bool, next_states[1])

    states = next_states
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

# 完成训练后，进行最终评估并保存模型
evaluations1.append(evaluate(env, network1, epoch, eval_episodes=eval_ep))
evaluations2.append(evaluate(env, network2, epoch, eval_episodes=eval_ep))
if save_model:
    network1.save(file_name1, directory="./models")
    network2.save(file_name2, directory="./models")
np.save("./results/%s" % file_name1, evaluations1)
np.save("./results/%s" % file_name2, evaluations2)
