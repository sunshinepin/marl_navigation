import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from velodyne_env_sum2_end import GazeboEnv

# Evaluation function
def evaluate(network, env1, env2, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        state1 = env1.reset()
        state2 = env2.reset()
        done1 = False
        done2 = False
        while not (done1 and done2):
            action1 = network.get_action(np.array(state1))
            action2 = network.get_action(np.array(state2))
            a_in1 = [(action1[0] + 1) / 2, action1[1]]
            a_in2 = [(action2[0] + 1) / 2, action2[1]]
            state1, reward1, done1, target1 = env1.step(a_in1)
            state2, reward2, done2, target2 = env2.step(a_in2)
            avg_reward += (reward1 + reward2) / 2
            if reward1 < -90 or reward2 < -90:
                col += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    print("..............................................")
    print(
        "Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f"
        % (eval_episodes, epoch, avg_reward, avg_col)
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

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, name=""):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.writer = SummaryWriter(comment=f"_{name}")
        self.iter_count = 0
        self.name = name

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=1,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        for it in range(iterations):
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
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

                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss += loss
        self.iter_count += 1
        self.writer.add_scalar(f"{self.name}_loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar(f"{self.name}_Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar(f"{self.name}_Max. Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_{self.name}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_{self.name}_critic.pth")

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load(f"{directory}/{filename}_{self.name}_actor.pth")
        )
        self.critic.load_state_dict(
            torch.load(f"{directory}/{filename}_{self.name}_critic.pth")
        )

# Set the parameters for the implementation
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
file_name = "TD3_velodyne"
save_model = True
load_model = False
random_near_obstacle = True

# Create the network storage folders
if not os.path.exists("./results"):
    os.makedirs("./results")
if save_model and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

# Create the training environment and network for car1
environment_dim = 20
robot_dim = 4
env1 = GazeboEnv("td3.launch", environment_dim, "car1", -5.0, 5.0, 0.01)
env2 = GazeboEnv("td3.launch", environment_dim, "car2", -5.0, -5.0, 0.01)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2
max_action = 1

network1 = TD3(state_dim, action_dim, max_action, name="car1")
network2 = TD3(state_dim, action_dim, max_action, name="car2")
replay_buffer1 = ReplayBuffer(buffer_size, seed)
replay_buffer2 = ReplayBuffer(buffer_size, seed)
if load_model:
    try:
        network1.load(file_name, "./pytorch_models")
        network2.load(file_name, "./pytorch_models")
    except:
        print("Could not load the stored model parameters for car1 and car2, initializing training with random parameters")

# Create evaluation data store
evaluations1 = []
evaluations2 = []

timestep = 0
timesteps_since_eval = 0
episode_num1 = 0
episode_num2 = 0
done1 = True
done2 = True
epoch = 1

count_rand_actions1 = 0
count_rand_actions2 = 0
random_action1 = []
random_action2 = []

# Begin the training loop
while timestep < max_timesteps:

    # Training loop for car1 and car2
    if done1 and done2:
        if timestep != 0:
            network1.train(
                replay_buffer1,
                episode_timesteps1,
                batch_size,
                discount,
                tau,
                policy_noise,
                noise_clip,
                policy_freq,
            )
            network2.train(
                replay_buffer2,
                episode_timesteps2,
                batch_size,
                discount,
                tau,
                policy_noise,
                noise_clip,
                policy_freq,
            )

        if timesteps_since_eval >= eval_freq:
            print("Validating car1 and car2")
            timesteps_since_eval %= eval_freq
            evaluations1.append(
                evaluate(network=network1, env1=env1, env2=env2, epoch=epoch, eval_episodes=eval_ep)
            )
            evaluations2.append(
                evaluate(network=network2, env1=env1, env2=env2, epoch=epoch, eval_episodes=eval_ep)
            )
            network1.save(file_name, directory="./pytorch_models")
            network2.save(file_name, directory="./pytorch_models")
            np.save(f"./results/{file_name}_car1", evaluations1)
            np.save(f"./results/{file_name}_car2", evaluations2)
            epoch += 1

        state1 = env1.reset()
        state2 = env2.reset()
        done1 = False
        done2 = False

        episode_reward1 = 0
        episode_reward2 = 0
        episode_timesteps1 = 0
        episode_timesteps2 = 0
        episode_num1 += 1
        episode_num2 += 1

    if expl_noise > expl_min:
        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

    action1 = network1.get_action(np.array(state1))
    action1 = (action1 + np.random.normal(0, expl_noise, size=action_dim)).clip(
        -max_action, max_action
    )

    action2 = network2.get_action(np.array(state2))
    action2 = (action2 + np.random.normal(0, expl_noise, size=action_dim)).clip(
        -max_action, max_action
    )

    if random_near_obstacle:
        if np.random.uniform(0, 1) > 0.85 and min(state1[4:-8]) < 0.6 and count_rand_actions1 < 1:
            count_rand_actions1 = np.random.randint(8, 15)
            random_action1 = np.random.uniform(-1, 1, 2)

        if count_rand_actions1 > 0:
            count_rand_actions1 -= 1
            action1 = random_action1
            action1[0] = -1

        if np.random.uniform(0, 1) > 0.85 and min(state2[4:-8]) < 0.6 and count_rand_actions2 < 1:
            count_rand_actions2 = np.random.randint(8, 15)
            random_action2 = np.random.uniform(-1, 1, 2)

        if count_rand_actions2 > 0:
            count_rand_actions2 -= 1
            action2 = random_action2
            action2[0] = -1

    a_in1 = [(action1[0] + 1) / 2, action1[1]]
    a_in2 = [(action2[0] + 1) / 2, action2[1]]
    next_state1, reward1, done1, target1 = env1.step(a_in1)
    next_state2, reward2, done2, target2 = env2.step(a_in2)

    if target1 and target2:
        done1 = True
        done2 = True

    done_bool1 = 0 if episode_timesteps1 + 1 == max_ep else int(done1)
    done_bool2 = 0 if episode_timesteps2 + 1 == max_ep else int(done2)

    episode_reward1 += reward1
    episode_reward2 += reward2

    replay_buffer1.add(state1, action1, reward1, done_bool1, next_state1)
    replay_buffer2.add(state2, action2, reward2, done_bool2, next_state2)

    state1 = next_state1
    state2 = next_state2
    episode_timesteps1 += 1
    episode_timesteps2 += 1

    timestep += 1
    timesteps_since_eval += 1

# After the training is done, evaluate the network and save it
evaluations1.append(evaluate(network=network1, env1=env1, env2=env2, epoch=epoch, eval_episodes=eval_ep))
evaluations2.append(evaluate(network=network2, env1=env1, env2=env2, epoch=epoch, eval_episodes=eval_ep))

if save_model:
    network1.save(f"{file_name}_car1", directory="./models")
    network2.save(f"{file_name}_car2", directory="./models")

np.save(f"./results/{file_name}_car1", evaluations1)
np.save(f"./results/{file_name}_car2", evaluations2)
