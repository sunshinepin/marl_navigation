import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
from utils import Functions
from replay_buffer import ReplayBuffer
from velodyne_env_sum4 import GazeboEnv

# Evaluation function
def evaluate(networks, env, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        obs_dict = env.reset()
        critic_state, actor_states = Functions.obs_dict_to_array(obs_dict)
        dones = [False] * len(actor_states)
        while not all(dones):
            actions = {f"agent_{i}": network.get_action(np.array(actor_states[i])) for i, network in enumerate(networks)}
            actions_in = {agent: [(action[0] + 1) / 2, action[1]] for agent, action in actions.items()}
            next_obs_dict, rewards, dones, targets = env.step(actions_in)
            avg_reward += sum(rewards) / len(rewards)
            if any(reward < -90 for reward in rewards):
                col += 1
            critic_state, actor_states = Functions.obs_dict_to_array(next_obs_dict)
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
expl_decay_steps = 100000
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
load_model = True
random_near_obstacle = True

# Create the network storage folders
if not os.path.exists("./results"):
    os.makedirs("./results")
if save_model and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

# Create the training environment and network for car1, car2, car3, and car4
environment_dim = 20
robot_dim = 4
car_names = ["car1", "car2", "car3", "car4"]
car_positions = [[0.0, 5.0, 0.01], [0.0, -5.0, 0.01], [-5.0, 0.0, 0.01], [5.0, 0.0, 0.01]]
env = GazeboEnv("td3.launch", environment_dim, car_names, car_positions)

# Create the training network and replay buffers
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2
max_action = 1

network1 = TD3(state_dim, action_dim, max_action, name="car1")
network2 = TD3(state_dim, action_dim, max_action, name="car2")
network3 = TD3(state_dim, action_dim, max_action, name="car3")
network4 = TD3(state_dim, action_dim, max_action, name="car4")
replay_buffer1 = ReplayBuffer(buffer_size, seed)
replay_buffer2 = ReplayBuffer(buffer_size, seed)
replay_buffer3 = ReplayBuffer(buffer_size, seed)
replay_buffer4 = ReplayBuffer(buffer_size, seed)
if load_model:
    try:
        network1.load("TD3_velodyne", "./pytorch_models")
        network2.load("TD3_velodyne", "./pytorch_models")
        network3.load("TD3_velodyne", "./pytorch_models")
        network4.load("TD3_velodyne", "./pytorch_models")
    except Exception as e:
        print("Could not load the stored model parameters for car1, car2, car3, and car4, initializing training with random parameters")
        print("Error:", e)

# Create evaluation data store
evaluations1 = []
evaluations2 = []
evaluations3 = []
evaluations4 = []

# Open a log file to record rewards and losses
log_file = open("training_log.txt", "w")

timestep = 0
timesteps_since_eval = 0
episode_num1 = 0
episode_num2 = 0
episode_num3 = 0
episode_num4 = 0
done1 = True
done2 = True
done3 = True
done4 = True
epoch = 1

count_rand_actions1 = 0
count_rand_actions2 = 0
count_rand_actions3 = 0
count_rand_actions4 = 0
random_action1 = []
random_action2 = []
random_action3 = []
random_action4 = []

# Training loop
while timestep < max_timesteps:

    # Training loop for car1, car2, car3, and car4
    if (done1 and done2 and done3 and done4) or (episode_timesteps1 >= max_ep and episode_timesteps2 >= max_ep and episode_timesteps3 >= max_ep and episode_timesteps4 >= max_ep):
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
            network3.train(
                replay_buffer3,
                episode_timesteps3,
                batch_size,
                discount,
                tau,
                policy_noise,
                noise_clip,
                policy_freq,
            )
            network4.train(
                replay_buffer4,
                episode_timesteps4,
                batch_size,
                discount,
                tau,
                policy_noise,
                noise_clip,
                policy_freq,
            )

            # Record training losses and rewards
            log_file.write(f"Epoch {epoch}, timestep {timestep}\n")
            log_file.write(f"Car1 Episode {episode_num1}: Reward {episode_reward1}\n")
            log_file.write(f"Car2 Episode {episode_num2}: Reward {episode_reward2}\n")
            log_file.write(f"Car3 Episode {episode_num3}: Reward {episode_reward3}\n")
            log_file.write(f"Car4 Episode {episode_num4}: Reward {episode_reward4}\n")
            log_file.flush()

        if timesteps_since_eval >= eval_freq:
            print("Validating car1, car2, car3, and car4")
            timesteps_since_eval %= eval_freq
            evaluations1.append(
                evaluate(networks=[network1, network2, network3, network4], env=env, epoch=epoch, eval_episodes=eval_ep)
            )
            network1.save(file_name, directory="./pytorch_models")
            network2.save(file_name, directory="./pytorch_models")
            network3.save(file_name, directory="./pytorch_models")
            network4.save(file_name, directory="./pytorch_models")
            np.save(f"./results/{file_name}_car1", evaluations1)
            np.save(f"./results/{file_name}_car2", evaluations2)
            np.save(f"./results/{file_name}_car3", evaluations3)
            np.save(f"./results/{file_name}_car4", evaluations4)
            epoch += 1

        states = env.reset()
        done1, done2, done3, done4 = False, False, False, False

        episode_reward1 = 0
        episode_reward2 = 0
        episode_reward3 = 0
        episode_reward4 = 0
        episode_timesteps1 = 0
        episode_timesteps2 = 0
        episode_timesteps3 = 0
        episode_timesteps4 = 0
        episode_num1 += 1
        episode_num2 += 1
        episode_num3 += 1
        episode_num4 += 1

    if expl_noise > expl_min:
        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

    actions = {f"agent_{i}": (network.get_action(np.array(state)) + np.random.normal(0, expl_noise, size=action_dim)).clip(-max_action, max_action) for i, (network, state) in enumerate(zip([network1, network2, network3, network4], states.values()))}

    if random_near_obstacle:
        for i, state in enumerate(states.values()):
            if np.random.uniform(0, 1) > 0.85 and min(state[4:-8]) < 0.6 and eval(f'count_rand_actions{i+1}') < 1:
                exec(f'count_rand_actions{i+1} = np.random.randint(8, 15)')
                exec(f'random_action{i+1} = np.random.uniform(-1, 1, 2)')

        for i in range(4):
            if eval(f'count_rand_actions{i+1} > 0'):
                exec(f'count_rand_actions{i+1} -= 1')
                exec(f'actions["agent_{i}"] = random_action{i+1}')
                actions[f"agent_{i}"][0] = -1

    actions_in = {agent: [(action[0] + 1) / 2, action[1]] for agent, action in actions.items()}
    next_states, rewards, dones, targets = env.step(actions_in)

    if all(targets):
        done1, done2, done3, done4 = True, True, True, True

    done_bools = [0 if episode_timesteps + 1 == max_ep else int(done) for episode_timesteps, done in zip([episode_timesteps1, episode_timesteps2, episode_timesteps3, episode_timesteps4], [done1, done2, done3, done4])]

    episode_reward1 += rewards[0]
    episode_reward2 += rewards[1]
    episode_reward3 += rewards[2]
    episode_reward4 += rewards[3]

    replay_buffer1.add(states["agent_0"], actions["agent_0"], rewards[0], done_bools[0], next_states["agent_0"])
    replay_buffer2.add(states["agent_1"], actions["agent_1"], rewards[1], done_bools[1], next_states["agent_1"])
    replay_buffer3.add(states["agent_2"], actions["agent_2"], rewards[2], done_bools[2], next_states["agent_2"])
    replay_buffer4.add(states["agent_3"], actions["agent_3"], rewards[3], done_bools[3], next_states["agent_3"])

    states = next_states
    episode_timesteps1 += 1
    episode_timesteps2 += 1
    episode_timesteps3 += 1
    episode_timesteps4 += 1

    timestep += 1
    timesteps_since_eval += 1

# After the training is done, evaluate the network and save it
evaluations1.append(evaluate(networks=[network1, network2, network3, network4], env=env, epoch=epoch, eval_episodes=eval_ep))

if save_model:
    network1.save(f"{file_name}_car1", directory="./models")
    network2.save(f"{file_name}_car2", directory="./models")
    network3.save(f"{file_name}_car3", directory="./models")
    network4.save(f"{file_name}_car4", directory="./models")

np.save(f"./results/{file_name}_car1", evaluations1)
np.save(f"./results/{file_name}_car2", evaluations2)
np.save(f"./results/{file_name}_car3", evaluations3)
np.save(f"./results/{file_name}_car4", evaluations4)

# Close the log file
log_file.close()
