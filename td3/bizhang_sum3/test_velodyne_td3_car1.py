import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from velodyne_env_sum3_obs1 import GazeboEnv

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


class TD3(object):
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )


# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0
max_ep = 800
file_name = "TD3_velodyne"

# Custom start and target points
start_point = [-0.0, 9.0, 0.0, 0.0]  # (-8, 5)
target_point = [-9.0, -9.0, 0.0, 0.0]  # (8, -5)

# Create the testing environment
environment_dim = 20
robot_dim = 4
env = GazeboEnv("td3.launch", environment_dim, car_names, car_positions, car_orientations)
env.set_self_state.pose.position.x = start_point[0]
env.set_self_state.pose.position.y = start_point[1]
env.goal_x = target_point[0]
env.goal_y = target_point[1]
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2

# Create the network
network = TD3(state_dim, action_dim)
try:
    network.load(file_name, "./pytorch_models")
except:
    raise ValueError("Could not load the stored model parameters")

done = False
episode_timesteps = 0
state = env.reset()

# Begin the testing loop for one episode
while not done:
    action = network.get_action(np.array(state))

    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target = env.step(a_in)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)

    if done:
        break
    else:
        state = next_state
        episode_timesteps += 1

print("Test run completed")
