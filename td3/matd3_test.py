import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from squaternion import Quaternion
from xuance_marl_sum3_env_dead import MyNewMultiAgentEnv

class EnvConfig:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

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
        model_path = os.path.abspath(f"{directory}/{filename}")
        print(f"Loading model from: {model_path}")
        self.actor.load_state_dict(torch.load(model_path))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

max_ep = 1000
file_names = [
    "sanxin_TD3_velodyne_actor_car1.pth",
    "sanxin_TD3_velodyne_actor_car2.pth",
    "sanxin_TD3_velodyne_actor_car3.pth"
]

state_dim = 24
action_dim = 2
num_tests = 10  # 根据文件中测试用例数量调整
# save_dir = "/home/xzh/xzh/madrl-navigation/data_processing/matd3_test/world1"
save_dir = "/home/xzh/xzh/madrl-navigation/data_processing/matd3_test/world2"

os.makedirs(save_dir, exist_ok=True)

# 测试数据文件路径
# test_data_file = "/home/xzh/xzh/madrl-navigation/data_processing/matd3_test/world1/start_goal.txt"
test_data_file = "/home/xzh/xzh/madrl-navigation/data_processing/matd3_test/world2/start_goal.txt"


def load_test_cases(file_path, num_agents=3):
    """
    从文件中加载测试用例。
    每行包含 num_agents * 5 个浮点数，表示每个智能体的 start_x, start_y, orientation, goal_x, goal_y。
    """
    test_cases = []
    with open(file_path, 'r') as f:
        for line in f:
            data = [float(x) for x in line.strip().split()]
            if len(data) != num_agents * 5:
                print(f"Warning: Skipping invalid line with {len(data)} values (expected {num_agents * 5})")
                continue
            test_case = {
                "car_positions": [],
                "car_orientations": [],
                "goal_positions": []
            }
            for i in range(num_agents):
                idx = i * 5
                test_case["car_positions"].append([data[idx], data[idx + 1], 0.1])  # z 固定为 0.1
                test_case["car_orientations"].append(data[idx + 2])
                test_case["goal_positions"].append([data[idx + 3], data[idx + 4]])
            test_cases.append(test_case)
    return test_cases

# 加载测试用例
test_cases = load_test_cases(test_data_file)
num_tests = min(num_tests, len(test_cases))  # 确保不超过文件中的测试用例数量
print(f"Loaded {len(test_cases)} test cases from {test_data_file}")

env_config_dict = {
    "env_id": "multi_agent_env",
    "car_names": ["car1", "car2", "car3"],
    "car_positions": None,  # 将在 test 中动态设置
    "car_orientations": None,  # 将在 test 中动态设置
    "goal_positions": None,  # 将在 test 中动态设置
    "max_episode_steps": max_ep
}

def get_robot_pose(env, agent_idx):
    return [env.odom_positions[agent_idx][0], 
            env.odom_positions[agent_idx][1], 
            env.start_orientations[agent_idx]]

def calculate_trajectory_length(path):
    length = sum(
        np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        for (x1, y1, _), (x2, y2, _) in zip(path[:-1], path[1:])
    )
    return length

def test(test_number):
    # 更新 env_config 的动态参数
    env_config = EnvConfig(env_config_dict)
    env_config.car_positions = test_cases[test_number - 1]["car_positions"]
    env_config.car_orientations = test_cases[test_number - 1]["car_orientations"]
    env_config.goal_positions = test_cases[test_number - 1]["goal_positions"]

    env = MyNewMultiAgentEnv(env_config)
    networks = [TD3(state_dim, action_dim) for _ in range(3)]
    
    for i, network in enumerate(networks):
        network.load(file_names[i], "/home/xzh/xzh/madrl-navigation/pytorch_models/matd3")
    
    obs, _ = env.reset()
    
    path_records = [[] for _ in range(3)]
    steps = 0
    collisions = [False] * 3
    goals_reached = [False] * 3
    start_time = time.time()

    while steps < max_ep:
        action_dict = {
            f"agent_{i}": network.get_action(obs[f"agent_{i}"])
            for i, network in enumerate(networks)
        }
        
        next_obs, rewards, dones, truncated, infos = env.step(action_dict)
        
        for i in range(3):
            agent = f"agent_{i}"
            path_records[i].append(get_robot_pose(env, i))
            if infos[agent]["reached_goal"] and not goals_reached[i]:
                goals_reached[i] = True
            if env.observe_collision(env.velodyne_data[i])[1]:
                collisions[i] = True
        
        all_done = all(not env.alive[i] for i in range(3))
        if all_done or all(goals_reached) or steps >= max_ep:
            break
        
        obs = next_obs
        steps += 1

    end_time = time.time()
    trajectory_lengths = [calculate_trajectory_length(path) for path in path_records]

    for i, path in enumerate(path_records):
        path_filename = f"{save_dir}/path_car{i+1}_{test_number}.txt"
        np.savetxt(
            path_filename, np.array(path), fmt='%.6f', delimiter=',',
            header='[x, y, theta]', comments=''
        )
        print(f"Car {i+1} path saved to {path_filename}")

    return [{
        'test_number': test_number,
        'car_id': i + 1,
        'success': goals_reached[i],
        'collision': collisions[i],
        'steps': steps,
        'navigation_time': end_time - start_time,
        'trajectory_length': trajectory_lengths[i]
    } for i in range(3)]

# 运行测试
test_results = []
for test_number in range(1, num_tests + 1):
    print(f"Running test {test_number}...")
    results = test(test_number)
    test_results.extend(results)

# 保存结果
summary_filename = f"{save_dir}/test_summary.txt"
with open(summary_filename, 'w') as f:
    f.write("TestNumber,CarID,Success,Collision,Steps,NavigationTime,TrajectoryLength\n")
    for r in test_results:
        f.write(
            f"{r['test_number']},{r['car_id']},{int(r['success'])},{int(r['collision'])},"
            f"{r['steps']},{r['navigation_time']:.2f},{r['trajectory_length']:.2f}\n"
        )

print(f"\nTest summary saved to {summary_filename}")
print(f"Completed {num_tests} tests.")