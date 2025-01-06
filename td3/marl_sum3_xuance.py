import numpy as np
from gym.spaces import Box
from xuance.environment import RawMultiAgentEnv
from marl_sum3_env import GazeboEnv  # 替换为你的GazeboEnv模块路径

class MyNewMultiAgentEnv(RawMultiAgentEnv):
    def __init__(self, env_config):
        super(MyNewMultiAgentEnv, self).__init__()
        self.env_id = env_config.env_id
        self.num_agents = len(env_config.car_names)
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        
        # 定义状态空间、观测空间和动作空间
        self.state_space = Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        self.observation_space = {agent: Box(-np.inf, np.inf, shape=(8,), dtype=np.float32) for agent in self.agents}
        self.action_space = {agent: Box(-1.0, 1.0, shape=(2,), dtype=np.float32) for agent in self.agents}
        
        self.max_episode_steps = env_config.max_episode_steps
        self._current_step = 0
        
        # 初始化GazeboEnv
        self.gazebo_env = GazeboEnv(
            launchfile=env_config.launchfile,
            environment_dim=env_config.environment_dim,
            car_names=env_config.car_names,
            car_positions=env_config.car_positions,
            car_orientations=env_config.car_orientations
        )
    
    def reset(self):
        states = self.gazebo_env.reset()
        observations = {}
        for agent_id, state in states.items():
            observations[agent_id] = state
        self._current_step = 0
        return observations, {}
    
    def step(self, action_dict):
        states, rewards, dones, target_reached = self.gazebo_env.step(action_dict)
        
        observations = {}
        rewards_dict = {}
        dones_dict = {}
        infos_dict = {}
        
        for agent_id, state in states.items():
            observations[agent_id] = state
            rewards_dict[agent_id] = rewards[agent_id]
            dones_dict[agent_id] = dones[agent_id]
            infos_dict[agent_id] = {"target_reached": target_reached[agent_id]}
        
        self._current_step += 1
        dones_dict["__all__"] = self._current_step >= self.max_episode_steps or all(dones.values())
        
        return observations, rewards_dict, dones_dict, infos_dict
    
    def render(self, mode='human'):
        return self.gazebo_env.render()
    
    def close(self):
        self.gazebo_env.close()
