"""
Data structure for implementing experience replay for multi-agent systems
Modified from Patrick Emami's single-agent version
"""
import random
from collections import deque
import numpy as np


class MultiAgentReplayBuffer(object):
    def __init__(self, buffer_size, num_agents, random_seed=123):
        """
        The right side of the deque contains the most recent experiences.
        Each experience is a dictionary containing data for all agents.
        """
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, obs, actions, rewards, dones, next_obs):
        """
        Add a multi-agent experience to the buffer.
        """
        experience = {
            'obs': {agent: obs[agent].copy() for agent in obs},
            'actions': {agent: actions[agent].copy() for agent in actions},
            'rewards': rewards.copy(),
            'dones': dones.copy(),
            'next_obs': {agent: next_obs[agent].copy() for agent in next_obs}
        }
        
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        """
        Sample a batch of experiences and return separated data for each agent.
        """
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        batch_data = {}
        
        for agent_idx in range(self.num_agents):
            agent_id = f"agent_{agent_idx}"
            batch_data[f'obs_{agent_id}'] = np.array([exp['obs'][agent_id] for exp in batch])
            batch_data[f'actions_{agent_id}'] = np.array([exp['actions'][agent_id] for exp in batch])
            batch_data[f'rewards_{agent_id}'] = np.array([exp['rewards'][agent_id] for exp in batch]).reshape(-1, 1)
            batch_data[f'dones_{agent_id}'] = np.array([exp['dones'][agent_id] for exp in batch]).reshape(-1, 1)
            batch_data[f'next_obs_{agent_id}'] = np.array([exp['next_obs'][agent_id] for exp in batch])
        
        return batch_data

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.count = 0