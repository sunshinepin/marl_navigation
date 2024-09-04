import math
import torch
import random
import numpy as np


class Functions:

    @staticmethod
    def obs_dict_to_array(obs_dict):
        agent_0_state = obs_dict["agent_0"]
        agent_1_state = obs_dict["agent_1"]
        agent_2_state = obs_dict["agent_2"]

        actor_states = np.vstack((agent_0_state, agent_1_state, agent_2_state))
        critic_state = np.concatenate((agent_0_state, agent_1_state, agent_2_state), axis=0)

        return critic_state, actor_states

    @staticmethod
    def action_array_to_dict(actions):
        action_env = {
            "agent_0": actions[0],
            "agent_1": actions[1],
            "agent_2": actions[2]
        }
        return action_env
