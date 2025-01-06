# env_registration.py

from xuance.environment import REGISTRY_MULTI_AGENT_ENV
from marl_sum3_xuance import MyNewMultiAgentEnv  # 替换为你的自定义环境类模块路径

def register_env():
    REGISTRY_MULTI_AGENT_ENV["MyNewMultiAgentEnv"] = MyNewMultiAgentEnv
