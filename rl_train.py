from const import QUAD_NAMES, buffered_env_ranges, DEFAULTS
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from rl_env import LTUHEnv
from lume_model.models import TorchModel

model = TorchModel("model_config.yaml")

env = LTUHEnv(QUAD_NAMES, buffered_env_ranges, DEFAULTS, model)

ppo_model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    clip_range=0.2,
)

ppo_model.learn(total_timesteps=100_000)

ppo_model.save("ppo_ltuh_quad_optimizer")