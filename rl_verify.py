from const import QUAD_NAMES, buffered_env_ranges, DEFAULTS
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from rl_env import LTUHEnv
from lume_model.models import TorchModel

model = TorchModel("model_config.yaml")

# Load your environment (replace with your actual environment)
eval_env = LTUHEnv(QUAD_NAMES, buffered_env_ranges, DEFAULTS, model)

eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=5000,
                             deterministic=True, render=False)

# Load the trained PPO model
ppo_model = PPO.load("ppo_ltuh_quad_optimizer", env=eval_env)

# Reset environment and get initial observation
obs, info = eval_env.reset()

done = False
while not done:
    # Predict action using the trained policy
    action, _ = ppo_model.predict(obs, deterministic=True)
    
    # Step through environment
    obs, reward, terminated, truncated, info = eval_env.step(action)
    
    # Check if the episode is over
    done = terminated or truncated
    
    # Optionally render (may not work on all environments)
    eval_env.render()

eval_env.close()
