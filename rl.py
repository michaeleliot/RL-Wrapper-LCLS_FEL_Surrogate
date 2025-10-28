from const import QUAD_NAMES, RANGES, DEFAULTS
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from env import LTUHEnv
from lume_model.models import TorchModel

model = TorchModel("model_config.yaml")

env = LTUHEnv(QUAD_NAMES, RANGES, DEFAULTS, model)
eval_env = LTUHEnv(QUAD_NAMES, RANGES, DEFAULTS, model)

eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=5000,
                             deterministic=True, render=False)

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

ppo_model.learn(total_timesteps=100_000, callback=eval_callback)

ppo_model.save("ppo_ltuh_quad_optimizer")

obs, _ = env.reset()
done = False
while not done:
    action, _ = ppo_model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()