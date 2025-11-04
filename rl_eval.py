import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
from stable_baselines3 import PPO
from rl_env import LTUHEnv
from const import QUAD_NAMES, buffered_env_ranges, DEFAULTS
from lume_model.models import TorchModel

model = TorchModel("model_config.yaml")
model.input_validation_config = {}
for name in model.input_names:
    model.input_validation_config[name] = "none"

model.output_validation_config = {}
for name in model.output_names:
    model.output_validation_config[name] = "none"


def evaluate_ltuh_model(env, model, n_episodes=30, target=2.25, tol=0.1, render=False):
    results = []

    for ep in trange(n_episodes, desc="Evaluating PPO"):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        beam_trace = []
        action_trace = []
        state_trace = []
        constraint_violations = 0
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            beam_trace.append(info["beam_intensity"])
            action_trace.append(action)
            state_trace.append(obs[:-1])

            if np.any(np.abs(obs[:-1]) > 1):
                constraint_violations += 1

            if render:
                env.render()
            step += 1

        final_beam = beam_trace[-1]
        results.append({
            "episode": ep,
            "total_reward": total_reward,
            "final_beam_intensity": final_beam,
            "intensity_error": abs(final_beam - target),
            "success": abs(final_beam - target) < tol,
            "mean_abs_action": np.mean(np.abs(action_trace)),
            "episode_length": step,
            "constraint_violations": constraint_violations / step,
            "beam_trace": beam_trace,
        })

    df = pd.DataFrame(results)
    return df

env = LTUHEnv(QUAD_NAMES, buffered_env_ranges, DEFAULTS, model)
ppo_model = PPO.load("ppo_ltuh_quad_optimizer", env=env)
df = evaluate_ltuh_model(env, ppo_model, n_episodes=50)
print(df.describe())
print("Success rate:", df['success'].mean())
