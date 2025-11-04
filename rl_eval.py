import sys
import os
import numpy as np
import pandas as pd
from tqdm import trange
from stable_baselines3 import PPO
from rl_env import LTUHEnv
from const import QUAD_NAMES, buffered_env_ranges, DEFAULTS
from lume_model.models import TorchModel

MODEL_SAVE_FOLDER = "final_rl_models"

def get_model_load_name():
    """
    Prompts the user to select an existing model name to load.
    """
    existing_models = [
        f.replace(".zip", "")
        for f in os.listdir(MODEL_SAVE_FOLDER)
        if f.endswith(".zip")
    ]
    
    if not existing_models:
        print(f"❌ No models found in '{MODEL_SAVE_FOLDER}'. Exiting.")
        sys.exit(1)

    print("\n--- Model Loading Options ---")
    print("Existing models found (select one to load):")
    for i, name in enumerate(existing_models):
        print(f"  [{i+1}] {name}")
    
    while True:
        choice = input("\nEnter selection number (e.g., '1'): ").strip()
        if choice.isdigit():
            try:
                index = int(choice) - 1
                if 0 <= index < len(existing_models):
                    selected_name = existing_models[index]
                    print(f"✅ Selected model: **{selected_name}**")
                    return selected_name
                else:
                    print("❌ Invalid selection number. Try again.")
            except ValueError:
                pass 
        print("❌ Invalid input. Please enter a number corresponding to the list.")

model = TorchModel("model_config.yaml")
model.input_validation_config = {}
for name in model.input_names:
    model.input_validation_config[name] = "none"

model.output_validation_config = {}
for name in model.output_names:
    model.output_validation_config[name] = "none"


def evaluate_ltuh_model(env, model, n_episodes=30, target=2.25, tol=0.1, render=False):
    """
    Runs the trained PPO model for n_episodes and collects performance metrics.
    """
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

model_to_load = get_model_load_name()
load_path = os.path.join(MODEL_SAVE_FOLDER, f"{model_to_load}.zip")

env = LTUHEnv(QUAD_NAMES, buffered_env_ranges, DEFAULTS, model)

print(f"Loading model from: {load_path}")
try:
    ppo_model = PPO.load(load_path, env=env)
except FileNotFoundError:
    print(f"❌ Error: Model file not found at {load_path}")
    sys.exit(1)

df = evaluate_ltuh_model(env, ppo_model, n_episodes=50)

print("\n--- Evaluation Summary ---")
print(df.describe())
print("\nSuccess rate (within tolerance):", df['success'].mean())