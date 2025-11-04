import sys
import os
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

model_to_load = get_model_load_name()
load_path = os.path.join(MODEL_SAVE_FOLDER, f"{model_to_load}.zip")

model = TorchModel("model_config.yaml")

eval_env = LTUHEnv(QUAD_NAMES, buffered_env_ranges, DEFAULTS, model)

print(f"\nLoading model from: {load_path}")
try:
    ppo_model = PPO.load(load_path, env=eval_env)
except FileNotFoundError:
    print(f"❌ Error: Model file not found at {load_path}")
    sys.exit(1)

obs, info = eval_env.reset()

print("\n--- Starting Verification Run ---")
print(f"Target Power: {eval_env.target_power}")
print("-" * 35)

done = False
while not done:
    action, _ = ppo_model.predict(obs, deterministic=True)
    
    obs, reward, terminated, truncated, info = eval_env.step(action)
    
    done = terminated or truncated
    
    eval_env.render()

eval_env.close()
print("\n--- Verification Run Complete ---")