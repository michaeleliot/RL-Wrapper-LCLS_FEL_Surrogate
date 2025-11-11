import sys
import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from lume_model.models import TorchModel

from const import QUAD_NAMES, buffered_env_ranges, DEFAULTS
from rl_env import LTUHEnv

MODEL_SAVE_FOLDER = "final_rl_models"
LOGS_FOLDER = "rl_training_logs"
TB_FOLDER = 'tb_logs'

os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)
os.makedirs(TB_FOLDER, exist_ok=True)


def get_model_save_name():
    """
    Prompts the user to select an existing model name to overwrite
    or enter a new name.
    """
    existing_models = [
        f.replace(".zip", "")
        for f in os.listdir(MODEL_SAVE_FOLDER)
        if f.endswith(".zip")
    ]
    
    print("\n--- Model Saving Options ---")
    if existing_models:
        print("Existing models found (select one to overwrite):")
        for i, name in enumerate(existing_models):
            print(f"  [{i+1}] {name}")
    
    print(f"  [N] Enter a new model name")
    
    choice = input("\nEnter selection (e.g., '1' or 'N'): ").strip().upper()
    
    if choice.isdigit():
        try:
            index = int(choice) - 1
            if 0 <= index < len(existing_models):
                selected_name = existing_models[index]
                print(f"✅ Selected existing model: **{selected_name}** (will be overwritten)")
                return selected_name
            else:
                print("❌ Invalid number selection.")
                return get_model_save_name()
        except ValueError:
            pass 
    
    if choice == 'N' or not choice:
        new_name = input("Enter a **new** unique name for the model (e.g., ppo_v2_new_range): ").strip()
        if not new_name:
            print("❌ Model name cannot be empty.")
            sys.exit(1)
            
        if new_name in existing_models:
            print(f"⚠️ Warning: '{new_name}' already exists. It will be overwritten.")
        
        return new_name
    
    print("❌ Invalid choice. Please try again.")
    return get_model_save_name()


model_name_base = get_model_save_name()
final_model_save_path = os.path.join(MODEL_SAVE_FOLDER, f"{model_name_base}.zip")

log_dir = os.path.join(LOGS_FOLDER, f"{model_name_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(log_dir, exist_ok=True)


model = TorchModel("model_config.yaml")
model.input_validation_config = {}
for name in model.input_names:
    model.input_validation_config[name] = "warn"

model.output_validation_config = {}
for name in model.output_names:
    model.output_validation_config[name] = "warn"

#env = LTUHEnv(QUAD_NAMES, buffered_env_ranges, DEFAULTS, model)
train_env = LTUHEnv(QUAD_NAMES, buffered_env_ranges, DEFAULTS, model)
train_env = Monitor(train_env, filename=os.path.join(log_dir, "monitor.csv"))
eval_env = LTUHEnv(QUAD_NAMES, buffered_env_ranges, DEFAULTS, model)
eval_env = Monitor(eval_env, filename=os.path.join(log_dir, "eval_monitor.csv"))

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=5000,
    n_eval_episodes=10,
    deterministic=True,
    render=False
)

print(f"Logging and saving best model (during training) to: {log_dir}")
print("-" * 40)

ppo_model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log=TB_FOLDER,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    clip_range=0.2,
)

print("Starting PPO training...")
ppo_model.learn(total_timesteps=100_000, callback=eval_callback, progress_bar=True)

ppo_model.save(final_model_save_path)

print("-" * 40)
print(f"✅ Training complete. Final model saved (overwritten/created) at: **{final_model_save_path}**")

print(f"Logs, best checkpoint, and evaluations saved to: {log_dir}")
print(f"TensorBoard logs: {TB_FOLDER}. run tensorboard --logdir {TB_FOLDER}")