from const import QUAD_NAMES, buffered_env_ranges, DEFAULTS
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from rl_env import LTUHEnv
from lume_model.models import TorchModel

model = TorchModel("model_config.yaml")

model.input_validation_config = {}
for name in model.input_names:
    model.input_validation_config[name] = "warn"

model.output_validation_config = {}
for name in model.output_names:
    model.output_validation_config[name] = "warn"

env = LTUHEnv(QUAD_NAMES, buffered_env_ranges, DEFAULTS, model)

eval_callback = EvalCallback(env, best_model_save_path="./logs/",
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

ppo_model.learn(total_timesteps=100_000, callback=eval_callback, progress_bar=True)

ppo_model.save("ppo_ltuh_quad_optimizer")