import sys
import os
import numpy as np
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from stable_baselines3 import PPO
from rl_env import LTUHEnv
from const import QUAD_NAMES, buffered_env_ranges, DEFAULTS
from lume_model.models import TorchModel

MODEL_SAVE_FOLDER = "final_rl_models"


def get_model_load_name():
    existing_models = [
        f.replace(".zip", "")
        for f in os.listdir(MODEL_SAVE_FOLDER)
        if f.endswith(".zip")
    ]
    if not existing_models:
        print(f"❌ No models found in '{MODEL_SAVE_FOLDER}'. Exiting.")
        sys.exit(1)

    print("\n--- Model Loading Options ---")
    for i, name in enumerate(existing_models):
        print(f"  [{i+1}] {name}")

    while True:
        choice = input("\nEnter selection number: ").strip()
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(existing_models):
                print(f"✅ Selected model: {existing_models[index]}")
                return existing_models[index]
        print("❌ Invalid selection. Try again.")


# --- Load surrogate TorchModel ---
model = TorchModel("model_config.yaml")
model.input_validation_config = {name: "none" for name in model.input_names}
model.output_validation_config = {name: "none" for name in model.output_names}


def evaluate_ltuh_model(env, model, n_episodes=30, target=2.25, tol=0.1, render=False):
    """
    Runs the trained PPO model and computes metrics:
    - Final beam intensity
    - Steps required
    - Mean and total change in observation space
    """
    results = []


    for ep in trange(n_episodes, desc="Evaluating PPO"):
        obs, info = env.reset()
        done = False
        beam_trace = []
        state_trace = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            beam_trace.append(info["beam_intensity"])
            state_trace.append(obs[:-1])  # ignore final bias term if exists
            obs = obs_next
            if render:
                env.render()

        # Compute observation change metrics
        state_trace = np.array(state_trace)
        if len(state_trace) > 1:
            diffs = np.abs(np.diff(state_trace, axis=0))
            mean_obs_change = np.mean(diffs)
            total_obs_change = np.sum(diffs)
        else:
            mean_obs_change = total_obs_change = 0.0

        final_beam = beam_trace[-1]
        results.append({
            "episode": ep,
            "final_beam_intensity": final_beam,
            "beam_error": abs(final_beam - target),
            "success": abs(final_beam - target) < tol,
            "mean_obs_change": mean_obs_change,
            "total_obs_change": total_obs_change,
            "beam_trace": beam_trace,
        })

    return pd.DataFrame(results)


def baseline_ltuh_model(env, n_episodes=30, target=2.25, tol=0.1, render=False):
    """
    Runs the trained PPO model and computes metrics:
    - Final beam intensity
    - Steps required
    - Mean and total change in observation space
    """
    results = []


    for ep in trange(n_episodes, desc="Getting Baseline"):
        obs, info = env.reset()
        done = False
        beam_trace = []
        state_trace = []
        
        while not done:
            obs_next, reward, terminated, truncated, info = env.step(obs[:-1])
            done = terminated or truncated

            beam_trace.append(info["beam_intensity"])
            state_trace.append(obs[:-1])  # ignore final bias term if exists
            obs = obs_next
            if render:
                env.render()

        # Compute observation change metrics
        state_trace = np.array(state_trace)
        if len(state_trace) > 1:
            diffs = np.abs(np.diff(state_trace, axis=0))
            mean_obs_change = np.mean(diffs)
            total_obs_change = np.sum(diffs)
        else:
            mean_obs_change = total_obs_change = 0.0

        final_beam = beam_trace[-1]
        results.append({
            "episode": ep,
            "final_beam_intensity": final_beam,
            "beam_error": abs(final_beam - target),
            "success": abs(final_beam - target) < tol,
            "mean_obs_change": mean_obs_change,
            "total_obs_change": total_obs_change,
            "beam_trace": beam_trace,
        })

    return pd.DataFrame(results)


# --- Load model and run evaluation ---
model_to_load = get_model_load_name()
load_path = os.path.join(MODEL_SAVE_FOLDER, f"{model_to_load}.zip")

env = LTUHEnv(QUAD_NAMES, buffered_env_ranges, DEFAULTS, model)

print(f"Loading model from: {load_path}")
ppo_model = PPO.load(load_path, env=env)

baseline_df = baseline_ltuh_model(env, n_episodes=50)
df = evaluate_ltuh_model(env, ppo_model, n_episodes=50)

print("\n--- Evaluation Summary ---")
print("Baseline", baseline_df[["final_beam_intensity", "mean_obs_change", "beam_error"]].describe())
print("Model", df[["final_beam_intensity", "mean_obs_change", "beam_error"]].describe())
print("\nSuccess rate:", df["success"].mean())

lower_bound = 2.0
upper_bound = 2.5


# def show_interactive_plots(df, target=2.25):
#     plt.style.use("ggplot")

#     def plot_final_intensity(ax):
#         ax.plot(df["episode"], df["final_beam_intensity"], "o-")
#         ax.axhspan(
#             ymin=lower_bound,
#             ymax=upper_bound,
#             color='green',
#             alpha=0.15,
#             label="Target Range"
#         )
#         ax.axhline(lower_bound, color="darkgreen", linestyle=":", linewidth=1)
#         ax.axhline(upper_bound, color="darkgreen", linestyle=":", linewidth=1)
#         ax.set_xlabel("Episode")
#         ax.set_ylabel("Final Beam Intensity")
#         ax.set_title("Final Beam Intensity per Episode")
#         ax.legend()


#     def plot_obs_change(ax):
#         ax.plot(df["episode"], df["mean_obs_change"], "o-", color="orange", label="Mean ΔObs")
#         ax.set_xlabel("Episode")
#         ax.set_ylabel("Mean Observation Change")
#         ax.set_title("Mean Observation Change Magnitude per Episode")
#         ax.legend()

#     def plot_total_obs_change(ax):
#         ax.plot(df["episode"], df["total_obs_change"], "o-", color="darkred")
#         ax.set_xlabel("Episode")
#         ax.set_ylabel("Total Observation Change")
#         ax.set_title("Total Observation Change per Episode")

#     def plot_beam_traces(ax):
#         for trace in df["beam_trace"]:
#             ax.plot(trace, alpha=0.3)
#         ax.axhspan(
#             ymin=lower_bound,
#             ymax=upper_bound,
#             color='green',
#             alpha=0.15,
#             label="Target Range"
#         )
#         ax.axhline(lower_bound, color="darkgreen", linestyle=":", linewidth=1)
#         ax.axhline(upper_bound, color="darkgreen", linestyle=":", linewidth=1)
#         ax.set_xlabel("Step")
#         ax.set_ylabel("Beam Intensity")
#         ax.set_title("Beam Intensity Traces")
#         ax.legend()


#     plots = [plot_beam_traces, plot_final_intensity, plot_obs_change, plot_total_obs_change, ]
#     titles = [
#         "Beam Intensity Over Time", "Final Beam Intensity",
#         "Mean Observation Change", "Total Observation Change", 
#     ]

#     fig = plt.figure(figsize=(8, 6))
#     gs = plt.GridSpec(2, 1, height_ratios=[20, 1])
#     ax = fig.add_subplot(gs[0])
#     index = {"i": 0}

#     def draw_plot():
#         ax.clear()
#         plots[index["i"]](ax)
#         fig.suptitle(f"{titles[index['i']]} ({index['i']+1}/{len(plots)})", fontsize=12)
#         fig.canvas.draw_idle()

#     btn_prev_ax = fig.add_axes([0.35, 0.02, 0.1, 0.05])
#     btn_next_ax = fig.add_axes([0.55, 0.02, 0.1, 0.05])
#     btn_prev = Button(btn_prev_ax, "◀ Previous")
#     btn_next = Button(btn_next_ax, "Next ▶")

#     def next_plot(event):
#         index["i"] = (index["i"] + 1) % len(plots)
#         draw_plot()

#     def prev_plot(event):
#         index["i"] = (index["i"] - 1) % len(plots)
#         draw_plot()

#     btn_prev.on_clicked(prev_plot)
#     btn_next.on_clicked(next_plot)

#     draw_plot()
#     plt.show()

def show_interactive_plots(df, baseline_df=None):
    plt.style.use("ggplot")

    def plot_final_intensity(ax):
        ax.plot(df["episode"], df["final_beam_intensity"], "o-", label="PPO Agent")
        if baseline_df is not None:
            ax.plot(baseline_df["episode"], baseline_df["final_beam_intensity"], "o--", label="Baseline", alpha=0.7)
        
        ax.axhspan(
            ymin=lower_bound,
            ymax=upper_bound,
            color='green',
            alpha=0.15,
            label="Target Range"
        )
        ax.axhline(lower_bound, color="darkgreen", linestyle=":", linewidth=1)
        ax.axhline(upper_bound, color="darkgreen", linestyle=":", linewidth=1)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Final Beam Intensity")
        ax.set_title("Final Beam Intensity per Episode")
        ax.legend()

    def plot_obs_change(ax):
        ax.plot(df["episode"], df["mean_obs_change"], "o-", color="orange", label="PPO Mean ΔObs")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean Observation Change")
        ax.set_title("Mean Observation Change Magnitude per Episode")
        ax.legend()

    def plot_total_obs_change(ax):
        ax.plot(df["episode"], df["total_obs_change"], "o-", color="darkred", label="PPO Total ΔObs")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Observation Change")
        ax.set_title("Total Observation Change per Episode")
        ax.legend()

    def plot_beam_traces(ax):
        # PPO traces
        for trace in df["beam_trace"]:
            ax.plot(trace, color="tab:blue", alpha=0.3)
        # Baseline traces
        if baseline_df is not None:
            for trace in baseline_df["beam_trace"]:
                ax.plot(trace, color="tab:gray", alpha=0.3)
        ax.axhspan(
            ymin=lower_bound,
            ymax=upper_bound,
            color='green',
            alpha=0.15,
            label="Target Range"
        )
        ax.axhline(lower_bound, color="darkgreen", linestyle=":", linewidth=1)
        ax.axhline(upper_bound, color="darkgreen", linestyle=":", linewidth=1)
        ax.set_xlabel("Step")
        ax.set_ylabel("Beam Intensity")
        ax.set_title("Beam Intensity Traces")
        ax.legend()

    plots = [
        plot_beam_traces,
        plot_final_intensity,
        plot_obs_change,
        plot_total_obs_change,
    ]
    titles = [
        "Beam Intensity Over Time",
        "Final Beam Intensity Comparison",
        "Mean Observation Change",
        "Total Observation Change",
    ]

    fig = plt.figure(figsize=(8, 6))
    gs = plt.GridSpec(2, 1, height_ratios=[20, 1])
    ax = fig.add_subplot(gs[0])
    index = {"i": 0}

    def draw_plot():
        ax.clear()
        plots[index["i"]](ax)
        fig.suptitle(f"{titles[index['i']]} ({index['i']+1}/{len(plots)})", fontsize=12)
        fig.canvas.draw_idle()

    btn_prev_ax = fig.add_axes([0.35, 0.02, 0.1, 0.05])
    btn_next_ax = fig.add_axes([0.55, 0.02, 0.1, 0.05])
    btn_prev = Button(btn_prev_ax, "◀ Previous")
    btn_next = Button(btn_next_ax, "Next ▶")

    def next_plot(event):
        index["i"] = (index["i"] + 1) % len(plots)
        draw_plot()

    def prev_plot(event):
        index["i"] = (index["i"] - 1) % len(plots)
        draw_plot()

    btn_prev.on_clicked(prev_plot)
    btn_next.on_clicked(next_plot)

    draw_plot()
    plt.show()


show_interactive_plots(df, baseline_df)
