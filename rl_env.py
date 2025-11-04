import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional


class LTUHEnv(gym.Env):
    """
    Reinforcement learning environment for optimizing FEL quadrupoles.
    Everything is stored and updated in normalized space [-1, 1].
    Only the surrogate model sees denormalized physical values.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        quad_names: List[str],
        ranges: Dict[str, List[float]],
        defaults: Dict[str, float],
        model,
        target_power: float = 2.25,
        step_scale: float = 0.05,
        max_steps: int = 100,
        seed: Optional[int] = None,
        normal_noise_std: float = 0.05,  
        uniform_sample_prob: float = 0.05,
    ):
        super().__init__()
        self.quad_names = quad_names
        self.ranges = ranges
        self.defaults = defaults
        self.model = model
        self.target_power = target_power
        self.step_scale = step_scale
        self.max_steps = max_steps
        self.n = len(quad_names)

        self.normal_noise_std = normal_noise_std
        self.uniform_sample_prob = uniform_sample_prob

        self._mids = np.array([(ranges[q][1] + ranges[q][0]) / 2 for q in quad_names], dtype=np.float64)
        self._half_ranges = np.array([(ranges[q][1] - ranges[q][0]) / 2 for q in quad_names], dtype=np.float64)
        self._default_normalized = self._normalize(np.array([defaults[q] for q in quad_names]))

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n,), dtype=np.float64)
        self.observation_space = spaces.Box(
            low=np.concatenate([np.full(self.n, -1.0), np.array([-1e6])]),
            high=np.concatenate([np.full(self.n, 1.0), np.array([1e6])]),
            dtype=np.float64,
        )

        self.rng = np.random.default_rng(seed)
        self.reset()

        print("\n--- Environment Check ---")
        print(f"{'Quad Name':<18} | {'Min Range':<18} | {'Max Range':<18} | {'Midpoint':<18} | {'Half Range':<18}")
        print("-" * 90)

        min_vals = np.array([self.ranges[q][0] for q in self.quad_names])
        max_vals = np.array([self.ranges[q][1] for q in self.quad_names])

        for i, name in enumerate(self.quad_names):
            print(f"{name:<18} | {min_vals[i]:<18.8f} | {max_vals[i]:<18.8f} | {self._mids[i]:<18.8f} | {self._half_ranges[i]:<18.8f}")
        
        print("--- End Evnironment Check ---\n")

    def _normalize(self, vals: np.ndarray) -> np.ndarray:
        return (vals - self._mids) / self._half_ranges

    def _denormalize(self, norm_vals: np.ndarray) -> np.ndarray:
        return norm_vals * self._half_ranges + self._mids

    def _evaluate_beam(self, norm_quads: np.ndarray) -> float:
        physical = self._denormalize(norm_quads)

        min_vals = np.array([self.ranges[q][0] for q in self.quad_names])
        max_vals = np.array([self.ranges[q][1] for q in self.quad_names])

        physical_clipped = np.clip(physical, min_vals, max_vals)

        mapping = {name: float(val) for name, val in zip(self.quad_names, physical_clipped)}
        out = self.model.evaluate(mapping)
        val = out.get("hxr_pulse_intensity")
        if hasattr(val, "detach"):
            val = val.detach().cpu().numpy()
        return float(val)

    def _objective(self, beam_intensity: float) -> float:
        return - (beam_intensity - self.target_power) ** 2

    def _reward(self, prev_obj: float, new_obj: float) -> float:
        r_hat = new_obj - prev_obj
        return r_hat if r_hat > 0 else 2 * r_hat 

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0

        if self.rng.random() < self.uniform_sample_prob:
            new_state = self.rng.uniform(low=-1.0, high=1.0, size=self.n)
        else:
            noise = self.rng.normal(loc=0.0, scale=self.normal_noise_std, size=self.n)
            new_state = self._default_normalized + noise
            
            new_state = np.clip(new_state, -1.0, 1.0)
            
        self.state = new_state.astype(np.float64)
        

        beam = self._evaluate_beam(self.state)
        self._last_objective = self._objective(beam)

        obs = np.concatenate([self.state, [beam]]).astype(np.float64)
        return obs, {}

    def step(self, action: np.ndarray):
      self.step_count += 1

      new_state = np.clip(action, -1, 1) 

      beam = self._evaluate_beam(new_state)

      new_obj = self._objective(beam)
      reward = self._reward(self._last_objective, new_obj)

      self.state = new_state
      self._last_objective = new_obj

      obs = np.concatenate([self.state, [beam]]).astype(np.float64)
      terminated = False
      truncated = self.step_count >= self.max_steps
      info = {"beam_intensity": beam, "objective": new_obj}
      return obs, reward, terminated, truncated, info

    def render(self):
      physical_vals = self._denormalize(self.state)

      print(f"\n--- Step {self.step_count} ---")

      current_beam = self._evaluate_beam(self.state)
      print(f"  Beam Intensity: {current_beam:.4f} (Target: {self.target_power})")
      print(f"  Objective:      {self._last_objective:.4f}")

      print("\n  Quadrupole Physical Values:")
      for i, name in enumerate(self.quad_names):
          print(f"    {name}: {physical_vals[i]:.4f}  (Normalized: {self.state[i]:.4f})")