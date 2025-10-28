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
        self.debug_logging = True

        self._mids = np.array([(ranges[q][1] + ranges[q][0]) / 2 for q in quad_names], dtype=np.float64)
        self._half_ranges = np.array([(ranges[q][1] - ranges[q][0]) / 2 for q in quad_names], dtype=np.float64)
        
        self._lows = np.array([ranges[q][0] for q in quad_names], dtype=np.float64)
        self._highs = np.array([ranges[q][1] for q in quad_names], dtype=np.float64)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n,), dtype=np.float64)
        self.observation_space = spaces.Box(
            low=np.concatenate([np.full(self.n, -1.0), np.array([-1e6])]),
            high=np.concatenate([np.full(self.n, 1.0), np.array([1e6])]),
            dtype=np.float64,
        )

        self.rng = np.random.default_rng(seed)
        self.reset()

    def _normalize(self, vals: np.ndarray) -> np.ndarray:
        return (vals - self._mids) / self._half_ranges

    def _denormalize(self, norm_vals: np.ndarray) -> np.ndarray:
        return norm_vals * self._half_ranges + self._mids

    def _evaluate_beam(self, norm_quads: np.ndarray) -> float:
        physical = self._denormalize(norm_quads)
        mapping = {name: float(val) for name, val in zip(self.quad_names, physical)}
        out = self.model.evaluate(mapping)
        val = out.get("hxr_pulse_intensity")
        if hasattr(val, "detach"):
            val = val.detach().cpu().numpy()
        return float(val)

    def _objective(self, beam_intensity: float) -> float:
        return - (beam_intensity - self.target_power) ** 2

    def _reward(self, prev_obj: float, new_obj: float) -> float:
        r_hat = prev_obj - new_obj
        return r_hat if r_hat > 0 else 2 * r_hat

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0

        default_physical = np.array([self.defaults[q] for q in self.quad_names])
        self.state = self._normalize(default_physical)

        beam = self._evaluate_beam(self.state)
        self._last_objective = self._objective(beam)

        obs = np.concatenate([self.state, [beam]]).astype(np.float64)
        return obs, {}

    def step(self, action: np.ndarray):
        self.step_count += 1

        delta = np.clip(action, -1, 1) * self.step_scale
        new_state = np.clip(self.state + delta, -1, 1)
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
        print(f"Step {self.step_count}: beam = {self._evaluate_beam(self.state):.4f}")