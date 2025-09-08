"""
Created on 2025/9/6 
Author: Hao Chen (chen960216@gmail.com)
"""
from copy import deepcopy
from typing import Optional
import numpy as np
import torch.nn as nn
import gym
from gym.spaces import Discrete

from rl_framework.alphazero.config.base import BaseConfig
from rl_framework.alphazero.core.model import StandardModel
from rl_framework.alphazero.core.util import DiscreteSupport
from rl_env.env import PalletPackingEnv  # adjust if your file/module name differs

"""
PLP Gym Wrapper + Config
Created on 2025/9/6
Author: ChatGPT
"""

from typing import Optional, Tuple, List
import numpy as np
import gym
from gym.spaces import Box, Discrete
import torch.nn as nn

# ----- your training stack -----
from rl_framework.alphazero.config.base import BaseConfig
from rl_framework.alphazero.core.model import StandardModel
from rl_framework.alphazero.core.util import DiscreteSupport

# ----- your PLP base env -----
from rl_env.env import PalletPackingEnv  # adjust if your file/module name differs


class PLPEnv(gym.Env):
    """
    Gym-compatible wrapper around your PalletPackingEnv.

    Observation (float32, all in [0, 1]):
        [ height_map/(Z), feasible_proj/(Z), dp_coord/(X,Y,Z), inventory_norm, time_norm ]
        length = X*Y + X*Y + 3 + item_sku + 1

    Action:
        Discrete(X*Y + 1), where index==X*Y is DO_NOTHING.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            pallet_dims: Tuple[int, int, int] = (10, 10, 10),
            item_sku: int = 5,
            seed: Optional[int] = None,
            cog_uncertainty_ratio: float = 0.1,
            toggle_visualization: bool = False,
            initial_objects: Optional[List[dict]] = None,
    ):
        super().__init__()
        self.base = PalletPackingEnv(
            pallet_dims=pallet_dims,
            item_sku=item_sku,
            seed=seed if seed is not None else 888,
            cog_uncertainty_ratio=cog_uncertainty_ratio,
            toggle_visualization=toggle_visualization,
        )
        # Allow overriding the built-in initial_objects if needed
        if initial_objects is not None:
            self.base.initial_objects = initial_objects

        self.pallet_dims = pallet_dims
        self.item_sku = item_sku
        self.X, self.Y, self.Z = self.pallet_dims
        self.max_steps = self.X * self.Y * self.Z  # equals total scan cells

        # Cache a stable SKU ordering by id (1..N) to keep inventory vector consistent
        self._sku_sorted = sorted(self.base.initial_objects, key=lambda o: o["id"])
        # initial counts per SKU in the same order
        self._sku_initial_counts = np.array([max(1, int(o["count"])) for o in self._sku_sorted], dtype=np.float32)

        # Action space matches your base env definition
        self.action_space = Discrete(self.X * self.Y + 1)
        self.DO_NOTHING_ACTION_IDX = self.X * self.Y

        # Build a dummy obs to size observation_space
        self._time_step = 0
        self._last_state = None
        dummy_state = self.base.reset()
        self._last_state = dummy_state
        dummy_obs = self._state_to_obs(dummy_state, time_step=0)

        low = np.zeros_like(dummy_obs, dtype=np.float32)
        high = np.ones_like(dummy_obs, dtype=np.float32)
        self.observation_space = np.zeros(self.pallet_dims, dtype=np.float32)

    # ---------- Gym API ----------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            # Under the hood your base env seeds numpy RNG and its own RNGs
            self.base.seed(seed)
        self._time_step = 0
        state = self.base.reset()
        self._last_state = state
        obs = self._state_to_obs(state, time_step=self._time_step)
        return obs

    def step(self, action: int):
        """
        Forward to your base env and convert state -> vector obs.
        Returns (obs, reward, terminated, info) in Gym v0 style.
        """
        # Clamp action to valid range as a safety net (your base env checks too)
        if not (0 <= int(action) < self.action_space.n):
            raise ValueError(f"Action {action} out of bounds for Discrete({self.action_space.n}).")

        next_state, reward, done, info = self.base.step(int(action))
        self._time_step = min(self._time_step + 1, self.max_steps)
        self._last_state = next_state

        obs = self._state_to_obs(next_state, time_step=self._time_step)
        terminated = bool(done)  # no explicit truncation logic here
        return obs, float(reward), terminated, info

    def render(self, mode="human"):
        # Delegate to your nice 3D + heatmap renderer
        # self.base.render(mode=mode)
        pass

    def close(self):
        # No special resources to free beyond matplotlib windows handled by base.render()
        pass

    def set_state(self, state):
        obs = self.base.set_state(state)
        return obs

    def _state_to_obs(self, state, time_step: int) -> np.ndarray:
        """
        Build a normalized, flat feature vector from EnvState.
        """
        X, Y, Z = self.X, self.Y, self.Z
        Zf = float(max(1, Z))

        # 1) height_map normalized by Z
        hm = state.height_map.astype(np.float32) / Zf  # (X, Y)
        hm_flat = hm.ravel()

        # 2) feasibility projection (like your render) normalized by Z
        feas = state.feasibility_map  # (X, Y, Z), bool
        # z indices start at 1 to keep ground level visible; same as your render
        z_idx = (np.arange(feas.shape[2], dtype=np.float32) + 1.0).reshape(1, 1, -1)
        proj = (feas * z_idx).max(axis=2).astype(np.float32) / Zf  # (X, Y)
        proj_flat = proj.ravel()

        # 3) dp_coord normalized
        dp = np.array(state.dp_coord, dtype=np.float32)
        dp_norm = dp / np.array([X, Y, Z], dtype=np.float32)

        # 4) inventory vector (order by self._sku_sorted), normalized by initial counts
        #    Keys in state.item_xyz_num_dict are canonical (w0,d0,h)
        inv_counts = np.zeros(len(self._sku_sorted), dtype=np.float32)
        for i, sku in enumerate(self._sku_sorted):
            key = tuple(sku["dims"])
            inv_counts[i] = float(state.item_xyz_num_dict.get(key, 0))
        inv_norm = inv_counts / self._sku_initial_counts

        # 5) normalized time
        t_norm = np.array([min(1.0, time_step / float(max(1, self.max_steps)))], dtype=np.float32)

        obs = np.concatenate([hm_flat, proj_flat, dp_norm.astype(np.float32), inv_norm, t_norm]).astype(np.float32)
        return obs


class Config(BaseConfig):
    """
    Training config tailored for the PLP Gym wrapper.
    Mirrors your Cartpole example, but adapts input/output sizes automatically.
    """

    def __init__(
            self,
            # ---- training loop ----
            training_steps: int = 2000,
            pretrain_steps: int = 0,
            model_broadcast_interval: int = 25,
            num_sgd_iter: int = 10,
            clear_buffer_after_broadcast: bool = False,
            root_value_targets: bool = False,
            replay_buffer_size: int = 100_000,
            demo_buffer_size: int = 0,
            batch_size: int = 512,
            lr: float = 2e-3,
            max_grad_norm: float = 5.0,
            weight_decay: float = 1e-4,
            momentum: float = 0.9,
            c_init: float = 3.0,
            c_base: float = 19_652,
            gamma: float = 0.997,
            frame_stack: int = 4,
            max_reward_return: bool = False,
            hash_nodes: bool = False,
            root_dirichlet_alpha: float = 1.5,
            root_exploration_fraction: float = 0.25,
            num_simulations: int = 50,
            num_envs_per_worker: int = 1,
            min_num_episodes_per_worker: int = 2,
            use_dirichlet: bool = True,
            test_use_dirichlet: bool = False,
            value_support: DiscreteSupport = DiscreteSupport(0, 50, 1.0),
            value_transform: bool = True,
            env_seed: Optional[int] = None,
            # ---- PLP-specific ----
            pallet_dims: Tuple[int, int, int] = (10, 10, 10),
            item_sku: int = 5,
            cog_uncertainty_ratio: float = 0.1,
            toggle_visualization: bool = False,
            initial_objects: Optional[List[dict]] = None,
    ):
        super().__init__(
            training_steps,
            pretrain_steps,
            model_broadcast_interval,
            num_sgd_iter,
            clear_buffer_after_broadcast,
            root_value_targets,
            replay_buffer_size,
            demo_buffer_size,
            batch_size,
            lr,
            max_grad_norm,
            weight_decay,
            momentum,
            c_init,
            c_base,
            gamma,
            frame_stack,
            max_reward_return,
            hash_nodes,
            root_dirichlet_alpha,
            root_exploration_fraction,
            num_simulations,
            num_envs_per_worker,
            min_num_episodes_per_worker,
            use_dirichlet,
            test_use_dirichlet,
            value_support,
            value_transform,
            env_seed,
        )
        # Store PLP knobs
        self.pallet_dims = pallet_dims
        self.item_sku = item_sku
        self.cog_uncertainty_ratio = cog_uncertainty_ratio
        self.toggle_visualization = toggle_visualization
        self.initial_objects = initial_objects

    def env_creator(self):
        return PLPEnv(
            pallet_dims=self.pallet_dims,
            item_sku=self.item_sku,
            seed=self.env_seed,
            cog_uncertainty_ratio=self.cog_uncertainty_ratio,
            toggle_visualization=self.toggle_visualization,
            initial_objects=self.initial_objects,
        )

    def create_model(self, device, amp):
        probe_env = self.env_creator()
        obs_shape = probe_env.observation_space.shape  # (D,)
        num_act = probe_env.action_space.n

        model = StandardModel(self, obs_shape, num_act, device, amp)
        hidden = 512

        model.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(obs_shape) * self.frame_stack, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        model.actor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_act),
        )
        model.critic = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.value_support.size),
        )
        model.to(device)
        return model
