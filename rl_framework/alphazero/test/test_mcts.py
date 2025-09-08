"""
Created on 2025/9/6 
Author: Hao Chen (chen960216@gmail.com)
"""
from copy import deepcopy
import time

import numpy as np
import ray

from rl_framework.alphazero.config.plp import Config
from rl_framework.alphazero.core.mcts import BatchTree, MCTS
from rl_framework.alphazero.config.base import BaseConfig
from rl_framework.alphazero.core.replay_buffer import TransitionBuffer, ReplayBuffer, MCTSRollingWindow
from rl_framework.alphazero.core.storage import SharedStorage
from rl_framework.alphazero.core.util import MinMaxStats
from rl_env.env import PalletPackingEnv, EnvState

# 1. Define the pallet and the objects to be stacked
PALLET_DIMENSIONS = (10, 10, 10)  # (width, height, depth)
ITEM_SKU = 5
# 2. Create the environment
env = PalletPackingEnv(PALLET_DIMENSIONS,
                       ITEM_SKU,
                       toggle_visualization=True)
envs = [env]

total_iter_steps = len(EnvState._grid)

num_envs = len(envs)
action_nums = env.action_nums
config = Config()
model = config.create_model('cpu', False)  # Create model

roots = BatchTree(num_envs, action_nums, config)  # Prepare datastructures
mcts = MCTS(config, model)
transition_buffers = [TransitionBuffer() for _ in range(num_envs)]
mcts_windows = [MCTSRollingWindow(config.obs_shape, config.frame_stack) for _ in range(num_envs)]
finished = [False] * num_envs

for i, env in enumerate(envs):  # Initialize rolling windows for frame stacking
    mcts_windows[i].add(env.reset().height_map, env.state)

while not all(finished):
    # Prepare roots
    priors, values = model.compute_priors_and_values(
        mcts_windows)  # Compute priors and values for nodes to be expanded

    noises = [np.random.dirichlet([config.root_dirichlet_alpha] * action_nums).astype(np.float32)
              for _ in range(num_envs)]
    roots.prepare(mcts_windows, config.root_exploration_fraction, priors, noises)
    windows = deepcopy(mcts_windows)
    root_visit_dists, root_values = mcts.search(roots, windows)  # Do MCTS search
    # Execute action sampled from MCTS policy
    actions = []
    for env_index, visit_dist in enumerate(root_visit_dists):
        if finished[env_index]:  # We can skip this, because this sub environment is done
            actions.append(None)
            continue
        # Calculate MCTS policy
        assert sum(visit_dist) > 0
        mcts_policy = visit_dist / np.sum(
            visit_dist)  # Convert child visit counts to probability distribution (TODO: temperature)

        # Take maximum visited child as action
        # We do it like this as to randomize action selection for case where visit counts are equal
        action = np.random.choice(np.argwhere(mcts_policy == np.max(mcts_policy)).flatten())
        # action = np.random.choice(range(self.env_action_space.n), p=mcts_policy)  # We could also sample instead of maxing
        actions.append(action)
        obs, reward, done, info = envs[env_index].step(action)  # Apply action
        # Priority by value error
        # priority = nn.L1Loss(reduction='none')(torch.Tensor([values[env_index]]), torch.Tensor([root_values[env_index]])).item()
        # priority += 1e-5
        # TODO: obs vs mcts_window obs
        transition_buffers[env_index].add_one(  # Add experience to data storage
            mcts_windows[env_index].latest_obs(),
            # The observation the action is based upon (vs. `obs`, which is the observation the action generated)
            reward,
            done,
            info,
            mcts_policy,
            root_values[env_index],
            mcts_windows[env_index].env_state,
            1.0  # TODO
        )

        mcts_windows[env_index].add(obs, envs[env_index].get_state(), reward=reward, action=action,
                                    info=info)  # Update rolling window for frame stacking

        if done:
            finished[env_index] = True
            if not config.root_value_targets:  # Overwrite root values calculated during MCTS search with actual trajectory state returns
                transition_buffers[env_index].augment_value_targets(max if config.max_reward_return else sum)

            # Priority by "goodness"
            # accu = max if self.config.max_reward_return else sum
            # priorities = [accu(transition_buffers[env_index].rewards)] * transition_buffers[env_index].size()
            # transition_buffers[env_index].priorities = priorities

    roots.apply_actions(actions)  # Move the tree roots to the new nodes of actions taken
    roots.clear()
