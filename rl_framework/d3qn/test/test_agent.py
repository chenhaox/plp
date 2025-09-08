"""
Created on 2025/8/8 
Author: Hao Chen (chen960216@gmail.com)
"""
import sys
import random
import time

import numpy as np
import torch
from agent.agent_nn import D3QN
from rl_env.env import PalletPackingEnv, generate_object_classes

PALLET_DIMENSIONS = (10, 10, 10)  # (width, height, depth)
SKU = 5
# ---- Hyper/Config ----
SEED = 42
EPSILON = 0.05  # small exploration for the smoke test
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# 2. Create the environment
env = PalletPackingEnv(PALLET_DIMENSIONS, item_sku=SKU, )
observation = env.reset()
total_iter_steps = env.total_iter_steps
# Some D3QN actors want action count; some infer internally. Adjust as needed.
d3qn = D3QN(grid_dim=env.pallet_dims[:3],
            num_actions=env.action_nums,
            num_res_block=3,
            num_filters=24,
            head_channels_adv=3,
            head_channels_val=3, )
d3qn = d3qn.to(DEVICE).eval()
# Sample a random action (object dimensions and position)
action = env.sample()
observation, reward, done, info = env.step(action)
manifest_list = [
    (*k, v) for k, v in observation.item_xyz_num_dict.items()
]
manifest_tensor = torch.tensor(manifest_list,
                               dtype=torch.float32).unsqueeze(0).repeat(1, 1, 1)
feasible_actions = env.cal_feasible_action(observation.dp_coord, observation)
a = time.time()
d3qn.forward(
    height_map=torch.as_tensor(observation.height_map[None, None],
                               device=DEVICE, dtype=torch.float32),
    dp_xy=torch.tensor([observation.dp_coord[:2]]),
    z_start=torch.tensor([observation.dp_coord[2,]]),
    manifest=manifest_tensor,
)
b = time.time()
print("Forward time:", b - a)
print("Compactness of the final state:", env.state.compactness)
print("Compactness using w,d of pallet of the final state:", env.state.compactness_wd_pallet_space)
print("Pyramid of the final state:", env.state.pyramid)
print("Number of placed items:", env.num_placed_items, "out of", sum(
        [v['count'] for v in env.initial_objects]
    ))
print("\nSimulation finished. Close the plot window to exit.")
