"""
Created on 2025/8/9 
Author: Hao Chen (chen960216@gmail.com)
"""
# --- Example Usage ---
import time
from rl_env.env import PalletPackingEnv, EnvState
# 1. Define the pallet and the objects to be stacked
PALLET_DIMENSIONS = (20, 20, 10)  # (width, height, depth)
ITEM_SKU = 5
# 2. Create the environment
env = PalletPackingEnv(PALLET_DIMENSIONS,
                       ITEM_SKU,
                       toggle_visualization=True)

total_iter_steps = len(EnvState._grid)
for i in range(100):
    st = time.time()
    observation = env.reset()
    for i in range(total_iter_steps):
        # Sample a random action (object dimensions and position)
        action = env.sample()
        observation, reward, done, info = env.step(action)
        if len(observation.item_xyz_num_dict) != 5:
            breakpoint()
            # env.render()
        if done:
            break
    ed = time.time()
    print(f"Total time taken: {ed - st:.2f} seconds")
    print("Compactness of the final state:", env.state.compactness)
    print("Compactness using w,d of pallet of the final state:", env.state.compactness_wd_pallet_space)
    print("Pyramid of the final state:", env.state.pyramid)
    print("Number of placed items:", env.num_placed_items, "out of", sum(
        [v['count'] for v in env.initial_objects]
    ))
    print("\nSimulation finished. Close the plot window to exit.")
