"""
Created on 2025/8/9 
Author: Hao Chen (chen960216@gmail.com)
"""
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import hydra

from agent.agent_nn import D3QN
from rl_env.env import PalletPackingEnv, EnvState
from rl_framework.d3qn.pipeline import dqn_select_action


class D3QNSolver:
    """
    Thin wrapper that:
      - holds env + net
      - loads weights
      - performs a single greedy rollout (or with small epsilon if desired)
      - returns a compact result dict with the full trace
    """

    def __init__(
            self,
            env: PalletPackingEnv,
            net: D3QN,
            device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    ):
        assert isinstance(env, PalletPackingEnv), f"env must be PalletPackingEnv, got {type(env)}"
        self.env = env
        self.net = net.to(device)
        self.device = device
        self.net.eval()

    # ------------------------------
    # Weights management
    # ------------------------------
    def load_weights_from_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.net.load_state_dict(state_dict)
        self.net.eval()

    def load_weights_from_file(self, path: str or Path) -> None:
        path = Path(path)
        state = torch.load(path, map_location="cpu")
        # allow either raw state_dict or a checkpoint dict with 'weights'
        if isinstance(state, dict) and "network" in state:
            self.net.load_state_dict(state["network"])
        elif isinstance(state, dict):
            # assume it's already a state_dict
            self.net.load_state_dict(state)
        else:
            raise ValueError(f"Unrecognized checkpoint format at {path}")
        self.net.to(self.device)
        self.net.eval()

    # ------------------------------
    # Rollout / Solve
    # ------------------------------
    @torch.no_grad()
    def solve(self, render: bool = False) -> Dict[str, Any]:
        """
        Runs one evaluation episode from env.reset() until done or step cap.

        Returns:
            {
                "total_reward": float,
                "steps": int,
                "done": bool,
                "trace": [
                    {
                        "t": int,
                        "state": <optional minimal snapshot>,
                        "action": int,
                        "reward": float,
                        "info": dict,
                    }, ...
                ],
                "final_state": <optional minimal snapshot>,
            }
        """
        # config overrides at call site
        # reset
        state: EnvState = self.env.reset()
        total_reward = 0.0
        trace: List[Dict[str, Any]] = []
        if render:
            self.env._prepare_visualization()

        # A small helper to choose action with (possible) eval epsilon
        def pick_action(s: EnvState) -> int:
            # otherwise greedy over feasible set using your utility
            feasible = self.env.cal_feasible_action(s.dp_coord, s)
            if feasible.size == 0:
                return int(self.env.DO_NOTHING_ACTION_IDX)
            if feasible.size == 1:
                return int(feasible[0])
            act = dqn_select_action(
                feasible_action_set=feasible,
                state=s,
                dqn=self.net,
                device=self.device,
            )
            return int(act.action_select)

        # rollout
        t = 0
        done = False
        while not done:
            action = pick_action(state)
            next_state, reward, done, info = self.env.step(action)
            total_reward += float(reward)
            # minimal (compact) snapshot — adjust to your needs
            trace.append({
                "t": t,
                "action": int(action),
                "reward": float(reward),
                "info": info if isinstance(info, dict) else {},
            })
            if action != self.env.DO_NOTHING_ACTION_IDX:
                if render and hasattr(self.env, "render"):
                    self.env.render()
                    print(f"Step {t}: action={action}, reward={reward}, done={done}, info={info}")
            state = next_state
            t += 1

        print("\n--- Summary ---")
        print("Accumulated reward:", total_reward)
        print("Compactness of the final state:", state.compactness)
        print("Compactness using w,d of pallet of the final state:", state.compactness_wd_pallet_space)
        print("Pyramid of the final state:", state.pyramid)
        print("Number of placed items:", self.env.num_placed_items, "out of", sum(
            [v['count'] for v in self.env.initial_objects]
        ))
        print("\nSimulation finished. Close the plot window to exit.")
        result = {
            "total_reward": float(total_reward),
            "steps": int(t),
            "done": bool(done),
            "trace": trace,
            "final_state": {
                # place anything you want to export; keeping it light here
                "dp_coord": getattr(state, "dp_coord", None).tolist() if hasattr(state, "dp_coord") else None,
            },
        }
        return result


def main(cfg: Dict[str, Any]) -> None:
    # Initialize the environment and solver with the provided configuration
    PALLET_DIMENSIONS = cfg['env']['pallet_dimensions']
    SKU = cfg['env']['sku']
    env = PalletPackingEnv(PALLET_DIMENSIONS, SKU)
    grid_dim = env.pallet_dims[:3]
    num_actions = env.action_nums
    net = D3QN(grid_dim=grid_dim,
               num_actions=num_actions,
               num_filters=cfg['ddqn']['num_filters'],
               num_res_block=cfg['ddqn']['num_res_block'],
               head_channels_adv=cfg['ddqn']['head_channels_adv'],
               head_channels_val=cfg['ddqn']['head_channels_val'],
               manifest_emb_dim=cfg['ddqn']['manifest_emb_dim'],
               )
    solver = D3QNSolver(env=env.copy(), net=net)

    # Load weights if specified in the config
    p = Path(__file__).parent.joinpath(r"./run/data/model.chkpt")
    solver.load_weights_from_file(str(p))

    # Solve the environment and print the result
    result = solver.solve(render=True)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    config_name = 'cfg.yaml'
    main = hydra.main(config_path='cfg',
                      config_name=config_name,
                      version_base='1.3', )(main)
    main()
    # plt.show()
