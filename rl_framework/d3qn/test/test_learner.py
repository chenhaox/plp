"""
Created on 2025/8/9 
Author: Hao Chen (chen960216@gmail.com)
"""
"""
Created on 2025/8/8 
Author: Hao Chen (chen960216@gmail.com)
"""
import copy
import time
import ray
import hydra
import torch
import random
import numpy as np
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

import file_utils as fu
from agent.agent_nn import D3QN
from rl_env.env import PalletPackingEnv
from rl_framework.d3qn.pipeline import SharedState, Actor
from rl_framework.d3qn.learner import Learner


@hydra.main(config_path='../cfg', config_name='cfg.yaml', version_base='1.3')
def main(cfg):
    # ray.init(local_mode=True, )
    ray.init()
    SEED = 42
    EPSILON = 0.05  # small exploration for the smoke test
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    PALLET_DIMENSIONS = (20, 20, 10)  # (width, height, depth)
    SKU = 5
    save_path = fu.workdir_d3qn / 'run'
    env = PalletPackingEnv(PALLET_DIMENSIONS, item_sku=SKU, )
    network = D3QN(grid_dim=env.pallet_dims[:3],
                   num_actions=env.action_nums, )
    replay_buffer = ray.remote(PrioritizedReplayBuffer).options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(),
                                                           soft=False)).remote(capacity=10,
                                                                               alpha=0.6)
    # cfg['env']['init_state_level']
    ckpnt = {'weights': network.state_dict(),
             'training_level': 60, }
    shared_state = SharedState.remote(ckpnt)
    RayActor = ray.remote(Actor)
    cfg['rl']['send_period'] = 3
    actor = RayActor.remote(actor_id=1,
                            env=copy.deepcopy(env),
                            net=copy.deepcopy(network),
                            cfg=cfg['rl'],
                            replay_buffer=replay_buffer,
                            shared_state=shared_state,
                            log_path=save_path.joinpath('log'),
                            toggle_visual=False)
    actor.start.remote()
    cfg['rl']['batch_sz'] = 10
    learner = Learner(net=copy.deepcopy(network),
                      cfg=cfg['rl'],
                      shared_state=shared_state,
                      replay_buffer=replay_buffer,
                      log_path=save_path.joinpath('log'), )
    learner.start(min_replay_sz=9)
    while True:
        buff_len = ray.get(replay_buffer.__len__.remote())
        print(buff_len, )
        time.sleep(1)


if __name__ == '__main__':
    main()
