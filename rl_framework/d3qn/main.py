"""
Created on 2025/8/9 
Author: Hao Chen (chen960216@gmail.com)
"""
# TODO
# 1. Actor Update SKU number (it should be a state need to be update frequently)
# 2. 考虑是否把没用数据用于训练
""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230720osaka
TODO:
    - num classes can be set:
"""

import copy
import time

import ray
import hydra
import torch
import file_utils as fu
from rl_env.env import PalletPackingEnv, generate_object_classes
from rl_framework.d3qn.pipeline import Actor, SharedState
from rl_framework.d3qn.learner import Learner
from rl_framework.d3qn.utils import get_time_stamp
from agent.agent_nn import D3QN
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
# from huri.learning.method.AlphaZero.utils import delete_all_files_in_directory
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import random

config_name = 'cfg.yaml'


def seed_everything(seed: int):
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def copy_file(src, dst):
    import shutil
    shutil.copy(src, dst)


def get_network_source_code(module_function):
    import inspect
    # Get the source code of the module function
    source_code = inspect.getsource(module_function)
    return source_code


def init_folder():
    save_path = fu.workdir_d3qn / 'run'
    fu.delete_all_files_in_directory(str(save_path))
    save_path.joinpath('log').mkdir(exist_ok=True)
    save_path.joinpath('data').mkdir(exist_ok=True)
    save_path.joinpath('params').mkdir(exist_ok=True)
    # copy config file
    copy_file(fu.workdir_d3qn / 'cfg' / config_name,
              save_path.joinpath('params', 'cfg.yaml'))
    import_text = 'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport math\nfrom typing import NamedTuple, Tuple\n'
    save_path.joinpath('params', 'network.py').write_text(import_text + get_network_source_code(D3QN), encoding='utf-8')
    return save_path


@hydra.main(config_path='cfg', config_name=config_name, version_base='1.3')
def main(cfg):
    ray.init(
        dashboard_host='0.0.0.0',
        object_store_memory=30 * 10 ** 9,
    )
    print("cfg information: ", cfg)
    print("Start ... ")
    # setup seed
    seed_everything(cfg['env']['seed'])
    # init saving path
    save_path = init_folder()
    # delete all sesstion files in ray
    # setup environment
    PALLET_DIMENSIONS = cfg['env']['pallet_dimensions']
    SKU = cfg['env']['sku']
    env_meta = PalletPackingEnv(PALLET_DIMENSIONS, SKU)
    # setup neural network
    grid_dim = env_meta.pallet_dims[:3]
    num_actions = env_meta.action_nums
    network = D3QN(grid_dim=grid_dim,
                   num_actions=num_actions,
                   num_filters=cfg['ddqn']['num_filters'],
                   num_res_block=cfg['ddqn']['num_res_block'],
                   head_channels_adv=cfg['ddqn']['head_channels_adv'],
                   head_channels_val=cfg['ddqn']['head_channels_val'],
                   manifest_emb_dim=cfg['ddqn']['manifest_emb_dim'],
                   )
    # start replay buffer
    alpha = .6
    replay_buffer = ray.remote(PrioritizedReplayBuffer).options(
        num_cpus=5, ).remote(capacity=int(cfg['rl']['replay_sz']), alpha=alpha, )
    # start shared state
    ckpnt = {'train_steps': 0,
             'weights': network.state_dict(),
             # 'weights': torch.load(
             #     r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed\runs\run_2\data\model_last.chkpt')[
             #     'dqn_state_dict'],
             'eval_average_len': 0, }
    print(ckpnt.keys())
    shared_state = SharedState.options(num_cpus=5).remote(ckpnt)
    # start Learner
    learner = ray.remote(Learner,
                         ).options(
        num_cpus=3,
        num_gpus=2,
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(),
                                                           soft=False)).remote(
        net=copy.deepcopy(network),
        cfg=cfg['rl'],
        shared_state=shared_state,
        replay_buffer=replay_buffer,
        log_path=save_path.joinpath('log'), )
    # start actor
    actor_procs = []
    RayActor = ray.remote(Actor, )
    for i in range(cfg['num_actor']):
        env = env_meta.copy()
        env.seed((i + 10) * cfg['env']['seed'])
        # r_b = replay_buffer
        actor = RayActor.options(num_cpus=1).remote(actor_id=i,
                                                    env=env,
                                                    net=copy.deepcopy(network),
                                                    cfg=cfg['rl'],
                                                    replay_buffer=replay_buffer,
                                                    shared_state=shared_state,
                                                    log_path=save_path.joinpath('log'), )
        actor_procs.append(actor)
    print("start learner")
    learner.start.remote()
    print("start actor")
    [actor.start.remote() for actor in actor_procs]

    while True:
        buff_len1 = ray.get(replay_buffer.__len__.remote())
        print(f"[{get_time_stamp()}] Replay buffer length: {buff_len1}, ", )
        time.sleep(cfg['eval']['eval_interval'] + 3)


if __name__ == '__main__':
    main()
