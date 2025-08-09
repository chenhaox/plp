"""
Created on 2025/8/8 
Author: Hao Chen (chen960216@gmail.com)
"""
import ray
import itertools
from collections import namedtuple
import numpy as np
import torch
import time
import copy
from pathlib import Path
from typing import NamedTuple
from rl_env.env import PalletPackingEnv, EnvState
from rl_framework.d3qn.utils import CsvWriter, write_to_csv, get_time_stamp, Trajectory

MAX_NUM_PENDING_TASKS = 100
MAX_SEND_PERIOD = 0
MAX_REPLAY_BUFFER_LEN = 6000


class ActorAction(NamedTuple):
    """
    Action class for the pallet packing environment.
    """
    action_value: np.ndarray
    action_select: int

    def is_action_select_valid(self):
        if self.action_select >= 0:
            return True
        else:
            return False


# def inverse_value_rescale(value, eps=1e-3):
#     temp = ((1 + 4 * eps * (value.abs() + 1 + eps)).sqrt() - 1) / (2 * eps)
#     return value.sign() * (temp.square() - 1)


# CheckPoint = namedtuple('CheckPoint', ['weights', 'training_level', 'train_steps'])
@ray.remote
class SharedState:
    """
    Class which run in a dedicated thread to store the network weights and some information.
    """

    def __init__(self, checkpoint):
        self.current_checkpoint = copy.deepcopy(checkpoint)

    # def save_checkpoint(self, path=None):
    #     if not path:
    #         path = self.config.results_path / "model.checkpoint"
    #
    #     torch.save(self.current_checkpoint, path)

    def get_checkpoint(self):
        return copy.deepcopy(self.current_checkpoint)

    def get_info(self, keys):
        if isinstance(keys, str):
            return self.current_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_checkpoint[key] for key in keys}
        else:
            raise TypeError

    def set_info(self, keys, values=None):
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError

    def extend_info(self, keys, values):
        if isinstance(keys, str) and isinstance(values, list):
            self.current_checkpoint[keys].extend(values)
            if len(self.current_checkpoint[keys]) > MAX_REPLAY_BUFFER_LEN:
                # self.current_checkpoint[keys] = self.current_checkpoint[keys][-MAX_REPLAY_BUFFER_LEN:]
                del self.current_checkpoint[keys][:MAX_REPLAY_BUFFER_LEN // 5]
        else:
            raise TypeError

    def get_info_pop(self, key, n=1):
        assert isinstance(key, str) and isinstance(self.current_checkpoint[key], list)
        if len(self.current_checkpoint[key]) > 0:
            n = min(len(self.current_checkpoint[key]), n)
            popped_elements = self.current_checkpoint[key][-n:]
            del self.current_checkpoint[key][-n:]
            return popped_elements
        else:
            return None

    def get_info_len(self, key):
        assert isinstance(key, str) and isinstance(self.current_checkpoint[key], list)
        return len(self.current_checkpoint[key])

    def set_info_clear(self, key):
        assert isinstance(key, str) and isinstance(self.current_checkpoint[key], list)
        self.current_checkpoint[key].clear()


def _encode_state_for_dqn(state: EnvState, device: str) -> torch.Tensor:
    """
    TODO: Replace with your real encoder.
    For now we pack height_map and a max-feasible-height projection as channels.
    """
    hmap = state.height_map.astype(np.float32)  # (X, Y)
    # project feasibility to the top-most feasible z at each (x,y)
    manifest_list = np.asarray([
        (*k, v) for k, v in state.item_xyz_num_dict.items()
    ])
    dp_coord = state.dp_coord.astype(np.float32)  # (X, Y)
    dp_xy = dp_coord[:2]
    z_start = dp_coord[2]
    return (torch.as_tensor(hmap[None, None],
                            device=device, dtype=torch.float32),
            torch.tensor([dp_xy]),
            torch.tensor([z_start]),
            torch.tensor(manifest_list,
                         dtype=torch.float32).unsqueeze(0).repeat(1, 1, 1)
            )


def dqn_select_action(feasible_action_set: np.ndarray,
                      state: EnvState,
                      dqn: torch.nn.Module,
                      device: str, ) -> ActorAction:
    with torch.no_grad():
        # TODO encode state
        obs = _encode_state_for_dqn(state, device)
        q = dqn(height_map=obs[0],
                dp_xy=obs[1],
                z_start=obs[2],
                manifest=obs[3], ).detach().squeeze(0).squeeze()
        if q.ndim != 1:
            raise RuntimeError(f"DQN must return 1D action-values, got shape {tuple(q.shape)}")
        feasible_action_set = torch.as_tensor(feasible_action_set, dtype=torch.int64, device=device)
        selected_action = feasible_action_set[q[feasible_action_set].argmax()].item()
    return ActorAction(
        action_value=q.cpu().numpy(),
        action_select=selected_action
    )


class EpsilonScheduler:
    def __init__(self, initial_epsilon, final_epsilon, decay_rate):
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.decay_rate = decay_rate
        self.current_step = 0

    def get_epsilon(self):
        decay_factor = max(0, 1 - self.decay_rate * self.current_step)
        epsilon = self.initial_epsilon * decay_factor

        return max(epsilon, self.final_epsilon)

    def step(self):
        self.current_step += 1
        return self.get_epsilon()

    def reset(self):
        self.current_step = 0

    def half_reset(self):
        self.initial_epsilon = .5
        self.current_step = 0


def debug_func(traj_list, replay_buffer):
    for i in traj_list:
        replay_buffer.add.remote(i[:])


# @ray.remote(num_cpus=1)
# TODO: Init when using
class Actor(object):
    """
    Actor object to start simulation and collect data
    """

    def __init__(self,
                 actor_id: int,
                 env: PalletPackingEnv,
                 net: 'DDQN',
                 cfg: dict,
                 replay_buffer: ray.actor.ActorClass,
                 shared_state: ray.actor.ActorClass,
                 log_path=None,
                 toggle_visual: bool = False,
                 ):
        assert isinstance(env, PalletPackingEnv), f'Only accept the env as PalletPackingEnv. ' \
                                                  f'Instead of {type(env)}'
        self._actor_id = actor_id
        self.env = env
        self.device = cfg['actor_device']
        # network
        self.dqn = net.to(self.device)
        self.dqn.eval()
        # set up replay buffer size
        self.replay_buffer = replay_buffer
        self.shared_state = shared_state
        # set up scheduler
        self.epsilon_decay = cfg['eps_decay']
        self.max_epsilon = cfg['eps_max']
        self.min_epsilon = cfg['eps_min']
        self.epsilon_scheduler = EpsilonScheduler(self.max_epsilon, self.min_epsilon, self.epsilon_decay)
        # hyper-parameter
        self.target_update_freq = cfg['update_freq']
        self.send_period = cfg['send_period']  # periodically send data to replay
        # log_path
        self.log_path = Path(log_path) if log_path is not None else None
        # n-step return
        self.n_step = cfg['n_step']
        self.gamma = cfg['gamma']
        # trajectory list
        self.trajectory_list = []
        # init replay sz
        self.init_replay_num = cfg['init_replay_num']
        self.average_len = 999
        self.toggle_visual = toggle_visual

    @property
    def epsilon(self):
        return self.epsilon_scheduler.get_epsilon()

    def sync_dqn_checkpoint(self):
        weights = ray.get(self.shared_state.get_info.remote('weights'))
        self.dqn.load_state_dict(weights)

    def select_action(self,
                      state: EnvState,
                      env: PalletPackingEnv, ) -> ActorAction:
        if self.epsilon > np.random.random():
            selected_action = env.sample()
            selected_action = ActorAction(action_value=np.array([]),
                                          action_select=selected_action)
        else:
            feasible_action_set = env.cal_feasible_action(
                state.dp_coord,
                state
            )
            if feasible_action_set.size == 0:
                return ActorAction(action_value=np.array([]), action_select=env.DO_NOTHING_ACTION_IDX)
            selected_action = dqn_select_action(state=state,
                                                dqn=self.dqn,
                                                device=self.device,
                                                feasible_action_set=feasible_action_set, )
        return selected_action

    def store_traj_list(self, ):
        samples = self.trajectory_list[0][:]
        for traj in self.trajectory_list[1:]:
            samples = samples.concat(traj[:])  # concat trajectory
        self.replay_buffer.add.remote(samples)
        del self.trajectory_list[:]

    def start(self, output_interval=100):
        """Actor starts"""
        env = self.env
        action_nums = env.action_nums
        target_update_freq = self.target_update_freq
        if self.log_path is not None and isinstance(self.log_path, Path):
            writer = CsvWriter(str(self.log_path.joinpath(f'actor_{self._actor_id}_log.csv')))
        # ====
        reset_num = self.env.total_iter_steps
        # dqn load shared net weight
        self.sync_dqn_checkpoint()
        # start actors
        step = 0
        episode = 0
        while 1:
            # rest counter, score
            episode += 1
            reset_cnt = 0
            score = 0
            # curriculum learning
            state = env.reset()
            traj = Trajectory(action_nums=action_nums,
                              n_step=self.n_step,
                              gamma=self.gamma, )
            start_ep_t = time.time()
            for e_step in itertools.count(1):
                # select action
                action = self.select_action(state, env)
                if not action.is_action_select_valid():
                    raise ValueError(f"Invalid action selected by actor {self._actor_id}: {action.action_select}")
                # step
                next_state, reward, done, _ = self.env.step(action.action_select)  # next_state reward done
                if reset_cnt == reset_num:
                    done = True
                # update score, coutner
                score += reward  # reward
                reset_cnt += 1
                step += 1
                # store trajectory
                traj.add_transition(state,
                                    action.action_select,
                                    reward,
                                    next_state,
                                    done,
                                    next_state_feasible_actions=env.cal_feasible_action(
                                        next_state.dp_coord,
                                        next_state
                                    ), )
                # reset state# reset state
                state = next_state
                # update
                if step % target_update_freq == 0:
                    self.sync_dqn_checkpoint()
                # if episode ends
                if done:  # done
                    # TODO addd
                    self.trajectory_list.append(traj)
                    break
                if reset_cnt % reset_num == 0:
                    # TODO
                    self.trajectory_list.append(traj)
                    break
            end_ep_t = time.time()
            # linearly decrease epsilon
            self.epsilon_scheduler.step()
            if len(self.trajectory_list) >= self.send_period:
                # self.store_traj_list(trajectory_list,
                #                      self.store_reanalyzer if self.epsilon > np.random.random() else False)
                self.store_traj_list()
            if step % output_interval == 0 and self.log_path is not None and isinstance(self.log_path, Path):
                write_to_csv(writer, [
                    ('timestamp', get_time_stamp(), '%1s'),
                    ('step', step, '%1d'),
                    ('time_consumption', end_ep_t - start_ep_t, '%1s'),
                    ('episode_lens', len(traj) + 1, '%1d'),
                    ('acc_rewards', score, '%1f'),
                    ('epsilon', self.epsilon, '%1f'),
                    ('episode', episode, '%1f'),
                ])


if __name__ == '__main__':
    pass
