import copy
import numpy as np
import torch
import time
import sys
import signal
import shutil
import collections
import os
import csv
from typing import List, Tuple, Mapping, Text, Any
from ray.rllib.policy.sample_batch import SampleBatch


def padding(size, array):
    if len(array) == 0:
        return np.zeros(size, dtype=int)
    max_v = max(array)
    pad_array = np.ones(size, dtype=int) * max_v
    pad_array[:len(array)] = array
    return pad_array


def feasible_action_to_mask(state_feasible_action_set: torch.Tensor, num_actions: int):
    """
    Convert a feasible action index tensor to a mask tensor where:
    - feasible action indices are set to 1
    - other indices are set to 0

    Args:
    - state_feasible_action_set (torch.Tensor): Tensor of shape [B, N] where each element is an action index
    - num_actions (int): Total number of possible actions (size of mask dimension)

    Returns:
    - torch.Tensor: Mask tensor of shape [B, num_actions] with 1s for feasible actions and 0s elsewhere
    """
    # Initialize the mask tensor of zeros
    mask = torch.zeros(state_feasible_action_set.size(0), num_actions, dtype=torch.bool,
                       device=state_feasible_action_set.device, )

    # Scatter 1s at the indices of feasible actions
    mask.scatter_(1, state_feasible_action_set, 1)

    return mask


class CsvWriter:
    """A logging object writing to a CSV file.

    Each `write()` takes a `OrderedDict`, creating one column in the CSV file for
    each dictionary key on the first call. Successive calls to `write()` must
    contain the same dictionary keys.
    """

    def __init__(self, fname: str):
        """Initializes a `CsvWriter`.

        Args:
          fname: File name(path) for file to be written to.
        """
        if fname is not None and fname != '':
            dirname = os.path.dirname(fname)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

        self._fname = fname
        self._header_written = False
        self._fieldnames = None

    def write(self, values: collections.OrderedDict) -> None:
        """Appends given values as new row to CSV file."""
        if self._fname is None or self._fname == '':
            return

        if self._fieldnames is None:
            self._fieldnames = values.keys()

        # Check if already has rows
        if not self._header_written and os.path.exists(self._fname):
            with open(self._fname, 'r', encoding='utf8') as csv_file:
                content = csv.reader(csv_file)
                if len(list(content)) > 0:
                    self._header_written = True

        # Open a file in 'append' mode, so we can continue logging safely to the
        # same file after e.g. restarting from a checkpoint.
        with open(self._fname, 'a', encoding='utf8') as file_:
            # Always use same fieldnames to create writer, this way a consistency
            # check is performed automatically on each write.
            writer = csv.DictWriter(file_, fieldnames=self._fieldnames)

            # Write a header if this is the very first write.
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(values)

    def close(self) -> None:
        """Closes the `CsvWriter`."""


def write_to_csv(writer: CsvWriter, log_output: List[Tuple]) -> None:
    writer.write(collections.OrderedDict((n, v) for n, v, _ in log_output))


def load_checkpoint(ckpt_file: str, device: torch.device) -> Mapping[Text, Any]:
    return torch.load(ckpt_file, map_location=torch.device(device))


def create_checkpoint(state_to_save: Mapping[Text, Any], ckpt_file: str) -> None:
    torch.save(state_to_save, ckpt_file)


def get_time_stamp(as_file_name: bool = False) -> str:
    t = time.localtime()
    if as_file_name:
        timestamp = time.strftime('%Y%m%d_%H%M%S', t)
    else:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', t)
    return timestamp


def handle_exit_signal():
    """Listen to exit signal like ctrl-c or kill from os and try to exit the process forcefully."""

    def shutdown(signal_code, frame):
        del frame
        print(
            f'Received signal {signal_code}: terminating process...',
        )
        sys.exit(128 + signal_code)

        # # Listen to signals to exit process.

    if sys.platform.startswith('linux'):
        signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)


def disable_auto_grad(network: torch.nn.Module) -> None:
    for p in network.parameters():
        p.requires_grad = False


def delete_all_files_in_directory(folder_path):
    folder_path = str(folder_path)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def is_folder_empty(folder_path):
    str(folder_path)
    return not os.listdir(folder_path)


class Trajectory(object):

    def __init__(self,
                 action_nums,
                 n_step=1,
                 gamma=1.0, ):
        self.n_step = n_step  # Number of steps for multi-step return
        self.action_nums = action_nums
        self.gamma = gamma  # Discount factor
        self.states = []
        self.state_feasible_actions = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.next_state_feasible_actions = []
        self.dones = []

    def add_transition(self,
                       state,
                       action,
                       reward,
                       next_state,
                       done,
                       state_feasible_actions,
                       next_state_feasible_actions):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.state_feasible_actions.append(state_feasible_actions)
        self.next_state_feasible_actions.append(next_state_feasible_actions)

    def compute_n_step_return(self, start_index, end_index):
        """
        Compute the n-step return for a given range of indices.
        :param start_index:
        :param end_index:
        :return:
        """
        discount = 1.0  # Initial discount factor
        # rewards = self.rewards[start_index:end_index]
        # G = sum(rewards)
        G = .0
        for i in range(start_index, end_index):
            G += self.rewards[i] * discount
            discount *= self.gamma  # Apply gamma for the next step
            if self.dones[i]:  # If episode ended, no need to look further
                break
        # Determine the next state based on the range of indices
        next_state_index = end_index - 1  # Default to last index in range
        if end_index < len(self.states):
            next_state = self.next_states[next_state_index]
        else:
            next_state = self.next_states[-1]
        # Check if done signal occurs in the range
        done = any(self.dones[start_index:end_index])
        return G, next_state, done

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        if isinstance(index, slice):
            # start, stop, step = index.indices(len(self.states))
            # if len(self.abs_states) > 0:
            feasible_action_dim = len(self.states[index][0].item_xyz_num_dict) * 2 + 1
            height_maps = []
            height_maps_next = []
            for st in self.states[index]:
                height_maps.append(st.height_map)
                height_maps_next.append(st.height_map)
            states_hm = [st.height_map for st in self.states[index]]
            states_dp = [st.dp_coord for st in self.states[index]]
            state_feasible_actions = [padding(feasible_action_dim, st) for st in
                                      self.state_feasible_actions[index]]
            next_states_hm = [st.height_map for st in self.next_states[index]]
            next_states_dp = [st.dp_coord for st in self.next_states[index]]
            next_state_feasible_actions = [padding(feasible_action_dim, st) for st in
                                           self.next_state_feasible_actions[index]]
            states_manifest = [st.manifest for st in self.states[index]]
            next_states_manifest = [st.manifest for st in self.next_states[index]]
            actions = self.actions[index]
            rewards = self.rewards[index]
            dones = self.dones[index]
            return SampleBatch({'states_hm': states_hm,
                                'states_dp': states_dp,
                                'states_manifest': states_manifest,
                                'state_feasible_action': state_feasible_actions,
                                'action': actions,
                                'reward': rewards,
                                'next_states_hm': next_states_hm,
                                'next_states_dp': next_states_dp,
                                'next_states_manifest': next_states_manifest,
                                'next_state_feasible_action': next_state_feasible_actions,
                                'done': dones,
                                })
        else:
            state_hm = self.states[index].height_map
            state_dp = self.states[index].dp_coord
            next_state_hm = self.next_states[index].height_map
            next_state_dp = self.next_states[index].dp_coord
            feasible_action_dim = len(self.states[index].item_xyz_num_dict) * 2 + 1
            state_feasible_actions = padding(feasible_action_dim, self.state_feasible_actions[index])
            next_state_feasible_actions = padding(feasible_action_dim, self.next_state_feasible_actions[index])
            states_manifest = self.states[index].manifest
            next_states_manifest = self.next_states[index].manifest
            # ---
            reward = self.rewards[index]
            action = self.actions[index]
            done = self.dones[index]
            return SampleBatch({'states_hm': [state_hm],
                                'states_dp': [state_dp],
                                'states_manifest': [states_manifest],
                                'state_feasible_action': [state_feasible_actions],
                                'action': [action],
                                'reward': [reward],
                                'next_states_hm': [next_state_hm],
                                'next_states_dp': [next_state_dp],
                                'next_states_manifest': [next_states_manifest],
                                'next_state_feasible_action': [next_state_feasible_actions],
                                'done': [done],
                                })

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return 'Trajectory(len={})'.format(len(self))
