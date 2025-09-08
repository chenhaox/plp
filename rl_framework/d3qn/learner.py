"""
Created on 2025/8/8 
Author: Hao Chen (chen960216@gmail.com)
"""
""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230720osaka

"""
import time
import copy
import torch
import torch.nn.functional as F
import ray
import numpy as np
from rl_env.env import PalletPackingEnv
import file_utils as fu
from rl_framework.d3qn.utils import CsvWriter, write_to_csv, feasible_action_to_mask, get_time_stamp
from ray.rllib.policy.sample_batch import SampleBatch

prior_eps: float = 1e-6


def get_state_to_save(network,
                      target_network,
                      optimizer,
                      train_steps):
    return {
        'network': network.state_dict(),
        'target_network': target_network.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_steps': train_steps,
    }


def value_rescale(value, eps=1e-3):
    return value.sign() * ((value.abs() + 1).sqrt() - 1) + eps * value


def inverse_value_rescale(value, eps=1e-3):
    temp = ((1 + 4 * eps * (value.abs() + 1 + eps)).sqrt() - 1) / (2 * eps)
    return value.sign() * (temp.square() - 1)


class BetaScheduler:
    def __init__(self, initial_beta: float, final_beta: float, total_steps: int, mode='linear'):
        """
        Initializes the BetaScheduler.

        Args:
            initial_beta (float): The initial value of beta.
            final_beta (float): The final value of beta (usually 1.0).
            total_steps (int): The total number of steps over which beta will be increased.
            mode (str): The mode of increase ('linear' or 'exponential').
        """
        assert 0 <= initial_beta <= 1, "Initial beta must be between 0 and 1."
        assert 0 <= final_beta <= 1, "Final beta must be between 0 and 1."
        assert initial_beta <= final_beta, "Final beta must be greater than or equal to initial beta."
        assert total_steps > 0, "Total steps must be a positive number."
        assert mode in ['linear', 'exponential'], "Mode must be 'linear' or 'exponential'."

        self.initial_beta = initial_beta
        self.final_beta = final_beta
        self.total_steps = total_steps
        self.mode = mode
        self.current_step = 0

    def step(self):
        """
        Update the current step and return the new value of beta.
        """
        if self.current_step < self.total_steps:
            if self.mode == 'linear':
                delta_beta = (self.final_beta - self.initial_beta) / self.total_steps
                beta = self.initial_beta + self.current_step * delta_beta
            elif self.mode == 'exponential':
                beta = self.initial_beta + (self.final_beta - self.initial_beta) * \
                       (1 - (self.total_steps - self.current_step) / self.total_steps) ** 3

            self.current_step += 1
            return beta
        else:
            return self.final_beta

    def reset(self):
        """
        Resets the current step to zero.
        """
        self.current_step = 0


# @ray.remote(num_gpus=0.8, num_cpus=1)
class Learner(object):
    def __init__(self,
                 net: 'DDQN',
                 cfg: dict,
                 shared_state: ray.actor.ActorClass,
                 replay_buffer: ray.actor.ActorClass,
                 log_path=None,
                 save_path=None):
        """
        [Important] Learner is assumed to run in the main thread. (Pytorch has some strange bug)

        :param args: Args have following parameters:
                    - lr: learning rate
                    - device: cpu or gpu
        :param replay_buffer: Shared replay buffer
        """
        # env action space dim
        self.shared_state = shared_state
        self.replay_buffer = replay_buffer
        # cfg
        self.lr = cfg['lr']
        self.gamma = cfg['gamma']
        self.tau = cfg['tau']
        self.target_update_freq = cfg['update_freq']
        self.batch_sz = cfg['batch_sz']
        self.save_period = cfg['save_period']
        self.device = cfg['device']
        self.n_step = cfg['n_step']
        self.save_checkpnt_path = fu.Path(save_path).joinpath('model.chkpt') if save_path is not None else None
        self.log_path = fu.Path(log_path) if log_path is not None else None
        # deep learning
        ## nn
        self.dqn = net
        self.dqn.to(self.device)
        self.dqn_target = copy.deepcopy(self.dqn)
        self.dqn.train()
        self.dqn_target.eval()
        # intrinsic curiosity module
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.lr, weight_decay=1e-6)
        self.beta_scheduler = BetaScheduler(initial_beta=cfg['beta_init'],
                                            final_beta=1,
                                            total_steps=cfg['beta_decay_step'],
                                            mode='linear')
        self.beta = cfg['beta_init']
        self.lr_dcay = cfg['lr_dcay']

    def _compute_loss(self, samples):
        device = self.device  # for shortening the following lines
        gamma = self.gamma
        state_hm = torch.as_tensor(samples['states_hm'].copy(), dtype=torch.float32, device=device).unsqueeze(1)
        next_state_hm = torch.as_tensor(samples['next_states_hm'].copy(), dtype=torch.float32, device=device).unsqueeze(
            1)
        state_dp = torch.as_tensor(samples['states_dp'].copy(), dtype=torch.float32, device=device)
        next_state_dp = torch.as_tensor(samples['next_states_dp'].copy(), dtype=torch.float32, device=device)
        state_manifest = torch.as_tensor(samples['states_manifest'].copy(), dtype=torch.float32, device=device)
        next_state_manifest = torch.as_tensor(samples['next_states_manifest'].copy(), dtype=torch.float32,
                                              device=device)
        action = torch.as_tensor(samples["action"].copy().reshape(-1, 1), dtype=torch.int64, device=device)
        reward = torch.as_tensor(samples["reward"].copy().reshape(-1, 1), dtype=torch.float32, device=device)
        done = torch.as_tensor(samples["done"].copy().reshape(-1, 1), dtype=torch.float32, device=device)
        state_feasible_action_set = torch.as_tensor(samples['state_feasible_action'].copy(),
                                                    dtype=torch.int64, device=device)
        state_feasible_action_mask = feasible_action_to_mask(state_feasible_action_set, self.dqn.num_actions)
        next_state_feasible_action_set = torch.as_tensor(samples['next_state_feasible_action'].copy(),
                                                         dtype=torch.int64, device=device)
        next_state_feasible_action_mask = feasible_action_to_mask(next_state_feasible_action_set,
                                                                  self.dqn.num_actions)
        # Q net, Q' target net, s current state, s' next state
        # double DQN Q(s, a) = r + y * Q'(s', argmax_a Q(s', a))
        curr_q_table_online = self.dqn(height_map=state_hm,
                                       dp_xy=state_dp[..., :2],
                                       z_start=state_dp[..., [2]],
                                       manifest=state_manifest,
                                       action_mask=state_feasible_action_mask
                                       )
        next_q_table_online = self.dqn(height_map=next_state_hm,
                                       dp_xy=next_state_dp[..., :2],
                                       z_start=next_state_dp[..., [2]],
                                       manifest=next_state_manifest,
                                       action_mask=next_state_feasible_action_mask,
                                       ).detach()
        curr_q_value_online = curr_q_table_online.gather(1, action)
        argmax_next_state_feasible_action = next_state_feasible_action_set.gather(1, next_q_table_online
                                                                                  .gather(1,
                                                                                          next_state_feasible_action_set)
                                                                                  .argmax(dim=1, keepdim=True))
        next_q_value_target = self.dqn_target(height_map=next_state_hm,
                                              dp_xy=next_state_dp[..., :2],
                                              z_start=next_state_dp[..., [2]],
                                              manifest=next_state_manifest,
                                              action_mask=next_state_feasible_action_mask,
                                              ).detach().gather(  # Double DQN
            1, argmax_next_state_feasible_action)
        mask = 1 - done
        # target = value_rescale((reward + gamma * inverse_value_rescale(next_q_value) * mask).to(self.device))
        # target = value_rescale((reward + (gamma ** self.n_step) * inverse_value_rescale(next_q_value) * mask)).to(
        #     self.device)
        target = (reward + gamma * next_q_value_target * mask).to(self.device)
        # target = value_rescale((reward + gamma * inverse_value_rescale(next_q_value_target) * mask)).to(self.device)
        # calculate element-wise dqn loss
        elementwise_loss = F.smooth_l1_loss(curr_q_value_online, target, reduction="none")
        return elementwise_loss

    def update_model(self):
        """Update the model by gradient descent."""
        samples: SampleBatch = ray.get(self.replay_buffer.sample.remote(self.batch_sz, self.beta))
        weights = torch.tensor(samples['weights'].reshape(-1, 1).copy(), dtype=torch.float32, device=self.device)
        indices = samples['batch_indexes']
        # update loss
        self.optimizer.zero_grad()
        # elementwise_loss = self._compute_loss2(samples)
        # loss = torch.mean(elementwise_loss * weights) / 2
        elementwise_loss = self._compute_loss(samples)
        # print(elementwise_loss_c.shape, weights.shape)
        # loss = torch.mean((elementwise_loss + elementwise_loss_c) * weights)
        loss = torch.mean(elementwise_loss * weights)
        # torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 2)
        loss.backward()
        self.optimizer.step()
        # update priorities
        loss_for_prior = elementwise_loss.detach()
        new_priorities = loss_for_prior.flatten() + prior_eps
        self.replay_buffer.update_priorities.remote(indices, new_priorities.cpu().numpy())
        return loss.item()

    def sync_dqn_train_steps(self, epoch):
        state_dict = {k: v.cpu() for k, v in self.dqn.state_dict().items()}
        self.shared_state.set_info.remote('weights', state_dict)
        # if self.training_level != training_level:
        #     self.training_level = training_level
        #     self.beta_scheduler.reset()
        #     # When you decide to change the learning rate
        #     self.lr = self.lr_dcay * self.lr
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = self.lr

    def start(self, num_epoch=1000000000, min_replay_sz=8000, output_interval=50):
        # wait for the replay buffer have enough samples
        while ray.get(self.replay_buffer.__len__.remote()) <= min_replay_sz:
            time.sleep(1)
        print("Finish Waiting. Start training...")
        if self.log_path is not None and isinstance(self.log_path, fu.Path):
            writer = CsvWriter(str(self.log_path.joinpath(f'train_log.csv')))
        # start training
        loss_total = 0
        rnd_loss_total = 0
        loss_interval_counter = 0
        for epoch in range(1, num_epoch + 1):
            st = time.time()
            loss = self.update_model()
            self.beta = self.beta_scheduler.step()
            # auto_garbage_collect()
            c_time = time.time() - st
            loss_total += loss
            loss_interval_counter += 1
            # update target network
            if epoch % self.target_update_freq == 0:
                # self.dqn_target.load_state_dict(self.dqn.state_dict())
                for param, target_param in zip(self.dqn.parameters(), self.dqn_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                    # TODO update learning rate
            # if epoch > 3000:  # warmup
            #     self.scheduler.step(loss)
            if epoch % self.save_period == 0:
                # Update shared network
                self.sync_dqn_train_steps(epoch)
                # Save checkpoint
                if self.save_checkpnt_path is not None:
                    torch.save(get_state_to_save(
                        network=self.dqn,
                        target_network=self.dqn_target,
                        optimizer=self.optimizer,
                        train_steps=epoch,
                    ), self.save_checkpnt_path)
                    print(f"Save checkpoint to {self.save_checkpnt_path} at epoch {epoch}")
            if epoch % output_interval == 0:
                # logging
                if self.log_path is not None and isinstance(self.log_path, fu.Path):
                    write_to_csv(writer, [
                        ('timestamp', get_time_stamp(), '%1s'),
                        ('lr', self.lr, '%1s'),
                        ('time_consumption', c_time, '%1s'),
                        ('learner_train_loss', loss, '%1f'),
                        ('learner_learning_rate', self.optimizer.param_groups[0]['lr'], '%1f'),
                        ('epoch', epoch, '%1d'),
                        ('loss', (loss_total) / loss_interval_counter, '%1f'),
                        ('beat', self.beta, '%1f')
                    ])
                    print(
                        f"[Learner] {get_time_stamp()} |"
                        f" epoch: {epoch} |"
                        f" avg_loss: {(loss_total) / loss_interval_counter:.7f} |"
                        f" time: {c_time:.4f}s |"
                        f" lr: {self.optimizer.param_groups[0]['lr']:.6f} |"
                        f" beta: {self.beta:.4f}"
                    )
                    loss_total = 0
                    loss_interval_counter = 0


if __name__ == '__main__':
    pass
