"""
TODO: switch to pytorch tensor
"""

import random
from collections import namedtuple

import torch
from omegaconf import DictConfig

from .trees import MinTree, SumTree

BatchBufferPayload = namedtuple(
    "BatchBufferPayload",
    ["states", "actions", "rewards", "next_states", "dones", "weights", "indexes"],
)


class PriorityExperienceReplay:
    def __init__(
        self,
        buffer_size,
        state_dim,
        action_dim,
        max_priority=1.0,
        alpha=0.6,
        beta=0.4,
        beta_constant=1e-5,
    ):
        self.buffer_size = buffer_size
        self.crt_idx = 0
        self.is_full = False

        # setup buffers
        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32)
        self.rewards = torch.zeros((buffer_size,), dtype=torch.float32)
        self.next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32)
        self.dones = torch.zeros((buffer_size,), dtype=torch.bool)

        self.sum_tree = SumTree(buffer_size)
        self.min_tree = MinTree(buffer_size)

        self.max_priority = 1.0
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_constant = 1e-5

    def empty(self):
        return self.crt_idx < 1 and not self.is_full

    def append(self, state, action, reward, next_state, done):
        self.states[self.crt_idx] = state.detach()
        self.actions[self.crt_idx] = action.detach()
        self.rewards[self.crt_idx] = reward
        self.next_states[self.crt_idx] = next_state.detach()
        self.dones[self.crt_idx] = done

        self.sum_tree.add_data(self.max_priority**self.alpha)
        self.min_tree.add_data(self.max_priority**self.alpha)

        self.crt_idx = (self.crt_idx + 1) % self.buffer_size
        if self.crt_idx == 0:
            self.is_full = True

    def update_priority(self, priority, index):
        self.sum_tree.update_priority(priority**self.alpha, index)
        self.min_tree.update_priority(priority**self.alpha, index)
        self.update_max_priority(priority)

    def update_max_priority(self, priority):
        self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size) -> BatchBufferPayload:
        rd_idx = []
        weight_batch = []
        index_batch = []
        sum_priority = self.sum_tree.sum_all_priority()

        N = self.buffer_size if self.is_full else self.crt_idx
        min_priority = self.min_tree.min_priority() / sum_priority
        max_weight = (N * min_priority) ** (-self.beta)

        segment_size = sum_priority / batch_size

        for i in range(batch_size):
            min_seg = segment_size * i
            max_seg = segment_size * (i + 1)

            j = random.uniform(min_seg, max_seg)
            priority, tree_index, buffer_index = self.sum_tree.search(j)
            rd_idx.append(buffer_index)

            pi = priority / sum_priority
            wi = (pi * N) ** (-self.beta) / max_weight

            weight_batch.append(wi)
            index_batch.append(tree_index)

        self.beta = min(1.0, self.beta + self.beta_constant)

        return BatchBufferPayload(
            self.states[rd_idx],
            self.actions[rd_idx],
            self.rewards[rd_idx],
            self.next_states[rd_idx],
            self.dones[rd_idx],
            torch.Tensor(weight_batch),
            index_batch,
        )
