import numpy as np


SENTIAL_REWARD_VALUE = 999


class ReplayMemory:
    def __init__(self, memory_size, embedding_dim):
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.crt_idx = 0

        # setup memory
        self.states = np.zeros(
            (memory_size, 3 * embedding_dim), dtype=np.float32)
        self.actions = np.zeros((memory_size, embedding_dim), dtype=np.float32)
        self.rewards = np.zeros((memory_size,), dtype=np.float32)
        self.next_states = np.zeros(
            (memory_size, 3 * embedding_dim), dtype=np.float32)
        self.dones = np.zeros((memory_size,), dtype=np.bool)
        self.rewards[memory_size - 1] = SENTIAL_REWARD_VALUE

    def is_full(self):
        return self.rewards[-1] != SENTIAL_REWARD_VALUE

    def append(self, state, action, reward, next_state, done):
        self.states[self.crt_idx] = state
        self.actions[self.crt_idx] = action
        self.rewards[self.crt_idx] = reward
        self.next_states[self.crt_idx] = next_state
        self.dones[self.crt_idx] = done
        self.crt_idx = (self.crt_idx + 1) % self.memory_size

    def sample(self, batch_size):
        sample_size = self.ctr_idx
        if self.is_full():
            sample_size = self.memory_size - 1
        batch_indexes = np.random.choice(sample_size, batch_size)
        return (
            self.states[batch_indexes],
            self.actions[batch_indexes],
            self.rewards[batch_indexes],
            self.next_states[batch_indexes],
            self.dones[batch_indexes]
        )
