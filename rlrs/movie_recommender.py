"""
Modeling

TODO:
 - Move training code to other place. 
"""

from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from path import Path

from rlrs.embeddings.base import DummyEmbedding
from rlrs.envs.offline_env import OfflineEnv
from rlrs.nets.actor import Actor
from rlrs.nets.critic import Critic
from rlrs.nets.state_module import DRRAve
from rlrs.replay.replay_buffer import PriorityExperienceReplay


class MovieRecommender:
    def __init__(self, env: OfflineEnv, cfg: DictConfig):
        self.dim = cfg["dim"]
        self.state_size = cfg["state_size"]
        self.actor_hidden_dim = cfg["actor_hidden_dim"]
        self.actor_lr = cfg["actor_lr"]
        self.actor_step_size = cfg["step_size"]

        self.critic_action_emb_dim = cfg["critic_action_emb_dim"]
        self.critic_hidden_dim = cfg["critic_hidden_dim"]
        self.critic_step_size = cfg["critic_step_size"]
        self.critic_lr = cfg["critic_lr"]

        self.discount_factor = cfg["discount_factor"]
        self.tau = cfg["tau"]

        self.replay_memory_size = cfg["replay_memory_size"]
        self.batch_size = cfg["batch_size"]

        self.eps_priority = cfg["eps_priority"]
        self.eps = cfg["eps"]
        self.eps_decay = cfg["eps_decay"]
        self.std = cfg["std"]

        # -- Other training setting
        self.M = cfg["M"]
        self.workspace = Path(cfg["workspace"])

        # -- Setup data
        self.users = self.env.users()
        self.items = self.env.items()
        self.num_users = self.env.num_users()
        self.num_items = self.env.num_items()

        self.user_embeddings = DummyEmbedding(self.dim)
        self.item_embeddings = DummyEmbedding(self.dim)

        # Extract the embeddings
        self.user_embeddings.fit(self.users)
        self.item_embeddings.fit(self.items)

        # -- Setup components

        self.env = env

        self.drr_ave = DRRAve(self.dim)

        self.drr_output_dim = self.drr_ave.output_dim

        # input of actor is the state module output
        # output of it is the embedding dim
        # Fig 3 in the paper
        self.actor = Actor(
            input_dim=self.drr_output_dim,
            hidden_dim=self.actor_hidden_dim,
            output_dim=self.dim,
            state_size=self.state_size,
            tau=self.tau,
            lr=self.actor_lr,
            step_size=self.actor_step_size,
        )

        self.critic = Critic(
            input_action_dim=self.dim,
            input_state_dim=self.drr_output_dim,
            embedding_dim=self.critic_action_emb_dim,
            hidden_dim=self.critic_hidden_dim,
            tau=self.tau,
            step_size=self.critic_step_size,
            lr=self.critic_lr,
        )

    def recommend(self, user_idx, action=None):
        """
        Get *action* from the actor, multiply with the historical positive embeddings.
        Recommended items are those with the highest scores.
        """
        if action is None:
            # 1. Get embeddings of historical positive embs
            positive_item_indexes = self.env.get_positive_items(user_idx)
            positive_embs = self.item_embeddings[positive_item_indexes]
            user_emb = self.user_embeddings[user_idx]

            # 2. Get state
            state = self.drr_ave((user_emb, positive_embs))
            # feed to Actor to generate the action scores
            action = self.actor(state)

        # TODO: make it more flexible
        # Nx D * Dx1
        scores = torch.mm(self.item_embeddings.all, action)
        return torch.argsort(scores, descending=True)

    def train_on_episode(self):
        episode_reward = 0
        # 3. (Line 5) observe the initial state
        user_id, prev_items, done = self.env.reset()
        while not done:  # Line 6
            user_emb = self.user_embeddings[user_id]
            items_emb = self.item_embeddings[prev_items]

            # Line 7: Find the state via DRR
            user_emb = torch.from_numpy(user_emb)
            items_emb = torch.from_numpy(items_emb)
            drr_inputs = (user_emb, items_emb)
            state = self.drr_ave(drr_inputs)

            # Line 8: Find the action based on the curernt policy
            action = self.actor(state)
            action = action.numpy()
            # and apply epsilon-greedy exploration
            if self.eps > np.random.uniform():
                self.eps -= self.eps_decay
                action += np.random.normal(0, self.std, size=action.shape)

            # Line 9: Recommend the new item
            recommended_item = self.recommend(user_id, action)

            # Line 10: Calculate the reward and the next state
            next_user_state = self.env.step(recommended_item)
            reward = next_user_state.reward

            # Line 11 Get representation of the next state
            next_item_embs = self.item_embeddings[next_user_state.prev_pos_items]
            next_item_embs = torch.from_numpy(next_item_embs)
            next_state_inputs = (user_emb, next_item_embs)
            next_state = self.drr_ave(next_state_inputs)

            # Line 12: Update the buffer
            self.buffer.append(state, action, reward, next_state, done)

            # Line 13: Sample a minibatch of N transitions
            # with **prioritized experience replay sampling**
            # Paper: https://arxiv.org/pdf/1511.05952.pdf)
            # Algorithm 1
            if not self.buffer.empty():
                payload = self.buffer.sample(self.batch_size)

                Q = calc_Q(
                    self.critic,
                    payload.actions,
                    payload.next_states,
                    is_target=False,
                )

                Q_target = calc_Q(
                    self.critic,
                    payload.actions,
                    payload.next_states,
                    is_target=True,
                )

                # Clipped Double Q-learn
                Q_min = torch.min(torch.hstack((Q, Q_target)), dim=0)[0]

                TD_err = calc_TD_error(
                    payload.rewards,
                    Q_min,
                    payload.dones,
                    self.discount_factor,
                )

                # Update the buffer
                for p, i in zip(TD_err, payload.indexes):
                    self.buffer.update_priority(abs(p) + self.eps_priority, i)

                # train critic
                critic_inputs = (payload.actions, payload.states)
                self.critic.fit(critic_inputs, TD_err, payload.weights)

                state_grads = self.critic.dq_da(critic_inputs)
                # train actor
                self.actor.fit(payload.states, state_grads)
                # soft update strategy
                self.critic.update_target()
                self.actor.update_target()

            # move to the next state
            prev_items = next_user_state.prev_pos_items
            episode_reward += next_user_state.reward
        return episode_reward

    def train(self):
        # 1. Initialize networks
        # Line 1-2
        self.actor.initialize()
        self.critic.initialize()
        # 2. Initialize Replay Buffer
        self.buffer = PriorityExperienceReplay(self.replay_memory_size, self.dim)

        for episode in range(self.M):
            episode_reward = self.train_on_episode()
            print(f"Episode reward: {episode_reward}")

            # saving model
            self.save()

    def _get_checkpoint_sub(self, subdir: Optional[Path] = None):
        path = self.workspace
        if subdir is not None:
            path = path / subdir
        return path

    def get_actor_checkpoint(self, subdir: Optional[Path] = None):
        actor_path = self._get_checkpoint_sub(subdir) / "actor.pth"
        return actor_path

    def get_critic_checkpoint(self, subdir: Optional[Path] = None):
        critic_path = self._get_checkpoint_sub(subdir) / "critic.pth"
        return critic_path

    def save(self, subdir: Optional[Path] = None):
        actor_checkpoint = self.get_actor_checkpoint(subdir)
        self.actor.save(actor_checkpoint)
        critic_checkpoint = self.get_critic_checkpoint(subdir)
        self.critic.save(critic_checkpoint)

    def load(self, subdir: Optional[Path] = None):
        actor_checkpoint = self.get_actor_checkpoint(subdir)
        self.actor.load(actor_checkpoint)
        critic_checkpoint = self.get_critic_checkpoint(subdir)
        self.critic.load(critic_checkpoint)


# Line 11 In PER paper
def calc_TD_error(reward, Q, dones, discount_factor):
    Q_target = np.zeros_like(Q)
    Q_target = reward + (1.0 - dones) * (discount_factor * Q)
    return Q_target


def calc_Q(network: Critic, action, state, is_target=False):
    inputs = (action, state)
    if is_target:
        return network.target_forward(inputs)
    return network.forward(inputs)
