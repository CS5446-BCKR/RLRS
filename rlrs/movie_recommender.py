"""
Modeling

TODO:
 - Move training code to other place.
"""

from typing import Optional

import numpy as np
import torch
from loguru import logger
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
        self.env = env
        self.cfg = cfg
        self.topk = cfg["topk"]
        self.dim = cfg["dim"]
        self.state_size = cfg["state_size"]

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
        self.workspace.makedirs_p()

        # -- Setup data
        self.users = self.env.users
        self.items = self.env.items
        self.num_users = self.env.num_users
        self.num_items = self.env.num_items

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
        self.cfg["actor"].update(
            {
                "input_dim": self.drr_output_dim,
                "output_dim": self.dim,
                "tau": self.cfg["tau"],
            }
        )
        self.actor = Actor.from_config(self.cfg["actor"])

        self.cfg["critic"].update(
            {
                "input_action_dim": self.dim,
                "input_state_dim": self.drr_output_dim,
                "tau": self.tau,
            }
        )

        self.critic = Critic.from_config(self.cfg["critic"])

    def recommend(self, action):
        """
        Get *action* from the actor, multiply with the historical positive embeddings.
        Recommended items are those with the highest scores.
        """
        # TODO: make it more flexible
        # Nx D * Dx1
        scores = torch.mm(self.item_embeddings.all, action[:, None])
        indices = torch.argsort(scores.squeeze(), descending=True).squeeze()
        indices = indices[: self.topk]
        return self.items.iloc[indices.numpy()].index

    def train_on_episode(self):
        episode_reward = 0
        # 3. (Line 5) observe the initial state
        init_state = self.env.reset()
        user_id = init_state.user_id
        prev_items = init_state.prev_pos_items
        done = init_state.done
        logger.debug(f"Reset the env: {init_state=}")
        user_emb = self.user_embeddings[user_id]
        while not done:  # Line 6
            items_emb = self.item_embeddings[prev_items]

            # Line 7: Find the state via DRR
            drr_inputs = (user_emb, items_emb)
            state = self.drr_ave(drr_inputs)

            # Line 8: Find the action based on the current policy
            action = self.actor(state)
            action = action.detach()
            # and apply epsilon-greedy exploration
            if self.eps > np.random.uniform():
                self.eps -= self.eps_decay
                action += torch.normal(torch.zeros_like(action), self.std)

            # Line 9: Recommend the new item
            recommended_items = self.recommend(action)
            logger.debug(f"Recommended items: {recommended_items}")

            # Line 10: Calculate the reward and the next state
            next_user_state = self.env.step(recommended_items)
            reward = next_user_state.reward
            done = next_user_state.done
            logger.debug(f"Reward from rec items: {reward}")

            # Line 11 Get representation of the next state
            next_item_embs = self.item_embeddings[next_user_state.prev_pos_items]
            logger.debug(f"Previous positive items: {next_user_state.prev_pos_items}")
            next_state_inputs = (user_emb, next_item_embs)
            next_state = self.drr_ave(next_state_inputs)

            # Line 12: Update the buffer
            self.buffer.append(state, action, reward, next_state, done)

            # Line 13: Sample a minibatch of N transitions
            # with **prioritized experience replay sampling**
            # Paper: https://arxiv.org/pdf/1511.05952.pdf)
            # Algorithm 1
            if not self.buffer.empty():
                logger.debug(f"Sample buffers: {self.batch_size}")
                payload = self.buffer.sample(self.batch_size)

                target_next_actions = self.actor.target_forward(payload.next_states)
                critic_next_inputs = (target_next_actions.detach(), payload.next_states)

                Q = self.critic.calcQ(critic_next_inputs, is_target=False)
                Q_target = self.critic.calcQ(critic_next_inputs, is_target=True)

                # Clipped Double Q-learn
                Q_min = torch.min(torch.hstack((Q, Q_target)), dim=-1)[0]

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
                critic_inputs = (payload.actions, payload.states.detach())
                self.critic.fit(critic_inputs, TD_err.detach(), payload.weights)

                state_grads = self.critic.dq_da(critic_inputs)
                # train actor
                self.actor.fit(payload.states, state_grads.detach())
                # soft update strategy
                self.critic.update_target()
                self.actor.update_target()

            # move to the next state
            prev_items = next_user_state.prev_pos_items
            episode_reward += next_user_state.reward
            logger.debug(f"Episode Reward: {episode_reward}")
        return episode_reward

    def train(self):
        # 1. Initialize networks
        # Line 1-2
        self.actor.initialize()
        self.critic.initialize()
        # 2. Initialize Replay Buffer
        self.buffer = PriorityExperienceReplay(
            self.replay_memory_size,
            state_dim=self.drr_output_dim,
            action_dim=self.dim,
        )

        for episode in range(self.M):
            logger.debug(f"Start episode #{episode}")
            episode_reward = self.train_on_episode()
            logger.debug(f"Episode reward: {episode_reward}")

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
    return reward + (1.0 - dones.float()) * (discount_factor * Q)


def calc_Q(network: Critic, action, state, is_target=False):
    inputs = (action, state)
    if is_target:
        return network.target_forward(inputs)
    return network.forward(inputs)
