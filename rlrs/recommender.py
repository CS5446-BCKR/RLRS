"""
Modeling

TODO:
 - Move training code to other place.
"""

from datetime import datetime
from typing import Optional

import mlflow
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from path import Path

from rlrs.embeddings import embedding_factory
from rlrs.envs.offline_env import MovieLenEnv
from rlrs.nets.actor import Actor
from rlrs.nets.critic import Critic, calc_Q_min, calc_TD_error
from rlrs.nets.state_module import DRRAve
from rlrs.replay.replay_buffer import PriorityExperienceReplay


class Recommender:
    def __init__(self, env: MovieLenEnv, cfg: DictConfig):
        self.env = env
        self.cfg = cfg
        self.topk = cfg["topk"]
        self.save_interval = cfg.get("save_interval", 1)
        self.mlflow_port = cfg.get("mlflow_port", 8080)
        user_emb_cfg = cfg["user_embedding"]
        item_emb_cfg = cfg["item_embedding"]
        assert user_emb_cfg["dim"] == item_emb_cfg["dim"]
        self.dim = item_emb_cfg["dim"]
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

        self.user_embeddings = embedding_factory(cfg["user_embedding"])
        self.item_embeddings = embedding_factory(cfg["item_embedding"])

        # Extract the embeddings
        self.user_embeddings.fit(self.users)
        self.item_embeddings.fit(self.items)

        # -- Setup components

        # DL networks
        self.drr_ave = None
        self.actor = None
        self.critic = None

        self._init_networks()
        assert all(net for net in [self.drr_ave, self.actor, self.critic])

        # vars for recommending
        self.user_id = None
        self.prev_items = None
        self.done = True
        self.user_emb = None

    def _init_networks(self):
        """
        If there is a checkpoint path, init from checkpoint
        """
        self.drr_ave = DRRAve.from_config(self.cfg["drr"])
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

        self.cfg["critic"].update(
            {
                "input_action_dim": self.dim,
                "input_state_dim": self.drr_output_dim,
                "tau": self.tau,
            }
        )

        self.actor = Actor.from_config(self.cfg["actor"])
        self.critic = Critic.from_config(self.cfg["critic"])
        if "eval" in self.cfg and self.cfg["eval"]:
            self.load()

    def recommend_from_action(self, action):
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

    def set_users(self, users_):
        self.env.set_users(users_)
        self.users = self.env.users
        self.num_users = self.env.num_users
        self.user_embeddings.fit(self.users)

    def reset(self):

        self.drr_ave.eval()
        self.critic.eval()
        self.actor.eval()

        init_state = self.env.reset()
        self.user_id = init_state.user_id
        self.done = init_state.done
        self.prev_items = init_state.prev_pos_items
        self.user_emb = self.user_embeddings[self.user_id]

    def recommend(self):
        if self.done:
            return None
        items_emb = self.item_embeddings[self.prev_items]
        state = self.drr_ave((self.user_emb, items_emb))
        action = self.actor(state).detach()
        recommended_items = self.recommend_from_action(action)
        return recommended_items

    def feedback(self, recs, positives):
        next_state = self.env.step(recs, positives)
        self.done = next_state.done
        self.prev_items = next_state.prev_pos_items
        return next_state.reward

    def recommend_items(self):
        """
        Parameters:
            items: a list of historical positive items
        """
        self.drr_ave.eval()
        self.critic.eval()
        self.actor.eval()
        init_state = self.env.reset()
        user_id = init_state.user_id
        prev_items = init_state.prev_pos_items
        done = init_state.done
        user_emb = self.user_embeddings[user_id]
        while not done:
            items_emb = self.item_embeddings[prev_items]
            # Line 7: Find the state via DRR
            state = self.drr_ave((user_emb, items_emb))

            # Line 8: Find the action based on the current policy
            action = self.actor(state).detach()
            # Line 9: Recommend the new item
            recommended_items = self.recommend_from_action(action)

            yield recommended_items

            # Line 10: Calculate the reward and the next state
            next_user_state = self.env.step(recommended_items)
            prev_items = next_user_state.prev_pos_items
            done = next_user_state.done

    def train_on_episode(self, iter_count):
        episode_reward = 0
        # 3. (Line 5) observe the initial state
        init_state = self.env.reset()
        user_id = init_state.user_id
        prev_items = init_state.prev_pos_items
        done = init_state.done
        logger.debug(f"Reset the env: {init_state=}")
        user_emb = self.user_embeddings[user_id]
        while not done:  # Line 6
            iter_count += 1
            logger.debug(f"Iteration: {iter_count}")
            items_emb = self.item_embeddings[prev_items]

            # Line 7: Find the state via DRR
            state = self.drr_ave((user_emb, items_emb))

            # Line 8: Find the action based on the current policy
            action = self.actor(state).detach()
            # and apply epsilon-greedy exploration
            if self.eps > np.random.uniform():
                self.eps -= self.eps_decay
                action += torch.normal(torch.zeros_like(action), self.std)

            # Line 9: Recommend the new item
            recommended_items = self.recommend_from_action(action)
            logger.debug(f"Recommended items: {recommended_items.values}")

            # Line 10: Calculate the reward and the next state
            next_user_state = self.env.step(recommended_items)
            reward = next_user_state.reward
            done = next_user_state.done
            logger.debug(f"Reward from rec items: {reward}")
            mlflow.log_metric("reward_rec_items", reward, step=iter_count)

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
                payload = self.buffer.sample(self.batch_size)

                target_next_actions = self.actor.target_forward(payload.next_states)
                critic_next_inputs = (target_next_actions.detach(), payload.next_states)

                Q_min = calc_Q_min(self.critic, critic_next_inputs)

                TD_err = calc_TD_error(
                    payload.rewards,
                    Q_min,
                    payload.dones,
                    self.discount_factor,
                ).detach()

                # Update the buffer
                for p, i in zip(TD_err, payload.indexes):
                    self.buffer.update_priority(abs(p) + self.eps_priority, i)

                # train critic
                critic_inputs = (payload.actions.detach(), payload.states.detach())
                critic_loss = self.critic.fit(critic_inputs, TD_err, payload.weights)
                mlflow.log_metric("critic_loss", critic_loss, step=iter_count)
                logger.debug(f"Online critic loss: {critic_loss}")

                state_grads = self.critic.dq_da(critic_inputs)
                # train actor
                self.actor.fit(payload.states, state_grads.detach())
                # soft update strategy
                self.critic.update_target()
                self.actor.update_target()

            # move to the next state
            prev_items = next_user_state.prev_pos_items
            episode_reward += next_user_state.reward
        logger.info(f"Episode Reward: {episode_reward}")
        mlflow.log_metric("episode_reward", episode_reward, step=iter_count)
        return iter_count

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
        mlflow.set_tracking_uri(uri=f"http://127.0.0.1:{self.mlflow_port}")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mlflow.set_experiment(f"Recommender: {now}")
        iter_count = 0
        with mlflow.start_run():
            mlflow.log_params(self.cfg)
            for episode in range(self.M):
                logger.info(f"Start episode #{episode}")
                iter_count = self.train_on_episode(iter_count)
                if (episode + 1) % self.save_interval == 0:
                    self.save(f"ep_{episode+1}")

    def _get_checkpoint_sub(self, subdir: Optional[Path] = None):
        path = self.workspace
        if subdir is not None:
            path = path / subdir
        path.makedirs_p()
        return path

    def get_actor_checkpoint(self, subdir: Optional[Path] = None):
        actor_path = self._get_checkpoint_sub(subdir) / "actor.pth"
        return actor_path

    def get_critic_checkpoint(self, subdir: Optional[Path] = None):
        critic_path = self._get_checkpoint_sub(subdir) / "critic.pth"
        return critic_path

    def get_drr_checkpoint(self, subdir: Optional[Path] = None):
        drr_path = self._get_checkpoint_sub(subdir) / "drr.pth"
        return drr_path

    def save(self, subdir: Optional[Path] = None):
        actor_checkpoint = self.get_actor_checkpoint(subdir)
        self.actor.save(actor_checkpoint)
        critic_checkpoint = self.get_critic_checkpoint(subdir)
        self.critic.save(critic_checkpoint)
        drr_checkpoint = self.get_drr_checkpoint(subdir)
        self.drr_ave.save(drr_checkpoint)

    def load(self, subdir: Optional[Path] = None):
        actor_checkpoint = self.get_actor_checkpoint(subdir)
        self.actor.load(actor_checkpoint)
        critic_checkpoint = self.get_critic_checkpoint(subdir)
        self.critic.load(critic_checkpoint)
        drr_checkpoint = self.get_drr_checkpoint(subdir)
        self.drr_ave.load(drr_checkpoint)
