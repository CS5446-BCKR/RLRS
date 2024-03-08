"""
Modeling

TODO:
 - Move training code to other place. 
"""

from omegaconf import DictConfig

from rlrs.nets.actor import Actor
from rlrs.nets.critic import Critic
from rlrs.nets.state_module import DRRAve
from rlrs.replay.replay_buffer import PriorityExperienceReplay


class MovieRecommender:
    def __init__(self, env, cfg: DictConfig):
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

        self.buffer = PriorityExperienceReplay(
            self.replay_memory_size, self.dim
        )


    def recommend(self, user_idx): ...

    def train(self): ...
