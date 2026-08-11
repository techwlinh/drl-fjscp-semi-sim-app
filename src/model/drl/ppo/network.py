from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """
    Actor-Critic Neural Network for PPO with Action Masking support.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Shared/Actor Trunk
        self.actor_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, action_dim),
        )

        # Critic Trunk
        self.critic_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def get_action_distribution(
        self, obs: torch.Tensor, action_mask: torch.Tensor = None
    ) -> Categorical:
        logits = self.actor_net(obs)

        if action_mask is not None:
            # Mask invalid actions with large negative value (-1e9)
            HUGE_NEG = -1e9
            logits = torch.where(
                action_mask > 0.5, logits, torch.tensor(HUGE_NEG, device=logits.device)
            )

        return Categorical(logits=logits)

    def forward(
        self, obs: torch.Tensor, action_mask: torch.Tensor = None
    ) -> Tuple[Categorical, torch.Tensor]:
        dist = self.get_action_distribution(obs, action_mask)
        value = self.critic_net(obs).squeeze(-1)
        return dist, value

    def act(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(obs, action_mask)

        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(
        self, obs: torch.Tensor, action_mask: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, values = self.forward(obs, action_mask)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy
