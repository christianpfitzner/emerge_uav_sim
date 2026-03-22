from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """
    Shared-parameter actor-critic for IPPO.

    Backbone is shared; actor_mean and critic are separate heads.
    Action distribution: diagonal Gaussian with learned (state-independent) log_std.
    No clamping here — the environment clips actions at its own bounds.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        log_std_init: float = -1.0,
    ):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))
        self.critic = nn.Linear(hidden_dim, 1)

        # log_std is clamped to [LOG_STD_MIN, LOG_STD_MAX] at runtime.
        # The ceiling is fixed high so log_std always starts below it and
        # gradients flow freely in both directions from the first update.
        # (Pinning LOG_STD_MAX == log_std_init blocked all gradients because
        # PyTorch clamp has zero gradient exactly at the boundary.)
        self.LOG_STD_MIN  = -3.0
        self.LOG_STD_MAX  = 0.5   # was 2.0; capped to prevent runaway entropy (max std≈1.65)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def _get_dist(self, features: torch.Tensor) -> Normal:
        mean = self.actor_mean(features)
        std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX).exp().expand_as(mean)
        return Normal(mean, std)

    def act(self, obs: torch.Tensor):
        """Sample action and return (action, log_prob, value)."""
        features = self.backbone(obs)
        dist = self._get_dist(features)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.critic(features).squeeze(-1)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        """Return (log_prob, value, entropy) for given obs and action."""
        features = self.backbone(obs)
        dist = self._get_dist(features)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(features).squeeze(-1)
        return log_prob, value, entropy

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(obs)
        return self.critic(features).squeeze(-1)
