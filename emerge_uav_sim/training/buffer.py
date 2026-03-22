from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import torch


class RolloutBuffer:
    """
    Fixed-length rollout buffer for IPPO with multiple agents.

    Shapes:
        obs       : (n_steps, n_agents, obs_dim)
        actions   : (n_steps, n_agents, act_dim)
        log_probs : (n_steps, n_agents)
        rewards   : (n_steps, n_agents)
        values    : (n_steps, n_agents)
        masks     : (n_steps, n_agents)  — 1 if agent alive, 0 otherwise
        dones     : (n_steps,)           — 1 if episode ended after this step
    """

    def __init__(self, n_steps: int, n_agents: int, obs_dim: int, act_dim: int):
        self.n_steps = n_steps
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.reset()

    def reset(self):
        n, a = self.n_steps, self.n_agents
        self.obs = np.zeros((n, a, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((n, a, self.act_dim), dtype=np.float32)
        self.log_probs = np.zeros((n, a), dtype=np.float32)
        self.rewards = np.zeros((n, a), dtype=np.float32)
        self.values = np.zeros((n, a), dtype=np.float32)
        self.masks = np.zeros((n, a), dtype=np.float32)
        self.dones = np.zeros(n, dtype=np.float32)
        self.advantages = np.zeros((n, a), dtype=np.float32)
        self.returns = np.zeros((n, a), dtype=np.float32)

    def store(
        self,
        step: int,
        agent_idx: int,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        mask: float,
    ):
        self.obs[step, agent_idx] = obs
        self.actions[step, agent_idx] = action
        self.log_probs[step, agent_idx] = log_prob
        self.rewards[step, agent_idx] = reward
        self.values[step, agent_idx] = value
        self.masks[step, agent_idx] = mask

    def set_done(self, step: int, done: bool):
        self.dones[step] = float(done)

    def compute_returns(
        self,
        last_values: np.ndarray,
        gamma: float,
        gae_lambda: float,
        normalize_rewards: bool = True,
    ):
        """
        Compute GAE advantages and discounted returns in-place.

        last_values       : (n_agents,) bootstrap values after the final step.
        normalize_rewards : divide rewards by their std across the rollout before
                            GAE computation — stabilises training when reward
                            magnitudes are small or inconsistent.
        """
        if normalize_rewards:
            active = self.masks > 0
            if active.any():
                rew_std = self.rewards[active].std() + 1e-8
                self.rewards = self.rewards / rew_std

        gae = np.zeros(self.n_agents, dtype=np.float32)

        for t in reversed(range(self.n_steps)):
            non_terminal = 1.0 - self.dones[t]

            if t == self.n_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]

            agent_mask = self.masks[t]  # (n_agents,)

            delta = (
                self.rewards[t]
                + gamma * non_terminal * next_values * agent_mask
                - self.values[t] * agent_mask
            )
            gae = delta + gamma * gae_lambda * non_terminal * gae * agent_mask
            self.advantages[t] = gae * agent_mask
            self.returns[t] = (self.advantages[t] + self.values[t]) * agent_mask

    def get_batches(
        self,
        batch_size: int,
        device: str = "cpu",
    ) -> Iterator[Tuple[torch.Tensor, ...]]:
        """
        Yield (obs, actions, log_probs, returns, advantages) tensor mini-batches.
        Only active (mask=1) agent-step pairs are included.
        Advantages are normalized over the active set.
        """
        n_total = self.n_steps * self.n_agents

        obs_flat = self.obs.reshape(n_total, self.obs_dim)
        actions_flat = self.actions.reshape(n_total, self.act_dim)
        log_probs_flat = self.log_probs.reshape(n_total)
        returns_flat = self.returns.reshape(n_total)
        advantages_flat = self.advantages.reshape(n_total)
        masks_flat = self.masks.reshape(n_total)

        active = masks_flat > 0
        if active.sum() == 0:
            return

        obs_a = obs_flat[active]
        actions_a = actions_flat[active]
        log_probs_a = log_probs_flat[active]
        returns_a = returns_flat[active]
        advantages_a = advantages_flat[active]

        # Normalize advantages over the active set
        adv_mean = advantages_a.mean()
        adv_std = advantages_a.std() + 1e-8
        advantages_a = (advantages_a - adv_mean) / adv_std

        n_active = len(obs_a)
        indices = np.random.permutation(n_active)

        for start in range(0, n_active, batch_size):
            idx = indices[start : start + batch_size]
            yield (
                torch.tensor(obs_a[idx], device=device),
                torch.tensor(actions_a[idx], device=device),
                torch.tensor(log_probs_a[idx], device=device),
                torch.tensor(returns_a[idx], device=device),
                torch.tensor(advantages_a[idx], device=device),
            )
