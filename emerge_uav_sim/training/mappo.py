from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from emerge_uav_sim.training.networks import ActorCritic
from emerge_uav_sim.training.buffer import RolloutBuffer

# Default hyperparameters
_DEFAULTS = {
    "n_steps": 512,
    "n_epochs": 4,
    "batch_size": 128,
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "hidden_dim": 128,
}


class MAPPOTrainer:
    """
    IPPO trainer with shared parameters.

    All agents share one ActorCritic; gradients are pooled across agents.
    The critic sees only each agent's own observation (no centralisation).
    """

    def __init__(self, env, cfg: Optional[Dict] = None):
        self.env = env
        hp = {**_DEFAULTS, **(cfg or {})}
        self._hp = hp

        self.n_steps: int          = hp["n_steps"]
        self.normalize_rewards: bool = hp.get("normalize_rewards", True)
        self.n_epochs: int = hp["n_epochs"]
        self.batch_size: int = hp["batch_size"]
        self.lr: float = hp["lr"]
        self.gamma: float = hp["gamma"]
        self.gae_lambda: float = hp["gae_lambda"]
        self.clip_eps: float = hp["clip_eps"]
        self.entropy_coef: float = hp["entropy_coef"]
        self.value_coef: float = hp["value_coef"]
        self.max_grad_norm: float = hp["max_grad_norm"]
        self.hidden_dim: int = hp["hidden_dim"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Infer dimensions from env
        sample_agent = env.possible_agents[0]
        self.obs_dim: int = env.observation_space(sample_agent).shape[0]
        self.act_dim: int = env.action_space(sample_agent).shape[0]
        self.n_agents: int = len(env.possible_agents)

        self.network = ActorCritic(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_dim=self.hidden_dim,
            log_std_init=hp.get("log_std_init", -1.0),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

        self.buffer = RolloutBuffer(
            self.n_steps, self.n_agents, self.obs_dim, self.act_dim
        )

        self.total_steps: int = 0
        self.update_count: int = 0
        self._best_reward: float = float("-inf")

        # Rolling state across rollouts
        self._current_obs: Optional[Dict] = None
        self._ep_reward_buf = np.zeros(self.n_agents, dtype=np.float32)

        # Running reward normalizer (Welford online mean/variance).
        # Dividing by a running std is more stable than per-rollout normalization
        # because a rollout where all agents die has near-zero reward variance,
        # making per-rollout std ≈ 0 and normalization unreliable.
        self._rew_mean: float = 0.0
        self._rew_var: float  = 1.0
        self._rew_count: int  = 0

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(self) -> float:
        """
        Run n_steps environment steps, store transitions in the buffer.
        Returns mean episode reward (NaN if no episode completed).
        """
        if self._current_obs is None:
            self._current_obs, _ = self.env.reset()
            self._ep_reward_buf[:] = 0.0

        ep_rewards: list = []
        agent_idx = self.env.agent_name_mapping  # {name: int}

        for step in range(self.n_steps):
            # Build per-agent obs array and alive mask
            obs_arr = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
            alive_mask = np.zeros(self.n_agents, dtype=np.float32)
            for agent, obs in self._current_obs.items():
                i = agent_idx[agent]
                obs_arr[i] = obs
                alive_mask[i] = 1.0

            # Forward pass
            with torch.no_grad():
                obs_t = torch.tensor(obs_arr, device=self.device)
                actions_t, log_probs_t, values_t = self.network.act(obs_t)

            actions_np = actions_t.cpu().numpy()
            log_probs_np = log_probs_t.cpu().numpy()
            values_np = values_t.cpu().numpy()

            # Build action dict for active agents
            actions_dict = {
                agent: actions_np[agent_idx[agent]]
                for agent in self.env.agents
            }

            next_obs_dict, rewards_dict, terminations, truncations, _ = (
                self.env.step(actions_dict)
            )

            # Store transitions
            for i in range(self.n_agents):
                name = self.env.possible_agents[i]
                reward = float(rewards_dict.get(name, 0.0))
                self._ep_reward_buf[i] += reward
                self.buffer.store(
                    step=step,
                    agent_idx=i,
                    obs=obs_arr[i],
                    action=actions_np[i],
                    log_prob=log_probs_np[i],
                    reward=reward,
                    value=values_np[i],
                    mask=alive_mask[i],
                )

            done = len(self.env.agents) == 0
            self.buffer.set_done(step, done)
            self.total_steps += 1

            if done:
                ep_rewards.append(float(self._ep_reward_buf.mean()))
                self._ep_reward_buf[:] = 0.0
                self._current_obs, _ = self.env.reset()
            else:
                self._current_obs = next_obs_dict

        # Bootstrap last values
        last_obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        for agent, obs in self._current_obs.items():
            last_obs[agent_idx[agent]] = obs
        with torch.no_grad():
            last_values = (
                self.network.get_value(
                    torch.tensor(last_obs, device=self.device)
                )
                .cpu()
                .numpy()
            )

        # Update running reward statistics (Welford) and normalize the buffer
        # rewards using the running std instead of per-rollout std.
        if self.normalize_rewards:
            active_rewards = self.buffer.rewards[self.buffer.masks > 0]
            for r in active_rewards:
                self._rew_count += 1
                delta = r - self._rew_mean
                self._rew_mean += delta / self._rew_count
                self._rew_var += delta * (r - self._rew_mean)
            running_std = float(np.sqrt(self._rew_var / max(self._rew_count - 1, 1)) + 1e-8)
            self.buffer.rewards = self.buffer.rewards / running_std

        self.buffer.compute_returns(last_values, self.gamma, self.gae_lambda,
                                    normalize_rewards=False)

        return float(np.mean(ep_rewards)) if ep_rewards else float("nan")

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        """Run n_epochs PPO epochs over the buffer. Returns mean losses."""
        totals = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "total": 0.0}
        n_updates = 0

        for _ in range(self.n_epochs):
            for obs, actions, old_log_probs, returns, advantages in (
                self.buffer.get_batches(self.batch_size, device=self.device)
            ):
                new_log_probs, values, entropy = self.network.evaluate(obs, actions)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, returns)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                totals["policy"] += policy_loss.item()
                totals["value"] += value_loss.item()
                totals["entropy"] += entropy.mean().item()
                totals["total"] += loss.item()
                n_updates += 1

        self.buffer.reset()
        self.update_count += 1

        if n_updates > 0:
            for k in totals:
                totals[k] /= n_updates
        return totals

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        total_steps: int,
        save_dir: Optional[str] = None,
        save_every: int = 50,
        show_training_panel: bool = True,
    ):
        """Train for total_steps *additional* env steps beyond the current checkpoint."""
        target_steps = self.total_steps + total_steps
        if self.total_steps > 0:
            print(f"Resuming from step {self.total_steps:,} — "
                  f"training for {total_steps:,} more steps (target: {target_steps:,})")

        header = (
            f"{'Update':>8} | {'Env Steps':>10} | {'EP Reward':>10} | "
            f"{'Policy':>10} | {'Value':>10} | {'Entropy':>9}"
        )
        print(header)
        print("-" * len(header))

        panel = None
        if show_training_panel:
            try:
                from emerge_uav_sim.training.training_panel import TrainingPanel
                panel = TrainingPanel(total_steps=target_steps)
            except Exception:
                pass

        while self.total_steps < target_steps:
            mean_reward = self.collect_rollout()
            losses = self.update()

            reward_str = f"{mean_reward:>10.3f}" if not np.isnan(mean_reward) else f"{'---':>10}"
            print(
                f"{self.update_count:>8} | {self.total_steps:>10} | {reward_str} | "
                f"{losses['policy']:>10.4f} | {losses['value']:>10.4f} | {losses['entropy']:>9.4f}"
            )

            if panel is not None:
                panel.update(self.update_count, self.total_steps, mean_reward, losses)

            if save_dir and self.update_count % save_every == 0:
                os.makedirs(save_dir, exist_ok=True)
                path = os.path.join(save_dir, f"ckpt_{self.update_count:06d}.pt")
                self.save(path)
                print(f"  [saved {path}]")

            if save_dir and not np.isnan(mean_reward) and mean_reward > self._best_reward:
                self._best_reward = mean_reward
                os.makedirs(save_dir, exist_ok=True)
                self.save(os.path.join(save_dir, "best.pt"))
                print(f"  [best model updated — reward {mean_reward:.3f}]")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.save(os.path.join(save_dir, "final.pt"))
            print(f"  [saved {os.path.join(save_dir, 'final.pt')}]")

        if panel is not None:
            panel.close()

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "update_count": self.update_count,
                "total_steps": self.total_steps,
                "best_reward": self._best_reward,
                "rew_mean": self._rew_mean,
                "rew_var": self._rew_var,
                "rew_count": self._rew_count,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.update_count = ckpt.get("update_count", 0)
        self.total_steps = ckpt.get("total_steps", 0)
        self._best_reward = ckpt.get("best_reward", float("-inf"))
        self._rew_mean  = ckpt.get("rew_mean",  0.0)
        self._rew_var   = ckpt.get("rew_var",   1.0)
        self._rew_count = ckpt.get("rew_count", 0)
        # Re-apply log_std_init so --log-std-init overrides the checkpoint value.
        # Also clear Adam's momentum for log_std, otherwise the stored gradient
        # history immediately pulls it back toward the old checkpoint value.
        log_std_init = self._hp.get("log_std_init", -1.0)
        with torch.no_grad():
            self.network.log_std.fill_(log_std_init)
        if self.network.log_std in self.optimizer.state:
            del self.optimizer.state[self.network.log_std]
