"""
Parallel rollout collection for MAPPO.

N worker subprocesses each own an independent environment instance.
Each collects n_steps transitions, computes GAE locally, then sends the
processed arrays back to the main process.  The main process concatenates
all workers' data and runs a single PPO update — giving n_workers × n_steps
experience per gradient step without blocking.

Usage:
    from emerge_uav_sim.training.parallel_trainer import ParallelMAPPOTrainer
    trainer = ParallelMAPPOTrainer(env, cfg=trainer_cfg, n_workers=4)
    trainer.train(...)          # same API as MAPPOTrainer
"""
from __future__ import annotations

import multiprocessing as mp
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from emerge_uav_sim.training.mappo import MAPPOTrainer, _DEFAULTS
from emerge_uav_sim.training.buffer import RolloutBuffer


# ---------------------------------------------------------------------------
# Worker subprocess
# ---------------------------------------------------------------------------

def _worker_fn(
    worker_id: int,
    pipe: mp.connection.Connection,
    env_cfg,                    # SimConfig (pickleable)
    n_steps: int,
    obs_dim: int,
    act_dim: int,
    hidden_dim: int,
    log_std_init: float,
    gamma: float,
    gae_lambda: float,
    normalize_rewards: bool,
    base_seed: int,
) -> None:
    """
    Persistent worker process.

    Protocol:
        main → worker  :  ('collect', state_dict)  |  ('stop', None)
        worker → main  :  ('result', data_dict)     |  ('done', None)
    """
    # Disable multi-threading in worker processes: forking after PyTorch/NumPy
    # initialises their thread pools (OpenMP/MKL) causes the child to deadlock
    # when it tries to use the inherited-but-dead thread pool.  Single-thread
    # mode avoids this entirely and is fine since workers run in parallel anyway.
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    import numpy as np
    import torch
    torch.set_num_threads(1)

    # Each worker gets a unique seed so episodes are diverse
    seed = base_seed + worker_id * 7919
    from emerge_uav_sim.envs.uav_team_env import UAVTeamEnv
    from emerge_uav_sim.training.networks import ActorCritic
    from emerge_uav_sim.training.buffer import RolloutBuffer

    env = UAVTeamEnv(config=env_cfg)
    n_agents = len(env.possible_agents)
    agent_idx = env.agent_name_mapping

    network = ActorCritic(obs_dim, act_dim, hidden_dim, log_std_init)
    network.eval()

    buf = RolloutBuffer(n_steps, n_agents, obs_dim, act_dim)

    current_obs, _ = env.reset(seed=seed)
    ep_reward_buf = np.zeros(n_agents, dtype=np.float32)

    while True:
        cmd, payload = pipe.recv()

        if cmd == 'stop':
            pipe.send(('done', None))
            break

        # --- Load latest weights ---
        network.load_state_dict(payload)
        network.eval()
        buf.reset()
        ep_rewards: List[float] = []

        # --- Collect n_steps ---
        for step in range(n_steps):
            obs_arr = np.zeros((n_agents, obs_dim), dtype=np.float32)
            alive_mask = np.zeros(n_agents, dtype=np.float32)
            for agent, obs in current_obs.items():
                i = agent_idx[agent]
                obs_arr[i] = obs
                alive_mask[i] = 1.0

            with torch.no_grad():
                actions_t, log_probs_t, values_t = network.act(
                    torch.tensor(obs_arr)
                )
            actions_np  = actions_t.numpy()
            log_probs_np = log_probs_t.numpy()
            values_np    = values_t.numpy()

            actions_dict = {
                agent: actions_np[agent_idx[agent]]
                for agent in env.agents
            }
            next_obs, rewards_dict, _, _, _ = env.step(actions_dict)

            for i in range(n_agents):
                name = env.possible_agents[i]
                r = float(rewards_dict.get(name, 0.0))
                ep_reward_buf[i] += r
                buf.store(step, i, obs_arr[i], actions_np[i],
                          log_probs_np[i], r, values_np[i], alive_mask[i])

            done = len(env.agents) == 0
            buf.set_done(step, done)

            if done:
                ep_rewards.append(float(ep_reward_buf.mean()))
                ep_reward_buf[:] = 0.0
                current_obs, _ = env.reset()
            else:
                current_obs = next_obs

        # --- Bootstrap & GAE ---
        last_obs = np.zeros((n_agents, obs_dim), dtype=np.float32)
        for agent, obs in current_obs.items():
            last_obs[agent_idx[agent]] = obs
        with torch.no_grad():
            last_values = network.get_value(
                torch.tensor(last_obs)
            ).numpy()

        buf.compute_returns(last_values, gamma, gae_lambda, normalize_rewards)

        # Send processed arrays (copies — subprocess continues with next rollout)
        pipe.send(('result', {
            'obs':        buf.obs.copy(),
            'actions':    buf.actions.copy(),
            'log_probs':  buf.log_probs.copy(),
            'advantages': buf.advantages.copy(),
            'returns':    buf.returns.copy(),
            'masks':      buf.masks.copy(),
            'ep_rewards': ep_rewards,
        }))


# ---------------------------------------------------------------------------
# Parallel trainer
# ---------------------------------------------------------------------------

class ParallelMAPPOTrainer(MAPPOTrainer):
    """
    Drop-in replacement for MAPPOTrainer that collects rollouts in parallel.

    ``n_workers`` subprocesses each own an env.  Every call to
    ``collect_rollout()`` gathers n_workers × n_steps transitions instead
    of just n_steps, giving proportionally richer gradient updates at roughly
    the same wall-clock time.
    """

    def __init__(self, env, cfg: Optional[Dict] = None, n_workers: int = 4):
        super().__init__(env, cfg)
        self.n_workers = n_workers
        self._pipes: List[mp.connection.Connection] = []
        self._procs: List[mp.Process] = []
        self._start_workers()

    # ------------------------------------------------------------------

    def _start_workers(self) -> None:
        hp = {**_DEFAULTS, **(self._hp if hasattr(self, '_hp') else {})}
        log_std_init = self.network.log_std.data[0].item()

        print(f"Starting {self.n_workers} worker processes …", flush=True)
        for i in range(self.n_workers):
            parent_conn, child_conn = mp.Pipe(duplex=True)
            p = mp.Process(
                target=_worker_fn,
                args=(
                    i, child_conn,
                    self.env.cfg,
                    self.n_steps,
                    self.obs_dim, self.act_dim,
                    self.hidden_dim, log_std_init,
                    self.gamma, self.gae_lambda,
                    self.normalize_rewards,
                    42,
                ),
                daemon=True,
            )
            p.start()
            child_conn.close()   # child end not needed in parent
            self._pipes.append(parent_conn)
            self._procs.append(p)

    # ------------------------------------------------------------------

    def collect_rollout(self) -> float:
        """
        Dispatch to all workers, wait for results, merge into one update.
        Returns mean episode reward across all workers (NaN if no completed ep).
        """
        # Weights to CPU dict (pickleable)
        state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}

        # Fire off all workers simultaneously
        for pipe in self._pipes:
            pipe.send(('collect', state_dict))

        # Gather results
        all_obs, all_act, all_lp, all_adv, all_ret, all_mask = [], [], [], [], [], []
        all_ep_rewards: List[float] = []

        for i, pipe in enumerate(self._pipes):
            if not pipe.poll(timeout=120):
                raise RuntimeError(
                    f"Worker {i} did not respond within 120 s — "
                    "it may have crashed. Check for import errors or OOM."
                )
            tag, data = pipe.recv()
            if tag != 'result':
                continue
            n, a = data['obs'].shape[:2]
            all_obs.append(data['obs'].reshape(n * a, self.obs_dim))
            all_act.append(data['actions'].reshape(n * a, self.act_dim))
            all_lp.append(data['log_probs'].reshape(n * a))
            all_adv.append(data['advantages'].reshape(n * a))
            all_ret.append(data['returns'].reshape(n * a))
            all_mask.append(data['masks'].reshape(n * a))
            all_ep_rewards.extend(data['ep_rewards'])

        # Combine across workers
        self._parallel_data = {
            'obs':        np.concatenate(all_obs),
            'actions':    np.concatenate(all_act),
            'log_probs':  np.concatenate(all_lp),
            'advantages': np.concatenate(all_adv),
            'returns':    np.concatenate(all_ret),
            'masks':      np.concatenate(all_mask),
        }

        self.total_steps += self.n_steps * self.n_workers

        return float(np.mean(all_ep_rewards)) if all_ep_rewards else float('nan')

    # ------------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        """PPO update on the combined data from all workers."""
        d = self._parallel_data
        active = d['masks'] > 0
        if not active.any():
            return {'policy': 0., 'value': 0., 'entropy': 0., 'total': 0.}

        obs_a      = d['obs'][active]
        actions_a  = d['actions'][active]
        lp_a       = d['log_probs'][active]
        returns_a  = d['returns'][active]
        adv_a      = d['advantages'][active]

        # Re-normalise advantages over the full combined set
        adv_a = (adv_a - adv_a.mean()) / (adv_a.std() + 1e-8)

        n_active = len(obs_a)
        totals = {'policy': 0., 'value': 0., 'entropy': 0., 'total': 0.}
        n_updates = 0

        import torch.nn.functional as F

        for _ in range(self.n_epochs):
            idx = np.random.permutation(n_active)
            for start in range(0, n_active, self.batch_size):
                b = idx[start: start + self.batch_size]
                obs_t    = torch.tensor(obs_a[b],     device=self.device)
                act_t    = torch.tensor(actions_a[b], device=self.device)
                old_lp_t = torch.tensor(lp_a[b],     device=self.device)
                ret_t    = torch.tensor(returns_a[b], device=self.device)
                adv_t    = torch.tensor(adv_a[b],     device=self.device)

                new_lp, values, entropy = self.network.evaluate(obs_t, act_t)
                ratio  = (new_lp - old_lp_t).exp()
                surr1  = ratio * adv_t
                surr2  = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_t
                p_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.mse_loss(values, ret_t)
                e_loss = -entropy.mean()
                loss   = p_loss + self.value_coef * v_loss + self.entropy_coef * e_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                totals['policy']  += p_loss.item()
                totals['value']   += v_loss.item()
                totals['entropy'] += entropy.mean().item()
                totals['total']   += loss.item()
                n_updates += 1

        self.update_count += 1
        if n_updates > 0:
            for k in totals:
                totals[k] /= n_updates
        return totals

    # ------------------------------------------------------------------

    def close(self) -> None:
        for pipe in self._pipes:
            try:
                pipe.send(('stop', None))
                pipe.recv()   # wait for ('done', None)
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self._pipes.clear()
        self._procs.clear()
