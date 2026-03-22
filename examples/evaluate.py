#!/usr/bin/env python3
"""
Evaluate a trained MAPPO checkpoint — no further training.

Usage:
    python examples/evaluate.py --load checkpoints/final.pt
    python examples/evaluate.py --load checkpoints/ckpt_000050.pt --episodes 3
    python examples/evaluate.py --load checkpoints/final.pt --no-render
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained MAPPO checkpoint")
    p.add_argument("--load", type=str, required=True,
                   help="Path to checkpoint (.pt file)")
    p.add_argument("--episodes", type=int, default=1,
                   help="Number of episodes to run (default: 1)")
    p.add_argument("--no-render", action="store_true",
                   help="Disable visual rendering (terminal stats only)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed (default: random)")
    p.add_argument("--hidden-dim", type=int, default=128,
                   help="Hidden dim — must match the checkpoint (default: 128)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Max steps per episode (default: TaskConfig default = 1000)")
    return p.parse_args()


def main():
    args = parse_args()

    import numpy as np
    import torch

    seed = args.seed if args.seed is not None else int(np.random.default_rng().integers(0, 2**31))
    print(f"Seed       : {seed}  (pass --seed {seed} to reproduce)")
    np.random.seed(seed)
    torch.manual_seed(seed)
    args.seed = seed

    from emerge_uav_sim.envs.uav_team_env import UAVTeamEnv
    from emerge_uav_sim.config.configs import SimConfig
    from emerge_uav_sim.training.mappo import MAPPOTrainer

    from emerge_uav_sim.config.configs import TaskConfig
    render_mode = None if args.no_render else "human"
    task_kwargs = {}
    if args.max_steps is not None:
        task_kwargs["max_steps"] = args.max_steps
    cfg = SimConfig(seed=args.seed, task=TaskConfig(**task_kwargs))
    env = UAVTeamEnv(config=cfg, render_mode=render_mode)

    trainer = MAPPOTrainer(env, cfg={"hidden_dim": args.hidden_dim})
    trainer.load(args.load)
    trainer.network.eval()

    print(f"Checkpoint : {args.load}")
    print(f"Trained for: {trainer.total_steps:,} env steps "
          f"({trainer.update_count} updates)")
    print(f"Device     : {trainer.device}")
    print(f"Episodes   : {args.episodes}\n")

    agent_idx = env.agent_name_mapping
    all_ep_rewards = []

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        total_rewards = np.zeros(trainer.n_agents)
        step = 0

        while env.agents:
            obs_arr = np.zeros((trainer.n_agents, trainer.obs_dim), dtype=np.float32)
            for agent, o in obs.items():
                obs_arr[agent_idx[agent]] = o

            with torch.no_grad():
                obs_t = torch.tensor(obs_arr, device=trainer.device)
                # Use mean action (no sampling) for deterministic evaluation
                mean = trainer.network.actor_mean(trainer.network.backbone(obs_t))
                actions_np = mean.cpu().numpy()

            actions_dict = {
                agent: actions_np[agent_idx[agent]]
                for agent in env.agents
            }

            obs, rewards, _, _, _ = env.step(actions_dict)
            for agent, r in rewards.items():
                total_rewards[agent_idx[agent]] += r
            step += 1

        all_ep_rewards.append(total_rewards.copy())

        # Per-episode summary
        stats = env.episode_stats()
        coverage = env._world.coverage_fraction * 100 if env._world else 0.0
        n_insp   = env._world.n_inspected          if env._world else 0
        n_pois   = len(env._world.pois)            if env._world else 0

        print(f"── Episode {ep}/{args.episodes} "
              f"({step} steps, coverage {coverage:.1f}%, "
              f"POIs {n_insp}/{n_pois}) ──")
        print(f"  {'Agent':<10} {'Role':<10} {'Reward':>8}  "
              f"{'Cells':>6}  {'Inspect':>7}  {'Relay':>6}")
        for i, agent in enumerate(env.possible_agents):
            info = stats.get(agent, {})
            print(f"  {agent:<10} {info.get('role','?'):<10} "
                  f"{total_rewards[i]:>8.2f}  "
                  f"{info.get('cells_discovered', 0):>6.0f}  "
                  f"{info.get('inspection_steps', 0):>7.0f}  "
                  f"{info.get('messages_relayed', 0):>6.0f}")
        print()

    # Overall summary across episodes
    if args.episodes > 1:
        arr = np.stack(all_ep_rewards)   # (episodes, n_agents)
        mean_per_agent = arr.mean(axis=0)
        print(f"── Mean reward over {args.episodes} episodes ──")
        for i, agent in enumerate(env.possible_agents):
            print(f"  {agent}: {mean_per_agent[i]:.2f}")
        print(f"  Team total: {mean_per_agent.sum():.2f}")

    env.close()


if __name__ == "__main__":
    main()
