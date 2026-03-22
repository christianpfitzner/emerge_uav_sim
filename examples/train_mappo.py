#!/usr/bin/env python3
"""Train IPPO (shared-parameter) agents on the UAV team environment."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_config_file(path: str) -> dict:
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    # Convert underscores to hyphens so keys match argparse dest names (argparse
    # converts hyphens→underscores internally, so we go the other way here).
    return {k.replace("-", "_"): v for k, v in data.items()}


def parse_args():
    # Pre-parse just --config so we can set file values as defaults before the
    # full parse. This means explicit CLI flags always override the config file.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()
    file_defaults = _load_config_file(pre_args.config) if pre_args.config else {}

    p = argparse.ArgumentParser(description="MAPPO trainer for emerge_uav_sim")
    p.add_argument("--config", type=str, default=None,
                   help="Path to a YAML parameter file (CLI args override file values)")
    p.add_argument("--total-steps", type=int, default=200_000,
                   help="Total env steps to train for (default: 200000)")
    p.add_argument("--n-steps", type=int, default=512,
                   help="Rollout length per update (default: 512)")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Adam learning rate (default: 3e-4)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--save-dir", type=str, default="checkpoints",
                   help="Directory to save checkpoints (default: checkpoints)")
    p.add_argument("--save-every", type=int, default=50,
                   help="Save checkpoint every N updates (default: 50)")
    p.add_argument("--render", action="store_true",
                   help="Render the environment during training (slow)")
    p.add_argument("--no-training-panel", action="store_true",
                   help="Disable the training progress window")
    p.add_argument("--n-workers", type=int, default=1,
                   help="Number of parallel rollout workers (default: 1). "
                        "Set to number of available CPU cores for max throughput.")
    p.add_argument("--load", type=str, default=None,
                   help="Path to checkpoint to resume from")
    # Reward shaping
    p.add_argument("--reward-coverage", type=float, default=0.1,
                   help="Reward per newly discovered coverage cell (default: 0.1)")
    p.add_argument("--exploration-bonus", type=float, default=0.0,
                   help="Per-step bonus per agent proportional to distance from base "
                        "(0=off, try 0.05–0.2 to encourage exploration)")
    p.add_argument("--reward-speed", type=float, default=0.0,
                   help="Per-step bonus per agent proportional to current speed "
                        "(0=off, try 0.02–0.05 to directly incentivise movement)")
    p.add_argument("--spread-bonus", type=float, default=0.0,
                   help="Per-step team bonus for mean pairwise agent distance "
                        "(0=off, try 0.05–0.15 to prevent clustering)")
    p.add_argument("--require-delivery", action="store_true",
                   help="Agents must relay discovered/inspected data back to base to earn rewards")
    p.add_argument("--reward-delivery-poi", type=float, default=10.0,
                   help="Extra bonus per POI delivered to base (default: 10.0)")
    p.add_argument("--connected-bonus", type=float, default=0.0,
                   help="Per-step bonus per agent connected to base via relay chain "
                        "(non-sparse signal; try 0.005–0.02 with --require-delivery)")
    p.add_argument("--penalty-redundant-poi", type=float, default=0.0,
                   help="Penalty per step for being near a POI another agent is already inspecting "
                        "(0=off, try -0.1 to -0.5 to discourage redundant clustering on POIs)")
    p.add_argument("--loitering-penalty", type=float, default=0.0,
                   help="Per-step penalty per agent within loitering-radius of base "
                        "(0=off, try 0.002–0.01 to discourage base-camping)")
    p.add_argument("--loitering-radius", type=float, default=15.0,
                   help="Radius around base that triggers the loitering penalty (default: 15)")
    p.add_argument("--reward-inspection", type=float, default=5.0,
                   help="Reward per fully inspected POI (default: 5.0)")
    p.add_argument("--role-shaping-strength", type=float, default=0.5,
                   help="Scale of role-specific shaping bonuses (default: 0.5)")
    p.add_argument("--penalty-same-cell", type=float, default=0.0,
                   help="Per-step penalty per agent when multiple agents share a grid cell (try -0.1 to -0.5)")
    p.add_argument("--reward-battery-return", type=float, default=2.0,
                   help="One-time bonus when agent docks at base, scaled by battery remaining (default: 2.0)")
    p.add_argument("--penalty-low-battery", type=float, default=0.0,
                   help="Per-step penalty scaled by (threshold - battery) when below threshold and not at base "
                        "(try 0.5–2.0 to push agents back before dying)")
    p.add_argument("--low-battery-threshold", type=float, default=0.3,
                   help="Battery level below which penalty_low_battery kicks in (default: 0.3)")
    p.add_argument("--penalty-revisit", type=float, default=0.0,
                   help="Per-step penalty for being in an already-explored cell (try -0.01 to -0.05)")
    p.add_argument("--penalty-revisit-known", type=float, default=0.0,
                   help="Additional penalty if agent's personal map shows cell was already visited (try -0.02 to -0.1)")
    p.add_argument("--penalty-dead", type=float, default=-5.0,
                   help="Penalty when an agent's battery hits zero (default: -5.0)")
    p.add_argument("--reward-relay", type=float, default=1.0,
                   help="Reward for relaying a message to base (default: 1.0)")
    p.add_argument("--relay-cooldown", type=int, default=10,
                   help="Min steps between relay reward+counter increments per agent "
                        "(prevents relay role dominance, 0=off, default: 10)")
    # PPO hyperparameters
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--log-std-init", type=float, default=-1.0,
                   help="Initial log_std for the policy (default: -1.0 → std≈0.37 → entropy≈2.5)")
    if file_defaults:
        p.set_defaults(**file_defaults)
    return p.parse_args()


def main():
    args = parse_args()

    import numpy as np
    import torch

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from emerge_uav_sim.envs.uav_team_env import UAVTeamEnv
    from emerge_uav_sim.config.configs import SimConfig
    from emerge_uav_sim.training.mappo import MAPPOTrainer

    from emerge_uav_sim.config.configs import TaskConfig

    render_mode = "human" if args.render else None
    cfg = SimConfig(
        seed=args.seed,
        task=TaskConfig(
            reward_coverage=args.reward_coverage,
            reward_explore_distance=args.exploration_bonus,
            reward_speed=args.reward_speed,
            reward_spread=args.spread_bonus,
            loitering_penalty=args.loitering_penalty,
            loitering_radius=args.loitering_radius,
            penalty_dead=args.penalty_dead,
            reward_relay=args.reward_relay,
            relay_count_cooldown=args.relay_cooldown,
            penalty_redundant_poi=args.penalty_redundant_poi,
            require_delivery=args.require_delivery,
            reward_delivery_poi=args.reward_delivery_poi,
            reward_connected_to_base=args.connected_bonus,
            reward_inspection=args.reward_inspection,
            role_shaping_strength=args.role_shaping_strength,
            penalty_same_cell=args.penalty_same_cell,
            reward_battery_return=args.reward_battery_return,
            penalty_low_battery=args.penalty_low_battery,
            low_battery_threshold=args.low_battery_threshold,
            penalty_revisit=args.penalty_revisit,
            penalty_revisit_known=args.penalty_revisit_known,
        ),
    )
    env = UAVTeamEnv(config=cfg, render_mode=render_mode)

    trainer_cfg = {
        "n_steps": args.n_steps,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_eps": args.clip_eps,
        "entropy_coef": args.entropy_coef,
        "value_coef": args.value_coef,
        "hidden_dim": args.hidden_dim,
        "log_std_init": args.log_std_init,
    }

    if args.n_workers > 1:
        from emerge_uav_sim.training.parallel_trainer import ParallelMAPPOTrainer
        trainer = ParallelMAPPOTrainer(env, cfg=trainer_cfg, n_workers=args.n_workers)
        print(f"Parallel workers: {args.n_workers}  "
              f"(effective batch: {args.n_steps * args.n_workers:,} steps/update)")
    else:
        trainer = MAPPOTrainer(env, cfg=trainer_cfg)

    if args.load:
        print(f"Loading checkpoint from {args.load}")
        trainer.load(args.load)

    print(f"Device: {trainer.device}")
    print(f"Obs dim: {trainer.obs_dim}, Act dim: {trainer.act_dim}, "
          f"Agents: {trainer.n_agents}")
    t = cfg.task
    print(f"\nReward config:")
    print(f"  coverage={t.reward_coverage}  (individual={t.individual_coverage_reward})")
    print(f"  inspection={t.reward_inspection}  relay={t.reward_relay}")
    print(f"  explore_distance={t.reward_explore_distance}  spread={t.reward_spread}")
    print(f"  loitering_penalty={t.loitering_penalty}  loitering_radius={t.loitering_radius}")
    print(f"  relay={t.reward_relay}  relay_cooldown={t.relay_count_cooldown} steps")
    print(f"  penalty_dead={t.penalty_dead}  penalty_collision={t.penalty_collision}")
    print(f"  require_delivery={t.require_delivery}  reward_delivery_poi={t.reward_delivery_poi}")
    print(f"\nTraining for {args.total_steps:,} env steps\n")

    trainer.train(
        total_steps=args.total_steps,
        save_dir=args.save_dir,
        save_every=args.save_every,
        show_training_panel=not args.no_training_panel,
    )

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
