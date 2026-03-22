"""
Run a trained policy checkpoint with Pygame visualization.

Usage:
    python3 examples/run_policy.py --load checkpoints/final.pt
    python3 examples/run_policy.py --load checkpoints/final.pt --episodes 3 --no-render
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from emerge_uav_sim import UAVTeamEnv, SimConfig
from emerge_uav_sim.training.networks import ActorCritic


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--load", required=True, help="Path to checkpoint .pt file")
    p.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    p.add_argument("--no-render", action="store_true", help="Disable visualization")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    render_mode = None if args.no_render else "human"
    env = UAVTeamEnv(SimConfig(seed=args.seed), render_mode=render_mode)

    # Build network and load weights
    sample_agent = env.possible_agents[0]
    obs_dim = env.observation_space(sample_agent).shape[0]
    act_dim = env.action_space(sample_agent).shape[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = ActorCritic(obs_dim=obs_dim, act_dim=act_dim).to(device)

    ckpt = torch.load(args.load, map_location=device)
    network.load_state_dict(ckpt["network"])
    network.eval()
    print(f"Loaded checkpoint: {args.load}  (trained for {ckpt.get('total_steps', '?')} steps)")

    agent_idx = env.agent_name_mapping

    for ep in range(args.episodes):
        obs_dict, _ = env.reset()
        total_rewards = {a: 0.0 for a in env.possible_agents}
        step = 0

        while env.agents:
            obs_arr = np.zeros((len(env.possible_agents), obs_dim), dtype=np.float32)
            for agent, obs in obs_dict.items():
                obs_arr[agent_idx[agent]] = obs

            with torch.no_grad():
                actions_t, _, _ = network.act(torch.tensor(obs_arr, device=device))
            actions_np = actions_t.cpu().numpy()

            actions_dict = {
                agent: actions_np[agent_idx[agent]]
                for agent in env.agents
            }

            obs_dict, rewards, _, _, _ = env.step(actions_dict)
            for a, r in rewards.items():
                total_rewards[a] += r
            step += 1

        print(f"\nEpisode {ep + 1} — {step} steps")
        print("Total rewards:")
        for a, r in total_rewards.items():
            print(f"  {a}: {r:.2f}")
        print("Role stats:")
        for agent, info in env.episode_stats().items():
            print(f"  {agent}: role={info['role']}, cells={info['cells_discovered']:.0f}, "
                  f"inspect={info['inspection_steps']:.0f}, relay={info['messages_relayed']:.0f}")

    env.close()


if __name__ == "__main__":
    main()
