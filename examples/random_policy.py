"""
Random policy rollout with real-time Pygame visualization.

Usage:
    python examples/random_policy.py
"""
import time
import numpy as np

from emerge_uav_sim import UAVTeamEnv, SimConfig, WorldConfig, UAVConfig, TaskConfig


def main():
    cfg = SimConfig(
        world=WorldConfig(n_agents=10, n_pois=8, n_obstacles=10),
        uav=UAVConfig(max_speed=5.0, comm_range=20.0),
        task=TaskConfig(max_steps=1000, name="area_coverage_inspection"),
        seed=0,
    )
    env = UAVTeamEnv(cfg, render_mode="human")
    obs, _ = env.reset()

    total_rewards = {a: 0.0 for a in env.possible_agents}
    step = 0
    t0 = time.time()

    while env.agents:
        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }
        obs, rewards, terminations, truncations, infos = env.step(actions)
        for a, r in rewards.items():
            total_rewards[a] += r
        step += 1

    elapsed = time.time() - t0
    print(f"\nEpisode finished in {step} steps ({elapsed:.1f}s)")
    print("Total rewards:")
    for a, r in total_rewards.items():
        print(f"  {a}: {r:.2f}")
    print("\nRole stats:")
    stats = env.episode_stats()
    for agent, info in stats.items():
        print(f"  {agent}: role={info['role']}, cells={info['cells_discovered']:.0f}, "
              f"inspect={info['inspection_steps']:.0f}, relay={info['messages_relayed']:.0f}")

    env.close()


if __name__ == "__main__":
    main()
