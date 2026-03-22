"""
PettingZoo API validation for UAVTeamEnv.

Usage:
    python examples/check_env.py
"""
from pettingzoo.test import parallel_api_test
from emerge_uav_sim import UAVTeamEnv, SimConfig, WorldConfig, TaskConfig


def main():
    cfg = SimConfig(
        world=WorldConfig(n_agents=4, n_pois=4, n_obstacles=3),
        task=TaskConfig(max_steps=200),
    )
    env = UAVTeamEnv(cfg)
    print("Running PettingZoo parallel_api_test ...")
    parallel_api_test(env, num_cycles=100)
    print("All checks passed!")


if __name__ == "__main__":
    main()
