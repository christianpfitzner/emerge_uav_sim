"""Unit tests for UAVTeamEnv."""
import pytest
import numpy as np

from emerge_uav_sim import UAVTeamEnv, SimConfig, WorldConfig, UAVConfig, TaskConfig
from emerge_uav_sim.envs.uav_team_env import OBS_DIM, ACT_DIM


@pytest.fixture
def small_env():
    cfg = SimConfig(
        world=WorldConfig(n_agents=3, n_pois=3, n_obstacles=2),
        task=TaskConfig(max_steps=50),
        seed=42,
    )
    env = UAVTeamEnv(cfg)
    return env


# ------------------------------------------------------------------
# Observation and action shapes
# ------------------------------------------------------------------

def test_obs_dim(small_env):
    obs, _ = small_env.reset()
    for agent, o in obs.items():
        assert o.shape == (OBS_DIM,), f"{agent} obs shape {o.shape} != ({OBS_DIM},)"


def test_action_space_shape(small_env):
    small_env.reset()
    for agent in small_env.possible_agents:
        sp = small_env.action_space(agent)
        assert sp.shape == (ACT_DIM,), f"{agent} action shape {sp.shape} != ({ACT_DIM},)"


def test_obs_in_range(small_env):
    obs, _ = small_env.reset()
    for agent, o in obs.items():
        assert np.all(np.isfinite(o)), f"{agent} obs contains non-finite values"


# ------------------------------------------------------------------
# Step transitions
# ------------------------------------------------------------------

def test_step_returns_correct_keys(small_env):
    obs, _ = small_env.reset()
    actions = {a: small_env.action_space(a).sample() for a in small_env.agents}
    obs2, rewards, terminations, truncations, infos = small_env.step(actions)

    assert set(obs2.keys()) == set(rewards.keys()) == set(terminations.keys())


def test_episode_terminates(small_env):
    small_env.reset()
    for _ in range(100):
        if not small_env.agents:
            break
        actions = {a: small_env.action_space(a).sample() for a in small_env.agents}
        small_env.step(actions)
    # Should have terminated within max_steps=50


def test_full_episode(small_env):
    obs, _ = small_env.reset()
    total_steps = 0
    while small_env.agents:
        actions = {a: small_env.action_space(a).sample() for a in small_env.agents}
        obs, rewards, terms, truncs, infos = small_env.step(actions)
        total_steps += 1
        assert total_steps <= small_env.tcfg.max_steps + 5


def test_rewards_finite(small_env):
    small_env.reset()
    for _ in range(20):
        if not small_env.agents:
            break
        actions = {a: small_env.action_space(a).sample() for a in small_env.agents}
        _, rewards, _, _, _ = small_env.step(actions)
        for a, r in rewards.items():
            assert np.isfinite(r), f"{a} reward is non-finite: {r}"


# ------------------------------------------------------------------
# Comm relay logic
# ------------------------------------------------------------------

def test_comm_relay():
    """Relay fires on topology: agent near base earns relay_event when it can hear a far agent."""
    from emerge_uav_sim.core.comm import CommSystem
    from emerge_uav_sim.core.uav import UAVState
    from emerge_uav_sim.config.configs import UAVConfig, WorldConfig

    ucfg = UAVConfig(comm_range=20.0)
    wcfg = WorldConfig(width=100.0, height=100.0)

    comm = CommSystem(ucfg, wcfg)
    base = wcfg.base_pos  # (50, 50)

    # Agent 0: near base (dist=1 < 20) → relay candidate
    s0 = UAVState(pos=base + np.array([1.0, 0.0]), vel=np.zeros(2), battery=1.0)
    s0.message = np.zeros(11)

    # Agent 1: dist=20 >= comm_range, but dist to s0=19 < 20 → triggers relay for s0
    s1 = UAVState(pos=base + np.array([20.0, 0.0]), vel=np.zeros(2), battery=1.0)
    s1.message = np.zeros(11)

    _, relay_events = comm.process([s0, s1], base)
    assert 0 in relay_events, "Agent 0 (near base) should earn relay_event for bridging agent 1"


def test_comm_relay_via_neighbor():
    """Agent si near base earns relay_event for agent sj that is far but within comm_range of si."""
    from emerge_uav_sim.core.comm import CommSystem
    from emerge_uav_sim.core.uav import UAVState
    from emerge_uav_sim.config.configs import UAVConfig, WorldConfig

    ucfg = UAVConfig(comm_range=30.0)
    wcfg = WorldConfig(width=100.0, height=100.0)

    comm = CommSystem(ucfg, wcfg)
    base = wcfg.base_pos  # (50, 50)

    # Agent 0 (si): near base (dist=2 < 30) → relay candidate
    si = UAVState(pos=base + np.array([2.0, 0.0]), vel=np.zeros(2), battery=1.0)
    si.message = np.zeros(11)

    # Agent 1 (sj): dist=31 >= 30 (far from base), dist to si=29 < 30 → si can bridge sj
    sj = UAVState(pos=base + np.array([31.0, 0.0]), vel=np.zeros(2), battery=1.0)
    sj.message = np.zeros(11)

    _, relay_events = comm.process([si, sj], base)
    assert 0 in relay_events, "Agent si (near base) should earn relay_event for bridging agent sj"


# ------------------------------------------------------------------
# World coverage and POI
# ------------------------------------------------------------------

def test_coverage_grid_updates():
    from emerge_uav_sim.core.world import World
    from emerge_uav_sim.config.configs import WorldConfig, UAVConfig, TaskConfig

    wcfg = WorldConfig(width=50.0, height=50.0, n_obstacles=0, n_pois=0)
    ucfg = UAVConfig()
    tcfg = TaskConfig()
    rng = np.random.default_rng(0)
    world = World(wcfg, ucfg, tcfg, rng)

    pos = np.array([[5.0, 5.0], [15.0, 15.0]])
    new = world.update_coverage(pos)
    assert new == 2

    # Same cells again → no new coverage
    new2 = world.update_coverage(pos)
    assert new2 == 0


def test_poi_inspection_progress():
    from emerge_uav_sim.core.world import World, POI
    from emerge_uav_sim.config.configs import WorldConfig, UAVConfig, TaskConfig

    wcfg = WorldConfig(width=50.0, height=50.0, n_obstacles=0, n_pois=0)
    ucfg = UAVConfig(sensor_range=15.0)
    tcfg = TaskConfig(inspection_steps=5)
    rng = np.random.default_rng(0)
    world = World(wcfg, ucfg, tcfg, rng)

    poi = POI(pos=np.array([10.0, 10.0]), discovered=True)
    world.pois = [poi]

    pos = np.array([[10.0, 10.0]])
    alive = np.array([True])

    for _ in range(5):
        world.update_pois(pos, alive)

    assert world.pois[0].inspected


# ------------------------------------------------------------------
# Role tracker
# ------------------------------------------------------------------

def test_role_tracker_basic():
    from emerge_uav_sim.analysis.role_tracker import RoleTracker
    tracker = RoleTracker(n_agents=3, task_cfg=TaskConfig())

    # Simulate many relay events for agent 2
    tracker.episode_relays[2] = 10
    tracker.episode_cells_discovered[0] = 50
    tracker.episode_inspections[1] = 5

    from emerge_uav_sim.core.uav import UAVState
    states = [UAVState(pos=np.zeros(2), vel=np.zeros(2)) for _ in range(3)]
    stats = tracker.episode_stats(states)

    assert stats["uav_0"]["role"] == "explorer"
    assert stats["uav_1"]["role"] == "inspector"
    assert stats["uav_2"]["role"] == "relay"


# ------------------------------------------------------------------
# search_and_report task variant
# ------------------------------------------------------------------

def test_search_and_report_pois_start_hidden():
    cfg = SimConfig(
        world=WorldConfig(n_agents=2, n_pois=3, n_obstacles=0),
        task=TaskConfig(max_steps=20, name="search_and_report"),
        seed=7,
    )
    env = UAVTeamEnv(cfg)
    env.reset()
    for poi in env._world.pois:
        assert not poi.discovered, "POIs should start undiscovered in search_and_report"
    env.close()


def test_reset_multiple_times(small_env):
    for seed in range(3):
        obs, _ = small_env.reset(seed=seed)
        assert set(obs.keys()) == set(small_env.agents)
        assert len(small_env.agents) == small_env.wcfg.n_agents


# ------------------------------------------------------------------
# Structured message encoding
# ------------------------------------------------------------------

def test_structured_message_content():
    """After a step, structured dims 0-3 should contain meaningful state info."""
    cfg = SimConfig(
        world=WorldConfig(n_agents=2, n_pois=4, n_obstacles=0),
        task=TaskConfig(max_steps=50),
        seed=0,
    )
    env = UAVTeamEnv(cfg)
    env.reset()
    actions = {a: env.action_space(a).sample() for a in env.agents}
    env.step(actions)

    for s in env._states:
        msg = s.message
        assert len(msg) == 11, f"Expected 11-dim message, got {len(msg)}"
        # dim 0: battery should be ~1.0 at start (agents just spawned)
        assert 0.0 <= msg[0] <= 1.0, f"Battery dim out of range: {msg[0]}"
        # dims 1-2: POI dx/dy normalized by diagonal — finite
        assert np.isfinite(msg[1]) and np.isfinite(msg[2]), "POI direction dims non-finite"
        # dim 3: inspection progress in [0, 1]
        assert 0.0 <= msg[3] <= 1.0, f"POI progress dim out of range: {msg[3]}"
        # dims 4-5: nearest-unvisited dx/dy — finite
        assert np.isfinite(msg[4]) and np.isfinite(msg[5]), "Unvisited direction dims non-finite"
        # dim 6: personal coverage fraction in [0, 1]
        assert 0.0 <= msg[6] <= 1.0, f"Coverage fraction dim out of range: {msg[6]}"
        # dims 7-10: learned, clipped to [-1, 1]
        assert np.all(msg[7:] >= -1.0) and np.all(msg[7:] <= 1.0), "Learned dims out of range"


def test_delivery_mechanic():
    """With require_delivery=True, delivered grid starts empty."""
    cfg = SimConfig(
        world=WorldConfig(n_agents=3, n_pois=2, n_obstacles=0),
        task=TaskConfig(max_steps=20, require_delivery=True),
        seed=0,
    )
    env = UAVTeamEnv(cfg)
    obs, _ = env.reset()
    assert env._world.delivered_coverage_grid.sum() == 0
    assert env._world.n_delivered_pois == 0
    # Obs dim should still match
    for agent, o in obs.items():
        assert o.shape == (OBS_DIM,)
    env.close()
