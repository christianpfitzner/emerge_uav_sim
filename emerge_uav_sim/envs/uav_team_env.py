from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import functools

import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers

from emerge_uav_sim.config.configs import SimConfig, WorldConfig, UAVConfig, TaskConfig
from emerge_uav_sim.core.uav import UAVState
from emerge_uav_sim.core.world import World
from emerge_uav_sim.core.comm import CommSystem
from emerge_uav_sim.analysis.role_tracker import RoleTracker


# Observation layout constants
_OWN_DIM = 6          # pos(2) + vel(2) + battery(1) + connected_to_base(1)
_BASE_DIM = 3          # dx, dy, dist (normalized)
_K_PEERS = 5
_PEER_DIM = 5          # rel_pos(2) + vel(2) + battery(1)
_M_POIS = 4
_POI_DIM = 4           # rel_pos(2) + progress(1) + discovered(1)
_STRUCTURED_MSG_DIM = 7  # auto-filled: battery, nearest-uninspected-POI dx/dy/progress,
                         #   nearest-unvisited-cell dx/dy (personal map), personal coverage fraction
_LEARNED_MSG_DIM = 4     # RL policy output (action dims 2–5)
_MSG_DIM = _STRUCTURED_MSG_DIM + _LEARNED_MSG_DIM  # total broadcast dim = 11
# Memory dims: personal exploration + base knowledge (when connected)
#   personal_fraction(1) + dir_nearest_unvisited_self(3)
#   base_fraction(1)     + dir_nearest_unknown_to_base(3)
_MEMORY_DIM = 8
OBS_DIM = _OWN_DIM + _BASE_DIM + _K_PEERS * _PEER_DIM + _M_POIS * _POI_DIM + _MSG_DIM + _MEMORY_DIM
# = 6 + 3 + 25 + 16 + 11 + 8 = 69

ACT_DIM = 2 + _LEARNED_MSG_DIM   # vel_delta(2) + learned_message(4)


class UAVTeamEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv for emergent UAV role specialization.

    Observation space: Box(53,) per agent
    Action space:      Box(6,) per agent
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "uav_team_v0"}

    def __init__(self, config: Optional[SimConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config or SimConfig()
        self.wcfg: WorldConfig = self.cfg.world
        self.ucfg: UAVConfig = self.cfg.uav
        self.tcfg: TaskConfig = self.cfg.task
        self.render_mode = render_mode

        self.possible_agents = [f"uav_{i}" for i in range(self.wcfg.n_agents)]
        self.agent_name_mapping = {a: i for i, a in enumerate(self.possible_agents)}

        self._rng = np.random.default_rng(self.cfg.seed)

        self._world: Optional[World] = None
        self._states: List[UAVState] = []
        self._comm = CommSystem(self.ucfg, self.wcfg)
        self._role_tracker: Optional[RoleTracker] = None
        self._step_count = 0
        self._received_msgs: Optional[np.ndarray] = None

        # Renderer (lazy init)
        self._renderer = None

    # ------------------------------------------------------------------
    # PettingZoo API: spaces
    # ------------------------------------------------------------------

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        low = np.array(
            [-self.ucfg.max_accel, -self.ucfg.max_accel, -1.0, -1.0, -1.0, -1.0],
            dtype=np.float32,
        )
        high = np.array(
            [self.ucfg.max_accel, self.ucfg.max_accel, 1.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        return spaces.Box(low=low, high=high, dtype=np.float32)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agents = list(self.possible_agents)
        self._step_count = 0

        # Build world
        self._world = World(self.wcfg, self.ucfg, self.tcfg, self._rng)

        # Spawn UAVs near base
        base = self.wcfg.base_pos
        grid_shape = (self._world.grid_rows, self._world.grid_cols)
        self._states = []
        for _ in range(self.wcfg.n_agents):
            offset = self._rng.uniform(-3.0, 3.0, size=2)
            pos = np.clip(base + offset, [0, 0], [self.wcfg.width, self.wcfg.height])
            self._states.append(
                UAVState(
                    pos=pos.astype(np.float64),
                    vel=np.zeros(2),
                    battery=1.0,
                    alive=True,
                    message=np.zeros(self.ucfg.msg_dim + self.ucfg.structured_msg_dim),
                    personal_grid=np.zeros(grid_shape, dtype=bool),
                    uploaded_grid=np.zeros(grid_shape, dtype=bool),
                )
            )

        # Comm & role tracker
        self._received_msgs = np.zeros((self.wcfg.n_agents, self.ucfg.msg_dim + self.ucfg.structured_msg_dim))
        self._role_tracker = RoleTracker(self.wcfg.n_agents, self.tcfg)
        self._connected_to_base = np.zeros(self.wcfg.n_agents, dtype=bool)
        self._last_rewards = np.zeros(self.wcfg.n_agents)
        # Initialise to True since agents spawn at base — prevents a spurious
        # dock_event (and free reward_battery_return) on the very first step.
        self._at_base = np.ones(self.wcfg.n_agents, dtype=bool)

        # Reset status panel history for new episode
        if self._renderer is not None and self._renderer._status_panel is not None:
            self._renderer._status_panel.reset()

        # Terminations / truncations
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}

        obs = self._build_observations()
        infos = {a: {} for a in self.agents}
        return obs, infos

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        if not self.agents:
            return {}, {}, {}, {}, {}

        self._step_count += 1
        n = self.wcfg.n_agents
        alive_mask = np.array([s.alive for s in self._states])

        # 1. Apply actions → update vel and pos; store messages
        collision_flags = np.zeros(n, dtype=bool)
        for i, agent in enumerate(self.possible_agents):
            if not alive_mask[i]:
                continue
            act = actions.get(agent, np.zeros(ACT_DIM))
            accel = np.clip(act[:2], -self.ucfg.max_accel, self.ucfg.max_accel)
            learned_msg = np.clip(act[2:], -1.0, 1.0)

            s = self._states[i]

            # Build structured dims:
            #   [0]   battery
            #   [1:3] dx/dy to nearest uninspected POI (normalized by diagonal)
            #   [3]   inspection progress of that POI
            #   [4:6] dx/dy to nearest unvisited cell in sender's personal map
            #   [6]   sender's personal coverage fraction
            # Peers receive all 7 dims and can use them to avoid already-explored
            # areas and coordinate towards uncovered territory without going via base.
            diag = np.sqrt(self.wcfg.width**2 + self.wcfg.height**2)
            structured = np.zeros(_STRUCTURED_MSG_DIM)
            structured[0] = s.battery
            uninspected = [p for p in self._world.pois if not p.inspected]
            if uninspected:
                nearest = min(uninspected, key=lambda p: np.linalg.norm(p.pos - s.pos))
                diff = nearest.pos - s.pos
                structured[1] = diff[0] / diag
                structured[2] = diff[1] / diag
                structured[3] = nearest.inspection_progress
            # Peer-to-peer map sharing: nearest unexplored cell + coverage fraction.
            # Receivers learn to avoid areas the sender already knows, enabling
            # implicit territory splitting without requiring base connectivity.
            if s.personal_grid is not None:
                unvisited_dir = self._nearest_unvisited(s.pos, s.personal_grid, diag)
                structured[4] = unvisited_dir[0]   # dx / diag
                structured[5] = unvisited_dir[1]   # dy / diag
                grid_size = self._world.grid_rows * self._world.grid_cols
                structured[6] = float(s.personal_grid.sum()) / grid_size

            s.message = np.concatenate([structured, learned_msg.astype(np.float64)])

            new_vel = s.vel + accel
            speed = np.linalg.norm(new_vel)
            if speed > self.ucfg.max_speed:
                new_vel = new_vel / speed * self.ucfg.max_speed
            s.vel = new_vel

            new_pos = s.pos + s.vel
            new_pos = np.clip(new_pos, [0, 0], [self.wcfg.width, self.wcfg.height])

            # 2. Bounce off obstacles
            new_pos, new_vel, obs_collided = self._world.bounce_from_obstacles(new_pos, new_vel)
            if obs_collided:
                collision_flags[i] = True
            s.pos = new_pos
            s.vel = new_vel

        # UAV-UAV collisions
        positions = np.array([s.pos for s in self._states])
        for i in range(n):
            if not alive_mask[i]:
                continue
            for j in range(i + 1, n):
                if not alive_mask[j]:
                    continue
                if np.linalg.norm(positions[i] - positions[j]) < self.ucfg.collision_radius * 2:
                    collision_flags[i] = True
                    collision_flags[j] = True

        # 3. Battery
        now_at_base = np.zeros(n, dtype=bool)
        for i in range(n):
            if not alive_mask[i]:
                continue
            s = self._states[i]
            speed = np.linalg.norm(s.vel)
            dist_to_base = np.linalg.norm(s.pos - self.wcfg.base_pos)
            now_at_base[i] = dist_to_base < self.ucfg.base_dock_radius
            if now_at_base[i]:
                s.battery = min(1.0, s.battery + self.ucfg.battery_charge_rate)
            else:
                if speed < 0.1:
                    drain = self.ucfg.battery_drain_hover
                else:
                    drain = self.ucfg.battery_drain_move * speed
                s.battery = max(0.0, s.battery - drain)
            if s.battery <= 0.0 and s.alive:
                s.alive = False
                alive_mask[i] = False
        # dock_events[i] = True when agent just arrived at base this step
        dock_events = now_at_base & ~self._at_base & alive_mask
        self._at_base = now_at_base

        # 4. POI inspection
        positions = np.array([s.pos for s in self._states])
        newly_inspected, closest_per_poi = self._world.update_pois(positions, alive_mask)

        # 5. Coverage grid
        # Capture revisit_global BEFORE update_coverage so we flag cells already explored
        revisit_global = np.zeros(n, dtype=bool)
        for i in range(n):
            if alive_mask[i]:
                col = int(np.clip(positions[i, 0] / self.wcfg.grid_resolution, 0, self._world.grid_cols - 1))
                row = int(np.clip(positions[i, 1] / self.wcfg.grid_resolution, 0, self._world.grid_rows - 1))
                revisit_global[i] = self._world.coverage_grid[row, col]
        alive_positions = positions[alive_mask]
        new_cells = self._world.update_coverage(alive_positions) if len(alive_positions) else 0

        # 6. Communication
        self._received_msgs, relay_events = self._comm.process(
            self._states, self.wcfg.base_pos, obstacles=self._world.obstacles
        )

        # Connectivity & delivery
        # Always track delivery for visualization; only use it for rewards when require_delivery=True.
        self._connected_to_base = self._comm.get_base_connected(
            self._states, self.wcfg.base_pos, obstacles=self._world.obstacles
        )
        connected_pos = positions[self._connected_to_base & alive_mask]
        new_delivered_cells, new_delivered_pois = self._world.update_delivered(connected_pos)
        if not self.tcfg.require_delivery:
            # Discard for reward purposes; delivery grid updated only for visualization
            new_delivered_cells = 0
            new_delivered_pois = []

        # Update personal grids and merge knowledge when connected to base.
        # Each alive agent marks its current cell; connected agents upload to base.
        # Capture revisit_personal BEFORE marking so we know if agent already had this cell.
        revisit_personal = np.zeros(n, dtype=bool)
        for i in range(n):
            s = self._states[i]
            if alive_mask[i] and s.personal_grid is not None:
                col = int(np.clip(positions[i, 0] / self.wcfg.grid_resolution, 0, self._world.grid_cols - 1))
                row = int(np.clip(positions[i, 1] / self.wcfg.grid_resolution, 0, self._world.grid_rows - 1))
                revisit_personal[i] = s.personal_grid[row, col]
        for i in range(n):
            s = self._states[i]
            if not alive_mask[i] or s.personal_grid is None:
                continue
            col = int(np.clip(positions[i, 0] / self.wcfg.grid_resolution, 0, self._world.grid_cols - 1))
            row = int(np.clip(positions[i, 1] / self.wcfg.grid_resolution, 0, self._world.grid_rows - 1))
            s.personal_grid[row, col] = True
            if self._connected_to_base[i]:
                # Only upload cells discovered since the last upload (delta),
                # so base knowledge grows incrementally rather than all at once.
                delta = s.personal_grid & ~s.uploaded_grid
                if delta.any():
                    self._world.merge_agent_knowledge(delta)
                    s.uploaded_grid |= delta

        # Sync delivered_coverage_grid with base_knowledge_grid so that all cells
        # an agent has ever visited are marked delivered as soon as they upload their
        # personal grid — not only the cell they are currently standing on.
        newly_from_base = self._world.base_knowledge_grid & ~self._world.delivered_coverage_grid
        if newly_from_base.any():
            if self.tcfg.require_delivery:
                new_delivered_cells += int(newly_from_base.sum())
            self._world.delivered_coverage_grid |= newly_from_base

        # 7. Compute rewards via role tracker
        step_data = {
            "new_cells": new_cells,
            "newly_inspected": newly_inspected,
            "closest_per_poi": closest_per_poi,
            "relay_events": relay_events,
            "collision_flags": collision_flags,
            "alive_mask": alive_mask,
            "positions": positions,
            "new_delivered_cells": new_delivered_cells,
            "new_delivered_pois": new_delivered_pois,
            "connected_to_base": self._connected_to_base,
            "revisit_global": revisit_global,
            "revisit_personal": revisit_personal,
            "dock_events": dock_events,
            "now_at_base": now_at_base,
        }
        rewards_arr = self._role_tracker.compute_rewards(self._states, self._world, step_data)
        self._last_rewards = rewards_arr

        # 8. Build observations
        obs = self._build_observations()

        # 9. Check termination conditions
        all_dead = not any(s.alive for s in self._states)
        all_inspected = self._world.all_inspected
        max_steps_reached = self._step_count >= self.tcfg.max_steps

        done = all_dead or all_inspected or max_steps_reached

        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for i, agent in enumerate(self.possible_agents):
            if agent not in self.agents:
                continue
            rewards[agent] = float(rewards_arr[i])
            terminations[agent] = done
            truncations[agent] = False
            infos[agent] = {
                "battery": self._states[i].battery,
                "alive": self._states[i].alive,
                "step": self._step_count,
            }

        self.terminations = terminations
        self.truncations = truncations

        if done:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return obs, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Memory feature builder
    # ------------------------------------------------------------------

    def _nearest_unvisited(self, pos: np.ndarray, grid: np.ndarray, diag: float) -> np.ndarray:
        """
        Return (dx/diag, dy/diag, dist/diag) toward the nearest grid cell not set in `grid`.
        Vectorized — no Python loops over cells.
        """
        unvisited = np.argwhere(~grid)   # (k, 2): rows of [row, col]
        if len(unvisited) == 0:
            return np.zeros(3, dtype=np.float32)
        res = self.wcfg.grid_resolution
        # cell center: x = (col+0.5)*res, y = (row+0.5)*res
        centers = np.column_stack([
            (unvisited[:, 1] + 0.5) * res,
            (unvisited[:, 0] + 0.5) * res,
        ])
        diffs = centers - pos
        dists = np.linalg.norm(diffs, axis=1)
        best = np.argmin(dists)
        d = diffs[best]
        return np.array([d[0] / diag, d[1] / diag, dists[best] / diag], dtype=np.float32)

    def _build_memory_features(self, agent_idx: int, s, diag: float) -> np.ndarray:
        """
        Build 8-dim memory observation for one agent:
          [0]   personal_fraction  — fraction of map this agent has visited
          [1:4] dir_nearest_unvisited_self — direction to nearest unvisited cell (personal)
          [4]   base_fraction — fraction of map base knows (0 if not connected)
          [5:8] dir_nearest_unknown_to_base — direction to nearest cell base doesn't know (0 if not connected)
        """
        feat = np.zeros(_MEMORY_DIM, dtype=np.float32)
        grid_size = self._world.grid_rows * self._world.grid_cols

        if s.personal_grid is not None:
            personal_visited = int(s.personal_grid.sum())
            feat[0] = personal_visited / grid_size
            feat[1:4] = self._nearest_unvisited(s.pos, s.personal_grid, diag)

        if self._connected_to_base[agent_idx]:
            base_known = int(self._world.base_knowledge_grid.sum())
            feat[4] = base_known / grid_size
            feat[5:8] = self._nearest_unvisited(s.pos, self._world.base_knowledge_grid, diag)

        return feat

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observations(self) -> Dict[str, np.ndarray]:
        obs_dict = {}
        n = self.wcfg.n_agents
        positions = np.array([s.pos for s in self._states])
        base = self.wcfg.base_pos

        for i, agent in enumerate(self.possible_agents):
            if agent not in self.agents:
                continue
            s = self._states[i]
            if not s.alive:
                obs_dict[agent] = np.zeros(OBS_DIM, dtype=np.float32)
                continue

            obs = np.zeros(OBS_DIM, dtype=np.float32)
            ptr = 0

            # Own state (6)
            obs[ptr:ptr+2] = s.pos / np.array([self.wcfg.width, self.wcfg.height])
            obs[ptr+2:ptr+4] = s.vel / self.ucfg.max_speed
            obs[ptr+4] = s.battery
            obs[ptr+5] = float(self._connected_to_base[i]) if hasattr(self, '_connected_to_base') else 0.0
            ptr += _OWN_DIM

            # Relative base (3)
            diff_base = base - s.pos
            dist_base = np.linalg.norm(diff_base)
            diag = np.sqrt(self.wcfg.width**2 + self.wcfg.height**2)
            obs[ptr:ptr+2] = diff_base / diag
            obs[ptr+2] = dist_base / diag
            ptr += _BASE_DIM

            # k=5 nearest peers (25)
            peer_dists = []
            for j in range(n):
                if j == i:
                    continue
                if not self._states[j].alive:
                    peer_dists.append((np.inf, j))
                else:
                    peer_dists.append((np.linalg.norm(positions[j] - s.pos), j))
            peer_dists.sort(key=lambda x: x[0])
            for k in range(_K_PEERS):
                if k < len(peer_dists) and peer_dists[k][0] < np.inf:
                    j = peer_dists[k][1]
                    pj = self._states[j]
                    rel_pos = (pj.pos - s.pos) / diag
                    obs[ptr:ptr+2] = rel_pos
                    obs[ptr+2:ptr+4] = pj.vel / self.ucfg.max_speed
                    obs[ptr+4] = pj.battery
                ptr += _PEER_DIM

            # m=4 nearest POIs (16)
            poi_dists = []
            for pi_idx, poi in enumerate(self._world.pois):
                d = np.linalg.norm(poi.pos - s.pos)
                poi_dists.append((d, pi_idx))
            poi_dists.sort(key=lambda x: x[0])
            for k in range(_M_POIS):
                if k < len(poi_dists):
                    _, pi_idx = poi_dists[k]
                    poi = self._world.pois[pi_idx]
                    if poi.discovered or self.tcfg.name != "search_and_report":
                        rel_pos = (poi.pos - s.pos) / diag
                        obs[ptr:ptr+2] = rel_pos
                        obs[ptr+2] = poi.inspection_progress
                        obs[ptr+3] = float(poi.discovered)
                ptr += _POI_DIM

            # Received messages (8 = 4 structured + 4 learned)
            obs[ptr:ptr+_MSG_DIM] = self._received_msgs[i]
            ptr += _MSG_DIM

            # Memory features (8)
            # personal_fraction(1) + dir_nearest_unvisited_self(3)
            # base_fraction(1)     + dir_nearest_unknown_to_base(3)
            obs[ptr:ptr+_MEMORY_DIM] = self._build_memory_features(i, s, diag)
            ptr += _MEMORY_DIM

            assert ptr == OBS_DIM, f"Obs dim mismatch: {ptr} != {OBS_DIM}"
            obs_dict[agent] = obs

        return obs_dict

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self):
        if self._renderer is None:
            from emerge_uav_sim.rendering.renderer import Renderer
            self._renderer = Renderer(self.wcfg, self.ucfg)
        return self._renderer.render(
            states=self._states,
            world=self._world,
            step=self._step_count,
            role_tracker=self._role_tracker,
            step_rewards=getattr(self, "_last_rewards", None),
        )

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Episode stats helper
    # ------------------------------------------------------------------

    def episode_stats(self) -> dict:
        if self._role_tracker is None:
            return {}
        return self._role_tracker.episode_stats(self._states)
