from __future__ import annotations
from typing import List, Dict, Any
import numpy as np

from emerge_uav_sim.config.configs import TaskConfig


ROLE_NAMES = ["explorer", "inspector", "relay"]


class RoleTracker:
    """
    Tracks per-agent behavior statistics and computes role-shaping bonuses.
    Runs inside the env step; does NOT modify agent states directly (except
    the world updates messages_relayed via CommSystem).
    """

    def __init__(self, n_agents: int, task_cfg: TaskConfig):
        self.n = n_agents
        self.tcfg = task_cfg

        # Episode-level counters (reset each episode)
        self.episode_cells_discovered = np.zeros(n_agents, dtype=float)
        self.episode_inspections = np.zeros(n_agents, dtype=float)
        self.episode_relays = np.zeros(n_agents, dtype=float)

        # Step-level counters (reset each step, used for shaping)
        self._step_cells = np.zeros(n_agents, dtype=float)
        self._step_inspect = np.zeros(n_agents, dtype=float)
        self._step_relay = np.zeros(n_agents, dtype=float)

        # Relay cooldown: track step of last counted relay per agent
        self._last_relay_step = np.full(n_agents, -9999, dtype=int)
        self._step_counter = 0

    # ------------------------------------------------------------------

    def compute_rewards(
        self,
        states,           # List[UAVState]
        world,            # World
        step_data: Dict[str, Any],
    ) -> np.ndarray:
        """
        Compute per-agent reward for this step.

        Team reward (shared equally among alive agents):
          - coverage bonus for newly visited cells
          - inspection bonus for newly completed POIs
          - relay bonus for relay events

        Individual role-shaping bonuses (added on top):
          - Explorer: gets a fraction of the coverage reward proportional to
            how many new cells were discovered near this agent specifically.
            Approximation: award all new_cells to the single closest agent.
          - Inspector: gets a bonus for each step it is the closest agent
            to a POI being actively inspected.
          - Relay: gets a bonus each time this agent relays a msg to base.

        Penalties:
          - collision penalty
          - death penalty (first step battery hits 0)
        """
        n = self.n
        tcfg = self.tcfg
        alive = step_data["alive_mask"]
        n_alive = alive.sum()

        rewards = np.zeros(n)
        self._step_counter += 1

        # ---- Shared team rewards ----
        new_cells: int = step_data["new_cells"]
        newly_inspected: List[int] = step_data["newly_inspected"]
        relay_events: List[int] = step_data["relay_events"]

        # Determine whether to reward on discovery or on delivery
        if tcfg.require_delivery:
            reward_cells = step_data.get("new_delivered_cells", 0)
            reward_inspected = step_data.get("new_delivered_pois", [])
        else:
            reward_cells = new_cells
            reward_inspected = newly_inspected

        team_inspection_reward = tcfg.reward_inspection * len(reward_inspected)
        if n_alive > 0:
            team_per_agent = team_inspection_reward / n_alive
        else:
            team_per_agent = 0.0
        for i in range(n):
            if alive[i]:
                rewards[i] += team_per_agent

        # ---- Role shaping ----
        s = tcfg.role_shaping_strength
        closest_per_poi: List[int] = step_data["closest_per_poi"]
        positions = step_data["positions"]
        base_pos = world.wcfg.base_pos
        dists_to_base = np.array([
            np.linalg.norm(positions[i] - base_pos) if alive[i] else -np.inf
            for i in range(n)
        ])

        # Explorer shaping: each agent rewarded for cells new to their personal map.
        # With individual_coverage_reward=True each agent earns reward_coverage when
        # they step on a cell they have not personally visited before — this gives every
        # agent an independent incentive to explore new territory and naturally discourages
        # clustering (once an area is covered, the agent must move elsewhere to earn more).
        # Legacy farthest-agent heuristic is used as fallback when personal data is absent.
        revisit_personal = step_data.get("revisit_personal", np.zeros(n, dtype=bool))
        if tcfg.individual_coverage_reward:
            for i in range(n):
                if alive[i] and not revisit_personal[i]:
                    rewards[i] += tcfg.reward_coverage
                    self.episode_cells_discovered[i] += 1
                    self._step_cells[i] += 1
        elif reward_cells > 0 and n_alive > 0:
            # Legacy: shared among all + shaping bonus to farthest agent
            explorer_idx = int(np.argmax(dists_to_base))
            for i in range(n):
                if alive[i]:
                    rewards[i] += tcfg.reward_coverage * reward_cells / n_alive
            rewards[explorer_idx] += s * tcfg.reward_coverage * reward_cells
            self.episode_cells_discovered[explorer_idx] += reward_cells
            self._step_cells[explorer_idx] += reward_cells

        # Inspector shaping: closest agent to a POI being inspected
        for poi_idx, agent_idx in enumerate(closest_per_poi):
            poi = world.pois[poi_idx]
            if agent_idx >= 0 and not poi.inspected and poi.inspection_progress > 0:
                inspect_bonus = s * (tcfg.reward_inspection / tcfg.inspection_steps)
                rewards[agent_idx] += inspect_bonus
                self._step_inspect[agent_idx] += 1
                self.episode_inspections[agent_idx] += 1

        # Fully inspected POI bonus (per closest agent at completion)
        for poi_idx in newly_inspected:
            ca = closest_per_poi[poi_idx] if poi_idx < len(closest_per_poi) else -1
            if ca >= 0:
                rewards[ca] += s * tcfg.reward_inspection
                self.episode_inspections[ca] += 1

        # Redundant POI penalty: agents near a POI that another agent is already inspecting
        # get penalized to prevent multiple agents crowding the same target.
        if tcfg.penalty_redundant_poi != 0.0:
            sensor_range = world.ucfg.sensor_range
            for poi_idx, closest_agent in enumerate(closest_per_poi):
                poi = world.pois[poi_idx]
                if poi.inspected or closest_agent < 0 or poi.inspection_progress <= 0:
                    continue
                for i in range(n):
                    if not alive[i] or i == closest_agent:
                        continue
                    if np.linalg.norm(positions[i] - poi.pos) < sensor_range:
                        rewards[i] += tcfg.penalty_redundant_poi

        # Relay shaping — cooldown gates BOTH the reward and the counter to prevent relay
        # from dominating. Without this, relay reward fires every step, vastly outweighing coverage.
        cooldown = tcfg.relay_count_cooldown
        for agent_idx in relay_events:
            if cooldown <= 0 or (self._step_counter - self._last_relay_step[agent_idx]) >= cooldown:
                rewards[agent_idx] += s * tcfg.reward_relay
                self._step_relay[agent_idx] += 1
                self.episode_relays[agent_idx] += 1
                self._last_relay_step[agent_idx] = self._step_counter

        # ---- Penalties ----
        collision_flags = step_data["collision_flags"]
        for i in range(n):
            if collision_flags[i]:
                rewards[i] += tcfg.penalty_collision

        for i in range(n):
            s_state = states[i]
            if not s_state.alive and alive[i]:  # just died
                rewards[i] += tcfg.penalty_dead

        # ---- POI delivery bonus ----
        if tcfg.require_delivery and tcfg.reward_delivery_poi > 0:
            new_delivered_pois = step_data.get("new_delivered_pois", [])
            connected = step_data.get("connected_to_base", np.zeros(n, dtype=bool))
            for poi_idx in new_delivered_pois:
                # Bonus to the connected agent closest to the delivered POI
                poi_pos = world.pois[poi_idx].pos
                best_i = -1
                best_d = np.inf
                for i in range(n):
                    if alive[i] and connected[i]:
                        d = np.linalg.norm(positions[i] - poi_pos)
                        if d < best_d:
                            best_d = d
                            best_i = i
                if best_i >= 0:
                    rewards[best_i] += tcfg.reward_delivery_poi

        # ---- Connected-to-base bonus ----
        if tcfg.reward_connected_to_base > 0.0:
            connected = step_data.get("connected_to_base", np.zeros(n, dtype=bool))
            for i in range(n):
                if alive[i] and connected[i]:
                    rewards[i] += tcfg.reward_connected_to_base

        # ---- Exploration bonuses ----
        diag = np.sqrt(world.wcfg.width**2 + world.wcfg.height**2)
        base_pos = world.wcfg.base_pos

        if tcfg.reward_explore_distance > 0.0:
            for i in range(n):
                if alive[i]:
                    dist = np.linalg.norm(positions[i] - base_pos)
                    rewards[i] += tcfg.reward_explore_distance * (dist / diag)

        if tcfg.reward_speed > 0.0:
            max_speed = world.ucfg.max_speed
            for i in range(n):
                if alive[i]:
                    speed = np.linalg.norm(states[i].vel)
                    rewards[i] += tcfg.reward_speed * (speed / max_speed)

        if tcfg.reward_spread > 0.0 and n_alive >= 2:
            alive_positions = positions[alive]
            dists = []
            for i in range(len(alive_positions)):
                for j in range(i + 1, len(alive_positions)):
                    dists.append(np.linalg.norm(alive_positions[i] - alive_positions[j]))
            mean_spread = np.mean(dists) / diag
            spread_bonus = tcfg.reward_spread * mean_spread   # fixed: no /n_alive
            for i in range(n):
                if alive[i]:
                    rewards[i] += spread_bonus

        if tcfg.loitering_penalty > 0.0:
            for i in range(n):
                if alive[i] and dists_to_base[i] < tcfg.loitering_radius:
                    rewards[i] -= tcfg.loitering_penalty

        # ---- Battery return reward ----
        # One-time bonus when an agent docks at base while still alive, scaled by
        # battery remaining — rewards returning early rather than flying until dead.
        if tcfg.reward_battery_return > 0.0:
            dock_events = step_data.get("dock_events", np.zeros(n, dtype=bool))
            for i in range(n):
                if dock_events[i] and alive[i]:
                    rewards[i] += tcfg.reward_battery_return * states[i].battery

        # ---- Low battery penalty ----
        # Continuous per-step penalty when battery < threshold and not at base.
        # Provides an early gradient toward returning rather than waiting until death.
        if tcfg.penalty_low_battery > 0.0:
            now_at_base = step_data.get("now_at_base", np.zeros(n, dtype=bool))
            for i in range(n):
                if alive[i] and not now_at_base[i]:
                    deficit = tcfg.low_battery_threshold - states[i].battery
                    if deficit > 0:
                        rewards[i] -= tcfg.penalty_low_battery * deficit

        # ---- Same-cell penalty ----
        if tcfg.penalty_same_cell != 0.0:
            res = world.wcfg.grid_resolution
            cols = world.grid_cols
            rows = world.grid_rows
            cells = []
            for i in range(n):
                if alive[i]:
                    col = int(np.clip(positions[i, 0] / res, 0, cols - 1))
                    row = int(np.clip(positions[i, 1] / res, 0, rows - 1))
                    cells.append((i, row, col))
            from collections import Counter
            cell_counts = Counter((r, c) for _, r, c in cells)
            for i, row, col in cells:
                if cell_counts[(row, col)] > 1:
                    rewards[i] += tcfg.penalty_same_cell

        # ---- Revisit penalties ----
        # penalty_revisit: applied whenever an agent is in a globally already-explored cell
        # penalty_revisit_known: additional penalty if the agent's personal map also knew it
        if tcfg.penalty_revisit != 0.0 or tcfg.penalty_revisit_known != 0.0:
            revisit_global = step_data.get("revisit_global", np.zeros(n, dtype=bool))
            revisit_personal = step_data.get("revisit_personal", np.zeros(n, dtype=bool))
            for i in range(n):
                if alive[i] and revisit_global[i]:
                    rewards[i] += tcfg.penalty_revisit
                    if revisit_personal[i]:
                        rewards[i] += tcfg.penalty_revisit_known

        return rewards

    # ------------------------------------------------------------------

    def episode_stats(self, states) -> dict:
        """Return per-agent behavior stats and inferred role labels."""
        stats = {}
        for i in range(self.n):
            counters = np.array([
                self.episode_cells_discovered[i],
                self.episode_inspections[i],
                self.episode_relays[i],
            ], dtype=float)
            total = counters.sum()
            if total > 0:
                normalized = counters / total
                role_idx = int(np.argmax(normalized))
            else:
                role_idx = 0
            stats[f"uav_{i}"] = {
                "role": ROLE_NAMES[role_idx],
                "cells_discovered": self.episode_cells_discovered[i],
                "inspection_steps": self.episode_inspections[i],
                "messages_relayed": self.episode_relays[i],
                "messages_relayed_cumulative": states[i].messages_relayed,
            }
        return stats
