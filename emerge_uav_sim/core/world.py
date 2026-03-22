from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

from emerge_uav_sim.config.configs import WorldConfig, UAVConfig, TaskConfig


@dataclass
class Obstacle:
    pos: np.ndarray   # center (2,)
    radius: float


@dataclass
class POI:
    pos: np.ndarray
    inspection_progress: float = 0.0   # 0 → 1
    inspected: bool = False
    discovered: bool = False           # True once any agent comes within sensor_range
    delivered: bool = False            # True once inspected data reaches base via comm chain


class World:
    def __init__(
        self,
        world_cfg: WorldConfig,
        uav_cfg: UAVConfig,
        task_cfg: TaskConfig,
        rng: np.random.Generator,
    ):
        self.wcfg = world_cfg
        self.ucfg = uav_cfg
        self.tcfg = task_cfg

        self.grid_cols = int(np.ceil(world_cfg.width / world_cfg.grid_resolution))
        self.grid_rows = int(np.ceil(world_cfg.height / world_cfg.grid_resolution))
        self.coverage_grid: np.ndarray = np.zeros(
            (self.grid_rows, self.grid_cols), dtype=bool
        )
        self.delivered_coverage_grid: np.ndarray = np.zeros(
            (self.grid_rows, self.grid_cols), dtype=bool
        )
        # Base knowledge: union of personal grids from all agents that have
        # delivered their data (connected to base). Shared back to connected agents.
        self.base_knowledge_grid: np.ndarray = np.zeros(
            (self.grid_rows, self.grid_cols), dtype=bool
        )

        self.obstacles: List[Obstacle] = []
        self.pois: List[POI] = []
        self._reset(rng)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset(self, rng: np.random.Generator):
        self.coverage_grid[:] = False
        self.delivered_coverage_grid[:] = False
        self.base_knowledge_grid[:] = False
        self.obstacles = self._place_obstacles(rng)
        self.pois = self._place_pois(rng)

    def reset(self, rng: np.random.Generator):
        self._reset(rng)

    # ------------------------------------------------------------------
    # Placement helpers
    # ------------------------------------------------------------------

    def _place_obstacles(self, rng: np.random.Generator) -> List[Obstacle]:
        obstacles = []
        base = self.wcfg.base_pos
        r = self.wcfg.obstacle_radius
        margin = r + 2.0
        for _ in range(self.wcfg.n_obstacles):
            for _attempt in range(100):
                pos = rng.uniform(
                    [margin, margin],
                    [self.wcfg.width - margin, self.wcfg.height - margin],
                )
                # Keep clear of base
                if np.linalg.norm(pos - base) < r * 3:
                    continue
                # No overlap with existing obstacles
                if any(np.linalg.norm(pos - o.pos) < 2 * r + 1 for o in obstacles):
                    continue
                obstacles.append(Obstacle(pos=pos, radius=r))
                break
        return obstacles

    def _place_pois(self, rng: np.random.Generator) -> List[POI]:
        pois = []
        base = self.wcfg.base_pos
        r = self.wcfg.obstacle_radius
        margin = 5.0
        for _ in range(self.wcfg.n_pois):
            for _attempt in range(100):
                pos = rng.uniform(
                    [margin, margin],
                    [self.wcfg.width - margin, self.wcfg.height - margin],
                )
                # Keep clear of base
                if np.linalg.norm(pos - base) < 10.0:
                    continue
                # Keep clear of obstacles
                if any(np.linalg.norm(pos - o.pos) < o.radius + 3 for o in self.obstacles):
                    continue
                # No overlap with existing POIs
                if any(np.linalg.norm(pos - p.pos) < 8.0 for p in pois):
                    continue
                discovered = self.tcfg.name != "search_and_report"
                pois.append(POI(pos=pos, discovered=discovered))
                break
        return pois

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def update_coverage(self, positions: np.ndarray) -> int:
        """Mark cells as visited; return count of newly covered cells."""
        new_cells = 0
        for pos in positions:
            col = int(pos[0] / self.wcfg.grid_resolution)
            row = int(pos[1] / self.wcfg.grid_resolution)
            col = np.clip(col, 0, self.grid_cols - 1)
            row = np.clip(row, 0, self.grid_rows - 1)
            if not self.coverage_grid[row, col]:
                self.coverage_grid[row, col] = True
                new_cells += 1
        return new_cells

    def update_pois(
        self, positions: np.ndarray, alive_mask: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """
        Advance inspection for agents within sensor_range of a POI.
        Discover hidden POIs (search_and_report mode).

        Returns:
            newly_inspected: indices of POIs completed this step
            closest_agent_per_poi: agent index closest to each POI (for inspector bonus)
        """
        newly_inspected = []
        closest_agent_per_poi = []

        for poi_idx, poi in enumerate(self.pois):
            if poi.inspected:
                closest_agent_per_poi.append(-1)
                continue

            dists = np.linalg.norm(positions - poi.pos, axis=1)
            # Mask dead agents
            dists[~alive_mask] = np.inf

            in_range = dists < self.ucfg.sensor_range
            closest = int(np.argmin(dists)) if in_range.any() else -1

            # Discover POI if any alive agent is close enough
            if in_range.any():
                poi.discovered = True

            # Advance inspection if any agent is within range
            if in_range.any():
                poi.inspection_progress += 1.0 / self.tcfg.inspection_steps
                poi.inspection_progress = min(poi.inspection_progress, 1.0)
                if poi.inspection_progress >= 1.0 and not poi.inspected:
                    poi.inspected = True
                    newly_inspected.append(poi_idx)

            closest_agent_per_poi.append(closest)

        return newly_inspected, closest_agent_per_poi

    def check_obstacle_collision(self, pos: np.ndarray) -> bool:
        """Return True if pos is inside any obstacle."""
        for obs in self.obstacles:
            if np.linalg.norm(pos - obs.pos) < obs.radius:
                return True
        return False

    def bounce_from_obstacles(
        self, pos: np.ndarray, vel: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Elastic bounce: push agent out of obstacle and reflect velocity. Returns (pos, vel, collided)."""
        collided = False
        for obs in self.obstacles:
            diff = pos - obs.pos
            dist = np.linalg.norm(diff)
            if dist < obs.radius + 0.5:
                collided = True
                if dist < 1e-6:
                    diff = np.array([1.0, 0.0])
                    dist = 1.0
                normal = diff / dist
                # Push outside
                pos = obs.pos + normal * (obs.radius + 0.5)
                # Reflect velocity
                vel = vel - 2 * np.dot(vel, normal) * normal
        return pos, vel, collided

    def update_delivered(
        self, connected_positions: np.ndarray
    ) -> tuple:
        """
        Mark cells and POIs as delivered when a base-connected agent is present.
        connected_positions: (k, 2) positions of agents that have a path to base.
        Returns (new_delivered_cells: int, new_delivered_poi_indices: List[int])
        """
        new_cells = 0
        if len(connected_positions) == 0:
            return 0, []

        for pos in connected_positions:
            col = int(pos[0] / self.wcfg.grid_resolution)
            row = int(pos[1] / self.wcfg.grid_resolution)
            col = int(np.clip(col, 0, self.grid_cols - 1))
            row = int(np.clip(row, 0, self.grid_rows - 1))
            if self.coverage_grid[row, col] and not self.delivered_coverage_grid[row, col]:
                self.delivered_coverage_grid[row, col] = True
                new_cells += 1

        new_pois = []
        for poi_idx, poi in enumerate(self.pois):
            if poi.inspected and not poi.delivered:
                dists = np.linalg.norm(connected_positions - poi.pos, axis=1)
                if dists.min() < self.ucfg.sensor_range:
                    poi.delivered = True
                    new_pois.append(poi_idx)

        return new_cells, new_pois

    def merge_agent_knowledge(self, personal_grid: np.ndarray) -> None:
        """Merge an agent's personal exploration grid into the base knowledge grid."""
        self.base_knowledge_grid |= personal_grid

    @property
    def delivered_fraction(self) -> float:
        return float(self.delivered_coverage_grid.sum()) / self.delivered_coverage_grid.size

    @property
    def n_delivered_pois(self) -> int:
        return sum(p.delivered for p in self.pois)

    @property
    def n_inspected(self) -> int:
        return sum(p.inspected for p in self.pois)

    @property
    def all_inspected(self) -> bool:
        return all(p.inspected for p in self.pois)

    @property
    def coverage_fraction(self) -> float:
        return float(self.coverage_grid.sum()) / self.coverage_grid.size
