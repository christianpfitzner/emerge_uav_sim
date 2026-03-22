from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class WorldConfig:
    width: float = 100.0
    height: float = 100.0
    n_agents: int = 6
    n_pois: int = 8
    n_obstacles: int = 5
    obstacle_radius: float = 5.0
    grid_resolution: float = 5.0       # coverage cell size in world units
    base_pos: np.ndarray = None        # defaults to center on reset

    def __post_init__(self):
        if self.base_pos is None:
            self.base_pos = np.array([self.width / 2, self.height / 2])


@dataclass
class UAVConfig:
    max_speed: float = 5.0
    max_accel: float = 2.0
    sensor_range: float = 15.0
    comm_range: float = 20.0
    collision_radius: float = 2.0
    battery_drain_move: float = 0.001    # per step, proportional to speed
    battery_drain_hover: float = 0.0005  # per step stationary
    battery_charge_rate: float = 0.02    # per step at base
    base_dock_radius: float = 5.0        # distance to count as "at base"
    msg_dim: int = 4                     # learned message dims (RL action)
    structured_msg_dim: int = 7          # auto-filled structured dims prepended to broadcast


@dataclass
class TaskConfig:
    name: str = "area_coverage_inspection"   # or "search_and_report"
    max_steps: int = 1000
    inspection_steps: int = 20               # steps needed to fully inspect a POI
    # Reward weights
    reward_coverage: float = 0.1             # per newly discovered cell
    reward_relay: float = 1.0                # when message reaches base via relay
    reward_battery_return: float = 2.0       # returning before battery dies
    penalty_collision: float = -1.0
    penalty_dead: float = -5.0               # battery fully depleted
    # Exploration incentives (default 0 = off, set > 0 to enable)
    reward_explore_distance: float = 0.0     # per-step bonus per agent: strength * (dist_base / diagonal)
    reward_spread: float = 0.0               # per-step team bonus: strength * (mean_pairwise_dist / diagonal)
    reward_speed: float = 0.0               # per-step bonus per agent: strength * (speed / max_speed)
    loitering_penalty: float = 0.0           # per-step penalty per agent within loitering_radius of base
    loitering_radius: float = 15.0           # world units — zone around base that triggers loitering penalty
    relay_count_cooldown: int = 10           # min steps between relay reward+counter increments per agent
                                             # prevents relay from dominating (0 = no cooldown)
    penalty_redundant_poi: float = 0.0       # penalty per step for being near a POI that another agent is
                                             # already inspecting (discourages redundant clustering on POIs)
    # Coverage reward attribution: individual=True gives reward only to the exploring agent
    # (fixes free-rider problem); False = old shared behaviour
    individual_coverage_reward: bool = True
    # Delivery mechanic
    require_delivery: bool = False          # if True, rewards only count when info reaches base
    reward_delivery_poi: float = 10.0       # extra bonus per POI delivered to base
    reward_connected_to_base: float = 0.0   # per-step bonus per agent connected to base
                                            # (non-sparse signal that stabilises training with require_delivery)
    reward_inspection: float = 5.0           # reward per fully inspected POI
    role_shaping_strength: float = 0.5      # scale of role-specific bonuses
    penalty_same_cell: float = 0.0          # per-step penalty per agent when sharing a grid cell
    penalty_revisit: float = 0.0            # per-step penalty for being in an already-explored cell
    penalty_revisit_known: float = 0.0      # additional penalty if agent's personal map shows cell already visited
    penalty_low_battery: float = 0.0        # per-step penalty scaled by (low_battery_threshold - battery)
                                            # when battery < threshold and not at base (try 0.5–2.0)
    low_battery_threshold: float = 0.3      # battery level below which penalty_low_battery kicks in


@dataclass
class SimConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    uav: UAVConfig = field(default_factory=UAVConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    seed: int = 42
