from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class UAVState:
    pos: np.ndarray        # shape (2,)
    vel: np.ndarray        # shape (2,)
    battery: float = 1.0  # [0, 1]
    alive: bool = True

    # Behavior counters for role tracking (accumulated per episode)
    steps_exploring: int = 0
    steps_inspecting: int = 0
    messages_relayed: int = 0

    # Latest outgoing message broadcast this step
    message: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # Personal exploration memory: bool grid of cells this agent has visited.
    # Initialized to None; set to the correct shape by UAVTeamEnv.reset().
    personal_grid: Optional[np.ndarray] = field(default=None, repr=False)

    # Snapshot of personal_grid at the time of last upload to base.
    # Only the delta (new cells since last upload) is merged into base_knowledge_grid,
    # so cells don't all flash green at once when an agent passes by base.
    uploaded_grid: Optional[np.ndarray] = field(default=None, repr=False)

    def copy(self) -> UAVState:
        return UAVState(
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            battery=self.battery,
            alive=self.alive,
            steps_exploring=self.steps_exploring,
            steps_inspecting=self.steps_inspecting,
            messages_relayed=self.messages_relayed,
            message=self.message.copy(),
            personal_grid=self.personal_grid.copy() if self.personal_grid is not None else None,
            uploaded_grid=self.uploaded_grid.copy() if self.uploaded_grid is not None else None,
        )
