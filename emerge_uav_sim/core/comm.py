from __future__ import annotations
from typing import List, Tuple
import numpy as np

from emerge_uav_sim.config.configs import UAVConfig, WorldConfig
from emerge_uav_sim.core.uav import UAVState


def has_los(p1: np.ndarray, p2: np.ndarray, obstacles) -> bool:
    """Return True if the line segment p1→p2 is not blocked by any obstacle circle."""
    d = p2 - p1
    seg_len = float(np.linalg.norm(d))
    if seg_len < 1e-6:
        return True
    d_norm = d / seg_len
    for obs in obstacles:
        to_obs = obs.pos - p1
        proj = float(np.clip(np.dot(to_obs, d_norm), 0.0, seg_len))
        closest = p1 + proj * d_norm
        if np.linalg.norm(closest - obs.pos) < obs.radius:
            return False
    return True


class CommSystem:
    """
    Range-limited communication with multi-hop relay support.

    Each agent broadcasts a msg_dim-vector.  Agents within comm_range
    receive each other's messages.

    Relay forwarding (search_and_report):
    - Messages can be tagged "for base" via a flag in message[4] > 0.5
      (first learned dim; dims 0-3 are structured/auto-filled).
    - If a receiving agent is within comm_range of base, it "relays" the
      message to base and the original sender's messages_relayed counter
      increments.
    """

    def __init__(self, uav_cfg: UAVConfig, world_cfg: WorldConfig):
        self.ucfg = uav_cfg
        self.wcfg = world_cfg

    def process(
        self,
        states: List[UAVState],
        base_pos: np.ndarray,
        obstacles=None,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Process one communication step.

        Returns:
            received_msgs: (n_agents, msg_dim) mean-pooled messages received
            relay_events: list of agent indices whose messages were relayed to base
        """
        n = len(states)
        msg_dim = self.ucfg.msg_dim + self.ucfg.structured_msg_dim
        comm_range = self.ucfg.comm_range

        positions = np.array([s.pos for s in states])
        messages = np.array([s.message for s in states])  # (n, msg_dim)
        alive = np.array([s.alive for s in states])

        received_msgs = np.zeros((n, msg_dim))
        relay_events: List[int] = []

        # Pairwise distances
        diff = positions[:, None, :] - positions[None, :, :]      # (n, n, 2)
        dists = np.linalg.norm(diff, axis=-1)                      # (n, n)

        for i in range(n):
            if not alive[i]:
                continue
            # Collect messages from neighbors within comm_range (excluding self)
            neighbors = [
                j for j in range(n)
                if j != i and alive[j] and dists[i, j] < comm_range
                and (obstacles is None or has_los(positions[i], positions[j], obstacles))
            ]
            if neighbors:
                received_msgs[i] = messages[neighbors].mean(axis=0)

        # Relay forwarding: agent i within comm_range of base can hear agent j
        # that is outside comm_range of base → i is acting as a relay.
        # Reward goes to i (the relay agent), not j (the sender), so there is a
        # direct incentive to position near base and stay in the comm chain.
        # No message flag required — relay fires purely on topology.
        dist_to_base = np.linalg.norm(positions - base_pos, axis=1)  # (n,)
        rewarded_relays = set()

        for i in range(n):
            if not alive[i]:
                continue
            if dist_to_base[i] >= comm_range:
                continue
            if obstacles is not None and not has_los(positions[i], base_pos, obstacles):
                continue
            # Agent i is near base — reward it for each distant agent it can hear
            for j in range(n):
                if j == i or not alive[j]:
                    continue
                if dist_to_base[j] >= comm_range and dists[i, j] < comm_range and (
                    obstacles is None or has_los(positions[i], positions[j], obstacles)
                ):
                    if i not in rewarded_relays:
                        rewarded_relays.add(i)
                        relay_events.append(i)

        # Update messages_relayed counters
        for idx in relay_events:
            states[idx].messages_relayed += 1

        return received_msgs, relay_events

    def get_base_connected(
        self, states, base_pos: np.ndarray, obstacles=None
    ) -> np.ndarray:
        """
        BFS from base: return bool array of agents that have a path to base
        via the comm network (direct or multi-hop relay), respecting LOS.
        """
        n = len(states)
        positions = np.array([s.pos for s in states])
        alive = np.array([s.alive for s in states])
        comm_r = self.ucfg.comm_range

        reachable = set()
        for i in range(n):
            if alive[i] and np.linalg.norm(positions[i] - base_pos) < comm_r:
                if obstacles is None or has_los(positions[i], base_pos, obstacles):
                    reachable.add(i)

        changed = True
        while changed:
            changed = False
            for i in range(n):
                if not alive[i] or i in reachable:
                    continue
                for j in reachable:
                    if np.linalg.norm(positions[i] - positions[j]) < comm_r:
                        if obstacles is None or has_los(positions[i], positions[j], obstacles):
                            reachable.add(i)
                            changed = True
                            break

        result = np.zeros(n, dtype=bool)
        for i in reachable:
            result[i] = True
        return result

    @staticmethod
    def get_neighbor_indices(
        positions: np.ndarray,
        alive: np.ndarray,
        comm_range: float,
    ) -> List[List[int]]:
        """Return list of neighbor agent indices for each agent."""
        n = len(positions)
        diff = positions[:, None, :] - positions[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        result = []
        for i in range(n):
            if not alive[i]:
                result.append([])
                continue
            neighbors = [
                j for j in range(n)
                if j != i and alive[j] and dists[i, j] < comm_range
            ]
            result.append(neighbors)
        return result
