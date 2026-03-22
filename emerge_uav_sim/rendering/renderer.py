from __future__ import annotations
from typing import List, Optional
import numpy as np

from emerge_uav_sim.config.configs import WorldConfig, UAVConfig

# Role colors: explorer=orange, inspector=red, relay=blue
_ROLE_COLORS = {
    "explorer": (255, 140, 0),
    "inspector": (220, 50, 50),
    "relay": (50, 100, 220),
    "unknown": (180, 180, 180),
}

_BG_COLOR = (20, 20, 30)
_GRID_COLOR = (35, 35, 50)
_OBSTACLE_COLOR = (90, 90, 100)
_POI_UNDISCOVERED_COLOR = (80, 80, 200)
_POI_INSPECTED_COLOR = (50, 200, 80)
_COMM_LINK_COLOR = (60, 120, 200, 80)
_COVERAGE_COLOR = (50, 200, 80, 40)
_HUD_TEXT_COLOR = (220, 220, 220)
_BASE_COLOR = (200, 180, 50)

# Comm-net overlay colours
_NET_HALO_FILL   = (60, 120, 200, 15)   # range circle fill
_NET_HALO_EDGE   = (60, 120, 200, 55)   # range circle outline
_NET_LINK_BOTH   = (40, 210, 160, 140)  # both endpoints reach base
_NET_LINK_ONE    = (80, 140, 220, 100)  # one endpoint reaches base
_NET_LINK_NONE   = (100, 100, 120, 55)  # isolated link
_NET_LINK_RELAY  = (230, 160, 30, 200)  # relay-flagged message
_NET_REACH_RING  = (40, 210, 160)       # ring on base-reachable UAVs

SCALE = 8          # world units → pixels
WINDOW_PADDING = 40
HUD_HEIGHT = 80

_FPS_LEVELS = [5, 10, 20, 30, 60]


class Renderer:
    """Pygame renderer for UAVTeamEnv."""

    def __init__(
        self,
        world_cfg: WorldConfig,
        uav_cfg: UAVConfig,
        show_comm_net: bool = True,
        show_status_panel: bool = True,
    ):
        self.wcfg = world_cfg
        self.ucfg = uav_cfg
        self.show_comm_net = show_comm_net  # toggle with N key at runtime
        self._pygame_init = False
        self._screen = None
        self._clock = None
        self._font = None

        # Speed / pause controls
        self._paused = False
        self._fps_idx = _FPS_LEVELS.index(30)   # default 30 fps

        # Optional status panel (second window)
        self._status_panel = None
        if show_status_panel:
            try:
                from emerge_uav_sim.rendering.status_panel import StatusPanel
                self._status_panel = StatusPanel(world_cfg.n_agents)
            except Exception:
                pass  # matplotlib not available — no status panel

        self.win_w = int(world_cfg.width * SCALE) + 2 * WINDOW_PADDING
        self.win_h = int(world_cfg.height * SCALE) + 2 * WINDOW_PADDING + HUD_HEIGHT

    def _init_pygame(self):
        if self._pygame_init:
            return
        import pygame
        pygame.init()
        self._screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption("Emergent UAV Role Simulator")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("monospace", 13)
        self._coverage_surf = None
        self._pygame = pygame
        self._pygame_init = True

    def _w2s(self, pos: np.ndarray) -> tuple:
        """World coordinates → screen coordinates."""
        x = int(pos[0] * SCALE + WINDOW_PADDING)
        y = int((self.wcfg.height - pos[1]) * SCALE + WINDOW_PADDING)
        return (x, y)

    def render(self, states, world, step: int, role_tracker=None, step_rewards=None):
        self._init_pygame()
        pygame = self._pygame
        screen = self._screen

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n:
                    self.show_comm_net = not self.show_comm_net
                elif event.key == pygame.K_SPACE:
                    self._paused = not self._paused
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_UP):
                    self._fps_idx = min(self._fps_idx + 1, len(_FPS_LEVELS) - 1)
                elif event.key in (pygame.K_MINUS, pygame.K_DOWN):
                    self._fps_idx = max(self._fps_idx - 1, 0)

        # Pause loop — keep the window alive but don't advance
        while self._paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self._paused = False
            # Draw PAUSED overlay
            overlay = pygame.Surface((self.win_w, self.win_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 110))
            screen.blit(overlay, (0, 0))
            pause_surf = self._font.render(
                "⏸  PAUSED  —  SPACE to resume", True, (255, 220, 60)
            )
            screen.blit(pause_surf, (
                self.win_w // 2 - pause_surf.get_width() // 2,
                self.win_h // 2 - pause_surf.get_height() // 2,
            ))
            pygame.display.flip()
            self._clock.tick(15)

        # Update status panel
        if self._status_panel is not None:
            self._status_panel.update(
                states, world, step, role_tracker, step_rewards
            )

        screen.fill(_BG_COLOR)

        # ---- Coverage overlay ----
        cell_px = int(self.wcfg.grid_resolution * SCALE)
        for row in range(world.grid_rows):
            for col in range(world.grid_cols):
                if world.delivered_coverage_grid[row, col]:
                    # Delivered: bright green
                    surf = pygame.Surface((cell_px, cell_px), pygame.SRCALPHA)
                    surf.fill((50, 210, 80, 55))
                    wx = col * self.wcfg.grid_resolution
                    wy = row * self.wcfg.grid_resolution
                    sx, sy = self._w2s(np.array([wx, wy + self.wcfg.grid_resolution]))
                    screen.blit(surf, (sx, sy))
                elif world.coverage_grid[row, col]:
                    # Discovered but not yet delivered: dim yellow
                    surf = pygame.Surface((cell_px, cell_px), pygame.SRCALPHA)
                    surf.fill((200, 180, 50, 30))
                    wx = col * self.wcfg.grid_resolution
                    wy = row * self.wcfg.grid_resolution
                    sx, sy = self._w2s(np.array([wx, wy + self.wcfg.grid_resolution]))
                    screen.blit(surf, (sx, sy))

        # ---- Obstacles ----
        for obs in world.obstacles:
            sx, sy = self._w2s(obs.pos)
            r_px = int(obs.radius * SCALE)
            pygame.draw.circle(screen, _OBSTACLE_COLOR, (sx, sy), r_px)

        # ---- Base ----
        base_s = self._w2s(self.wcfg.base_pos)
        pygame.draw.circle(screen, _BASE_COLOR, base_s, int(self.ucfg.base_dock_radius * SCALE), 2)
        base_label = self._font.render("BASE", True, _BASE_COLOR)
        screen.blit(base_label, (base_s[0] - 15, base_s[1] - 18))

        # ---- POIs ----
        for poi in world.pois:
            sx, sy = self._w2s(poi.pos)
            if not poi.discovered:
                color = _POI_UNDISCOVERED_COLOR
                pygame.draw.rect(screen, color, (sx - 5, sy - 5, 10, 10), 2)
            else:
                t = poi.inspection_progress
                r = int(_POI_UNDISCOVERED_COLOR[0] * (1 - t) + _POI_INSPECTED_COLOR[0] * t)
                g = int(_POI_UNDISCOVERED_COLOR[1] * (1 - t) + _POI_INSPECTED_COLOR[1] * t)
                b = int(_POI_UNDISCOVERED_COLOR[2] * (1 - t) + _POI_INSPECTED_COLOR[2] * t)
                pygame.draw.circle(screen, (r, g, b), (sx, sy), 6)
                if poi.delivered:
                    pygame.draw.circle(screen, (255, 255, 255), (sx, sy), 9, 2)
                # Progress arc
                if t > 0 and t < 1:
                    import math
                    end_angle = -math.pi / 2 + 2 * math.pi * t
                    pygame.draw.arc(
                        screen, (255, 255, 255),
                        (sx - 8, sy - 8, 16, 16),
                        -math.pi / 2, end_angle, 2
                    )

        # ---- Comm network overlay ----
        if self.show_comm_net:
            self._draw_comm_net(screen, pygame, states, world)

        # ---- UAVs ----
        role_map = {}
        if role_tracker is not None:
            for i in range(self.wcfg.n_agents):
                counters = np.array([
                    role_tracker.episode_cells_discovered[i],
                    role_tracker.episode_inspections[i],
                    role_tracker.episode_relays[i],
                ])
                total = counters.sum()
                if total > 0:
                    from emerge_uav_sim.analysis.role_tracker import ROLE_NAMES
                    role_map[i] = ROLE_NAMES[int(np.argmax(counters))]
                else:
                    role_map[i] = "unknown"

        for i, s in enumerate(states):
            if not s.alive:
                continue
            sx, sy = self._w2s(s.pos)
            role = role_map.get(i, "unknown")
            color = _ROLE_COLORS.get(role, _ROLE_COLORS["unknown"])

            pygame.draw.circle(screen, color, (sx, sy), 7)
            pygame.draw.circle(screen, (255, 255, 255), (sx, sy), 7, 1)

            # Velocity arrow
            vel_len = np.linalg.norm(s.vel)
            if vel_len > 0.1:
                vel_dir = s.vel / vel_len
                tip_w = s.pos + vel_dir * (vel_len / self.ucfg.max_speed) * 10
                tip_s = self._w2s(tip_w)
                pygame.draw.line(screen, (255, 255, 255), (sx, sy), tip_s, 2)

            # Battery bar
            bar_w = 14
            bar_h = 3
            bx = sx - bar_w // 2
            by = sy - 12
            pygame.draw.rect(screen, (60, 60, 60), (bx, by, bar_w, bar_h))
            fill = max(0, int(s.battery * bar_w))
            bat_color = (80, 200, 80) if s.battery > 0.4 else (200, 180, 50) if s.battery > 0.2 else (200, 50, 50)
            pygame.draw.rect(screen, bat_color, (bx, by, fill, bar_h))

            # Agent index label
            label = self._font.render(str(i), True, (255, 255, 255))
            screen.blit(label, (sx + 8, sy - 6))

        # ---- HUD ----
        hud_y = int(self.wcfg.height * SCALE) + 2 * WINDOW_PADDING + 8
        step_txt = self._font.render(f"Step: {step}", True, _HUD_TEXT_COLOR)
        screen.blit(step_txt, (10, hud_y))

        alive_count = sum(s.alive for s in states)
        alive_txt = self._font.render(f"Alive: {alive_count}/{self.wcfg.n_agents}", True, _HUD_TEXT_COLOR)
        screen.blit(alive_txt, (110, hud_y))

        n_insp = world.n_inspected
        insp_txt = self._font.render(
            f"POIs: {n_insp}/{len(world.pois)}", True, _HUD_TEXT_COLOR
        )
        screen.blit(insp_txt, (230, hud_y))

        cov_txt = self._font.render(
            f"Coverage: {world.coverage_fraction*100:.1f}%", True, _HUD_TEXT_COLOR
        )
        screen.blit(cov_txt, (340, hud_y))

        if hasattr(world, 'delivered_coverage_grid'):
            del_cov = world.delivered_fraction * 100
            del_poi = world.n_delivered_pois
            del_txt = self._font.render(
                f"Delivered: {del_cov:.1f}% / POIs: {del_poi}/{len(world.pois)}",
                True, (100, 220, 100)
            )
            screen.blit(del_txt, (10, hud_y + 18))

        net_label = "[N] Net: ON " if self.show_comm_net else "[N] Net: off"
        net_color = _NET_REACH_RING if self.show_comm_net else (100, 100, 120)
        net_txt = self._font.render(net_label, True, net_color)
        screen.blit(net_txt, (510, hud_y))

        fps = _FPS_LEVELS[self._fps_idx]
        speed_txt = self._font.render(
            f"[↑↓] {fps}fps  [SPACE] pause", True, (140, 140, 160)
        )
        screen.blit(speed_txt, (self.win_w - speed_txt.get_width() - 10, hud_y))

        pygame.display.flip()
        self._clock.tick(fps)

        # Return rgb_array if needed
        return pygame.surfarray.array3d(screen).transpose(1, 0, 2)

    def _draw_comm_net(self, screen, pygame, states, world) -> None:
        """
        Rich comm-net overlay (toggle with N key). All links respect LOS (obstacles block comm).

        Layers (back → front):
        1. Comm-range halos — faint circles showing each agent's reach.
        2. Agent↔agent links (within range + LOS unblocked):
              teal   — both endpoints reach base
              blue   — one endpoint reaches base
              gray   — isolated link
              amber  — relay-flagged message
        3. Agent↔base lines for agents directly connected to base (bright teal).
        4. Green ring on every UAV reachable from base.
        """
        from emerge_uav_sim.core.comm import has_los

        n = len(states)
        positions = np.array([s.pos for s in states])
        alive = np.array([s.alive for s in states])
        comm_r = self.ucfg.comm_range
        base = self.wcfg.base_pos
        obstacles = world.obstacles

        # --- 1. Comm-range halos ---
        halo_surf = pygame.Surface((self.win_w, self.win_h), pygame.SRCALPHA)
        r_px = int(comm_r * SCALE)
        for i in range(n):
            if not alive[i]:
                continue
            cx, cy = self._w2s(positions[i])
            pygame.draw.circle(halo_surf, _NET_HALO_FILL, (cx, cy), r_px)
            pygame.draw.circle(halo_surf, _NET_HALO_EDGE, (cx, cy), r_px, 1)
        screen.blit(halo_surf, (0, 0))

        # --- BFS with LOS: which agents can reach base? ---
        reachable = set()
        for i in range(n):
            if alive[i] and np.linalg.norm(positions[i] - base) < comm_r:
                if has_los(positions[i], base, obstacles):
                    reachable.add(i)
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if not alive[i] or i in reachable:
                    continue
                for j in reachable:
                    if np.linalg.norm(positions[i] - positions[j]) < comm_r:
                        if has_los(positions[i], positions[j], obstacles):
                            reachable.add(i)
                            changed = True
                            break

        # --- 2. Agent↔agent links ---
        link_surf = pygame.Surface((self.win_w, self.win_h), pygame.SRCALPHA)
        for i in range(n):
            if not alive[i]:
                continue
            relay_i = len(states[i].message) > 4 and states[i].message[4] > 0.5
            for j in range(i + 1, n):
                if not alive[j]:
                    continue
                if np.linalg.norm(positions[i] - positions[j]) >= comm_r:
                    continue
                if not has_los(positions[i], positions[j], obstacles):
                    continue
                relay_j = len(states[j].message) > 4 and states[j].message[4] > 0.5
                pi = self._w2s(positions[i])
                pj = self._w2s(positions[j])
                if relay_i or relay_j:
                    pygame.draw.line(link_surf, _NET_LINK_RELAY, pi, pj, 2)
                elif i in reachable and j in reachable:
                    pygame.draw.line(link_surf, _NET_LINK_BOTH, pi, pj, 2)
                elif i in reachable or j in reachable:
                    pygame.draw.line(link_surf, _NET_LINK_ONE, pi, pj, 1)
                else:
                    pygame.draw.line(link_surf, _NET_LINK_NONE, pi, pj, 1)
        screen.blit(link_surf, (0, 0))

        # --- 3. Lines from directly-connected agents to base ---
        base_s = self._w2s(base)
        for i in range(n):
            if not alive[i] or i not in reachable:
                continue
            if np.linalg.norm(positions[i] - base) < comm_r:
                pygame.draw.line(screen, _NET_REACH_RING, self._w2s(positions[i]), base_s, 2)

        # --- 4. Base-reachability ring on UAVs ---
        for i in reachable:
            if not alive[i]:
                continue
            sx, sy = self._w2s(positions[i])
            pygame.draw.circle(screen, _NET_REACH_RING, (sx, sy), 11, 2)

    def close(self):
        if self._status_panel is not None:
            self._status_panel.close()
            self._status_panel = None
        if self._pygame_init:
            self._pygame.quit()
            self._pygame_init = False
            self._screen = None
