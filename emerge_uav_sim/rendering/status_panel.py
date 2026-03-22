"""
Real-time status panel — runs in a background thread so it never blocks
the Pygame render loop.

Second window shows:
  • System overview  (step, alive, coverage, POIs)
  • Battery          per agent  (green / yellow / red + role badge)
  • Role activity    stacked bar  (explorer / inspector / relay)
  • Cumulative reward per agent
  • Coverage % over time
  • Team reward per step over time

The window stays open at the end of the episode until the user closes it.
"""
from __future__ import annotations

import queue
import threading
from typing import List, Optional
import numpy as np

_ROLE_HEX = {
    "explorer": "#FF8C00",
    "inspector": "#DC3232",
    "relay":     "#3264DC",
    "unknown":   "#B4B4B4",
}
_FIG_BG  = "#14141E"
_AX_BG   = "#1A1A28"
_TICK_C  = "#AAAACC"
_GRID_C  = "#303050"
_SPINE_C = "#404060"


def _style_ax(ax):
    ax.set_facecolor(_AX_BG)
    ax.tick_params(colors=_TICK_C, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(_SPINE_C)
    ax.grid(axis="y", color=_GRID_C, linewidth=0.5, zorder=0)


# ---------------------------------------------------------------------------
# Snapshot helper – copies everything before handing to the other thread
# ---------------------------------------------------------------------------

def _snapshot(states, world, step: int, role_tracker, step_rewards) -> dict:
    n = len(states)
    cells = np.zeros(n)
    insp  = np.zeros(n)
    relay = np.zeros(n)
    if role_tracker is not None:
        cells = role_tracker.episode_cells_discovered.copy()
        insp  = role_tracker.episode_inspections.copy()
        relay = role_tracker.episode_relays.copy()

    return {
        "step":        step,
        "batteries":   np.array([s.battery for s in states]),
        "alive":       np.array([s.alive   for s in states]),
        "coverage":    float(world.coverage_fraction) * 100.0,
        "n_inspected": int(world.n_inspected),
        "n_pois":      len(world.pois),
        "cells":       cells,
        "insp":        insp,
        "relay":       relay,
        "rewards":     step_rewards.copy() if step_rewards is not None else np.zeros(n),
    }


# ---------------------------------------------------------------------------
# StatusPanel
# ---------------------------------------------------------------------------

class StatusPanel:
    """
    Opens a matplotlib window in a background thread.
    Call ``update()`` from the main loop (non-blocking – drops frames when busy).
    Call ``close()`` at the end to show the final state and wait for the user
    to close the window.
    """

    def __init__(self, n_agents: int, update_every: int = 15):
        self.n = n_agents
        self._every = update_every

        self._q: queue.Queue = queue.Queue(maxsize=1)   # 1-frame buffer
        self._stop   = threading.Event()
        self._ready  = threading.Event()   # set once figure is created
        self._closed = threading.Event()   # set when plt.show() returns

        self._cum_rewards = np.zeros(n_agents)
        self._hist_len = 300
        # Shared history – only the worker thread writes/reads these
        self._steps: List[int]   = []
        self._cov:   List[float] = []
        self._rew:   List[float] = []

        self._active = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        # Wait briefly for the thread to confirm matplotlib is available
        self._ready.wait(timeout=5.0)

    # ------------------------------------------------------------------
    # Public API (called from the main / Pygame thread)
    # ------------------------------------------------------------------

    def update(
        self,
        states,
        world,
        step: int,
        role_tracker,
        step_rewards: Optional[np.ndarray] = None,
    ) -> None:
        if not self._active or self._stop.is_set():
            return
        if step % self._every != 0:
            return
        snap = _snapshot(states, world, step, role_tracker, step_rewards)
        try:
            self._q.put_nowait(snap)
        except queue.Full:
            pass  # panel busy → drop frame, no blocking

    def reset(self) -> None:
        """Clear per-episode accumulators (call at env.reset)."""
        self._cum_rewards[:] = 0.0
        # history cleared inside worker via a sentinel
        try:
            self._q.put_nowait({"_reset": True})
        except queue.Full:
            pass

    def close(self) -> None:
        """
        Signal the worker to stop accepting new frames, then block until
        the user closes the matplotlib window (or 10 min timeout).
        """
        if not self._active:
            return
        self._stop.set()
        try:
            self._q.put_nowait({"_close": True})
        except queue.Full:
            pass
        self._closed.wait(timeout=600)

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        # --- matplotlib init (all matplotlib lives in this thread) ---
        try:
            import matplotlib
            for backend in ("TkAgg", "Qt5Agg", "Qt6Agg", "GTK3Agg"):
                try:
                    matplotlib.use(backend)
                    break
                except Exception:
                    continue
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except Exception:
            self._ready.set()
            return

        try:
            plt.ion()
            fig = plt.figure("UAV Status Panel", figsize=(11, 7.5))
            fig.patch.set_facecolor(_FIG_BG)
            try:
                fig.canvas.manager.set_window_title("UAV Status Panel")
            except Exception:
                pass

            gs = gridspec.GridSpec(
                5, 2, figure=fig,
                hspace=0.60, wspace=0.38,
                left=0.07, right=0.97,
                top=0.94, bottom=0.06,
                height_ratios=[0.5, 1, 1, 1, 1.2],
            )

            # Row 0 – system info
            ax_sys = fig.add_subplot(gs[0, :])
            ax_sys.set_facecolor(_FIG_BG)
            ax_sys.axis("off")
            sys_text = ax_sys.text(
                0.5, 0.5, "Waiting for first update…",
                ha="center", va="center",
                fontsize=12, color="white",
                fontfamily="monospace",
                transform=ax_sys.transAxes,
            )

            ax_bat  = fig.add_subplot(gs[1, :])
            ax_role = fig.add_subplot(gs[2, :])
            ax_rew  = fig.add_subplot(gs[3, :])
            ax_cov  = fig.add_subplot(gs[4, 0])
            ax_hist = fig.add_subplot(gs[4, 1])

            for ax, title in [
                (ax_bat,  "Battery"),
                (ax_role, "Role Activity  (episode cumulative, normalised)"),
                (ax_rew,  "Cumulative Reward per Agent"),
            ]:
                _style_ax(ax)
                ax.set_title(title, color="white", fontsize=9, pad=3)

            for ax, title in [
                (ax_cov,  "Coverage % over time"),
                (ax_hist, "Team Reward / Step"),
            ]:
                _style_ax(ax)
                ax.grid(axis="both", color=_GRID_C, linewidth=0.5, zorder=0)
                ax.set_title(title, color="white", fontsize=9, pad=3)
                ax.set_xlabel("Step", color=_TICK_C, fontsize=7)

            fig.canvas.draw()
            fig.canvas.flush_events()

        except Exception:
            self._ready.set()
            return

        self._active = True
        self._ready.set()

        n = self.n
        xs = np.arange(n)
        xlabels = [f"UAV {i}" for i in range(n)]

        # --- event loop ---
        while not self._stop.is_set():
            try:
                data = self._q.get(timeout=0.05)
            except queue.Empty:
                try:
                    fig.canvas.flush_events()
                except Exception:
                    break
                continue

            if data.get("_close"):
                break
            if data.get("_reset"):
                self._cum_rewards[:] = 0.0
                self._steps.clear()
                self._cov.clear()
                self._rew.clear()
                continue

            # --- accumulate history ---
            step = data["step"]
            self._cum_rewards += data["rewards"]
            self._steps.append(step)
            self._cov.append(data["coverage"])
            self._rew.append(float(data["rewards"].sum()))
            if len(self._steps) > self._hist_len:
                self._steps.pop(0)
                self._cov.pop(0)
                self._rew.pop(0)

            # --- role inference ---
            roles: List[str] = []
            for i in range(n):
                c = np.array([data["cells"][i], data["insp"][i], data["relay"][i]])
                if c.sum() > 0:
                    from emerge_uav_sim.analysis.role_tracker import ROLE_NAMES
                    roles.append(ROLE_NAMES[int(np.argmax(c))])
                else:
                    roles.append("unknown")

            # --- system text ---
            alive_n = int(data["alive"].sum())
            sys_text.set_text(
                f"Step: {step:>5}  │  Alive: {alive_n}/{n}  │  "
                f"Coverage: {data['coverage']:5.1f}%  │  "
                f"POIs inspected: {data['n_inspected']}/{data['n_pois']}"
            )

            # --- battery ---
            ax_bat.cla()
            _style_ax(ax_bat)
            ax_bat.set_title("Battery", color="white", fontsize=9, pad=3)
            bats = data["batteries"]
            bat_colors = []
            for i, (alive, b) in enumerate(zip(data["alive"], bats)):
                if not alive:
                    bat_colors.append("#404040")
                elif b > 0.4:
                    bat_colors.append("#50C850")
                elif b > 0.2:
                    bat_colors.append("#C8B432")
                else:
                    bat_colors.append("#C83232")
            ax_bat.bar(xs, bats, color=bat_colors, width=0.7, zorder=3)
            ax_bat.axhline(0.2, color="#C83232", lw=0.8, ls="--", alpha=0.5)
            ax_bat.axhline(0.4, color="#C8B432", lw=0.8, ls="--", alpha=0.5)
            ax_bat.set_ylim(0, 1.12)
            ax_bat.set_xticks(xs)
            ax_bat.set_xticklabels(xlabels, fontsize=7, color=_TICK_C)
            ax_bat.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax_bat.yaxis.set_tick_params(labelsize=7, colors=_TICK_C)
            for i, (b, role, alive) in enumerate(zip(bats, roles, data["alive"])):
                if alive:
                    rc = _ROLE_HEX.get(role, "#B4B4B4")
                    ax_bat.text(i, min(b + 0.06, 1.08),
                                role[0].upper(), ha="center", va="bottom",
                                fontsize=8, color=rc, fontweight="bold")
                else:
                    ax_bat.text(i, 0.06, "✕", ha="center", va="bottom",
                                fontsize=8, color="#888888")

            # --- role activity ---
            ax_role.cla()
            _style_ax(ax_role)
            ax_role.set_title(
                "Role Activity  (episode cumulative, normalised)",
                color="white", fontsize=9, pad=3,
            )
            c_norm = data["cells"]
            i_norm = data["insp"]
            r_norm = data["relay"]
            denom  = max((c_norm + i_norm + r_norm).max(), 1.0)
            ax_role.bar(xs, c_norm / denom,
                        color="#FF8C00", width=0.7, label="Explorer", zorder=3)
            ax_role.bar(xs, i_norm / denom,
                        bottom=c_norm / denom,
                        color="#DC3232", width=0.7, label="Inspector", zorder=3)
            ax_role.bar(xs, r_norm / denom,
                        bottom=(c_norm + i_norm) / denom,
                        color="#3264DC", width=0.7, label="Relay", zorder=3)
            ax_role.legend(loc="upper right", fontsize=6, framealpha=0.3,
                           labelcolor="white", facecolor=_FIG_BG, edgecolor=_SPINE_C)
            ax_role.set_ylim(0, 1.05)
            ax_role.set_xticks(xs)
            ax_role.set_xticklabels(xlabels, fontsize=7, color=_TICK_C)

            # --- cumulative reward ---
            ax_rew.cla()
            _style_ax(ax_rew)
            ax_rew.set_title("Cumulative Reward per Agent",
                             color="white", fontsize=9, pad=3)
            rew_colors = [_ROLE_HEX.get(r, "#B4B4B4") for r in roles]
            ax_rew.bar(xs, self._cum_rewards, color=rew_colors, width=0.7, zorder=3)
            ax_rew.axhline(0, color="#808080", lw=0.5)
            ax_rew.set_xticks(xs)
            ax_rew.set_xticklabels(xlabels, fontsize=7, color=_TICK_C)
            ylo = min(0.0, float(self._cum_rewards.min()))
            yhi = max(1.0, float(self._cum_rewards.max()))
            margin = max((yhi - ylo) * 0.12, 0.5)
            ax_rew.set_ylim(ylo - margin, yhi + margin)

            # --- coverage history ---
            ax_cov.cla()
            _style_ax(ax_cov)
            ax_cov.grid(axis="both", color=_GRID_C, linewidth=0.5, zorder=0)
            ax_cov.set_title("Coverage % over time", color="white", fontsize=9, pad=3)
            ax_cov.set_xlabel("Step", color=_TICK_C, fontsize=7)
            ax_cov.set_ylim(0, 100)
            if self._steps:
                ax_cov.plot(self._steps, self._cov, color="#32C882", lw=1.5, zorder=3)
                ax_cov.fill_between(self._steps, self._cov,
                                    alpha=0.15, color="#32C882")

            # --- step reward history ---
            ax_hist.cla()
            _style_ax(ax_hist)
            ax_hist.grid(axis="both", color=_GRID_C, linewidth=0.5, zorder=0)
            ax_hist.set_title("Team Reward / Step", color="white", fontsize=9, pad=3)
            ax_hist.set_xlabel("Step", color=_TICK_C, fontsize=7)
            if self._steps:
                ax_hist.plot(self._steps, self._rew, color="#AA64DC", lw=1.5, zorder=3)
                ax_hist.axhline(0, color="#808080", lw=0.5)

            try:
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            except Exception:
                break

        # --- stay open until user closes the window ---
        try:
            plt.ioff()
            plt.show(block=True)
        except Exception:
            pass

        self._closed.set()
