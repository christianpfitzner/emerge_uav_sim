"""
Real-time training progress window — background thread, never blocks training.

Shows:
  • Episode Reward   (raw + smoothed moving average)
  • Policy Loss
  • Value Loss
  • Entropy
  • Steps/s throughput

Window stays open after training ends until the user closes it.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Optional
import numpy as np

_FIG_BG  = "#14141E"
_AX_BG   = "#1A1A28"
_TICK_C  = "#AAAACC"
_GRID_C  = "#303050"
_SPINE_C = "#404060"

_C_REWARD  = "#32C882"
_C_SMOOTH  = "#FFFFFF"
_C_POLICY  = "#FF8C00"
_C_VALUE   = "#3264DC"
_C_ENTROPY = "#AA64DC"
_C_BEST    = "#FFD700"


def _style_ax(ax, title: str):
    ax.set_facecolor(_AX_BG)
    ax.set_title(title, color="white", fontsize=9, pad=4)
    ax.tick_params(colors=_TICK_C, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(_SPINE_C)
    ax.grid(color=_GRID_C, linewidth=0.5, zorder=0)
    ax.set_xlabel("Update", color=_TICK_C, fontsize=7)


def _smooth(values: list, window: int) -> list:
    if len(values) < 2:
        return list(values)
    half = window // 2
    out = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(float(np.mean(values[lo:hi])))
    return out


class TrainingPanel:
    """
    Opens a matplotlib training-progress window in a background thread.

    Call ``update()`` after every training update (non-blocking).
    Call ``close()`` when training ends — the window stays open until
    the user closes it.
    """

    def __init__(self, total_steps: int, update_every: int = 1):
        self._total_steps = total_steps
        self._every = update_every

        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._stop   = threading.Event()
        self._ready  = threading.Event()
        self._closed = threading.Event()

        # History stored in worker thread only
        self._active = False
        self._t_start = time.time()

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        update_idx: int,
        env_steps: int,
        mean_reward: float,
        losses: dict,
    ) -> None:
        if not self._active or self._stop.is_set():
            return
        if update_idx % self._every != 0:
            return
        elapsed = time.time() - self._t_start
        steps_per_sec = env_steps / elapsed if elapsed > 0 else 0.0
        snap = {
            "update":    update_idx,
            "steps":     env_steps,
            "reward":    mean_reward,
            "policy":    losses.get("policy", float("nan")),
            "value":     losses.get("value",  float("nan")),
            "entropy":   losses.get("entropy", float("nan")),
            "sps":       steps_per_sec,
        }
        try:
            self._q.put_nowait(snap)
        except queue.Full:
            pass

    def close(self) -> None:
        if not self._active:
            return
        self._stop.set()
        try:
            self._q.put_nowait({"_close": True})
        except queue.Full:
            pass
        self._closed.wait(timeout=600)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker(self) -> None:
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
            fig = plt.figure("MAPPO Training", figsize=(12, 7))
            fig.patch.set_facecolor(_FIG_BG)
            try:
                fig.canvas.manager.set_window_title("MAPPO Training Progress")
            except Exception:
                pass

            gs = gridspec.GridSpec(
                2, 3, figure=fig,
                hspace=0.52, wspace=0.35,
                left=0.07, right=0.97,
                top=0.91, bottom=0.09,
                height_ratios=[1.6, 1.0],
            )

            # Top: reward (spans all 3 cols)
            ax_rew = fig.add_subplot(gs[0, :])
            _style_ax(ax_rew, "Episode Reward")

            # Bottom row
            ax_pol = fig.add_subplot(gs[1, 0])
            ax_val = fig.add_subplot(gs[1, 1])
            ax_ent = fig.add_subplot(gs[1, 2])
            _style_ax(ax_pol, "Policy Loss")
            _style_ax(ax_val, "Value Loss")
            _style_ax(ax_ent, "Entropy")

            # Header text
            ax_rew.set_title("Episode Reward", color="white", fontsize=9, pad=4)
            header = fig.text(
                0.5, 0.965, "Waiting for first update…",
                ha="center", va="top",
                fontsize=10, color=_TICK_C,
                fontfamily="monospace",
                transform=fig.transFigure,
            )

            fig.canvas.draw()
            fig.canvas.flush_events()
        except Exception:
            self._ready.set()
            return

        self._active = True
        self._ready.set()

        # History
        updates: list  = []
        rewards: list  = []
        policy:  list  = []
        value:   list  = []
        entropy: list  = []
        best_reward    = float("-inf")
        best_update    = None
        smooth_window  = 20

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

            # Accumulate
            u = data["update"]
            r = data["reward"]
            updates.append(u)
            rewards.append(r if not np.isnan(r) else (rewards[-1] if rewards else 0.0))
            policy.append(data["policy"])
            value.append(data["value"])
            entropy.append(data["entropy"])

            if not np.isnan(r) and r > best_reward:
                best_reward = r
                best_update = u

            sps  = data["sps"]
            step = data["steps"]
            pct  = 100.0 * step / self._total_steps if self._total_steps > 0 else 0.0
            header.set_text(
                f"Update {u:>6}  │  Steps {step:>9,}  ({pct:.1f}%)  │  "
                f"{sps:>6.0f} steps/s  │  Best reward: {best_reward:.3f}"
            )

            xs = updates

            # ---- Reward plot ----
            ax_rew.cla()
            _style_ax(ax_rew, "Episode Reward")
            ax_rew.plot(xs, rewards, color=_C_REWARD, lw=1.0, alpha=0.4, zorder=2)
            if len(rewards) >= 3:
                sm = _smooth(rewards, smooth_window)
                ax_rew.plot(xs, sm, color=_C_SMOOTH, lw=1.8, zorder=3,
                            label=f"MA-{smooth_window}")
            if best_update is not None:
                ax_rew.axvline(best_update, color=_C_BEST, lw=1.0,
                               ls="--", alpha=0.7, zorder=1)
                ax_rew.text(best_update, ax_rew.get_ylim()[1] if ax_rew.get_ylim()[1] != 0 else 1,
                            f" best={best_reward:.2f}",
                            color=_C_BEST, fontsize=7, va="top")
            ax_rew.axhline(0, color="#606060", lw=0.5)
            if len(rewards) >= 3:
                ax_rew.legend(loc="upper left", fontsize=7, framealpha=0.3,
                              labelcolor="white", facecolor=_FIG_BG,
                              edgecolor=_SPINE_C)

            # ---- Loss plots ----
            for ax, data_y, color, title in [
                (ax_pol, policy,  _C_POLICY,  "Policy Loss"),
                (ax_val, value,   _C_VALUE,   "Value Loss"),
                (ax_ent, entropy, _C_ENTROPY, "Entropy"),
            ]:
                ax.cla()
                _style_ax(ax, title)
                valid = [(x, y) for x, y in zip(xs, data_y) if not np.isnan(y)]
                if valid:
                    vx, vy = zip(*valid)
                    ax.plot(vx, vy, color=color, lw=1.2, zorder=3)
                    if len(vy) >= 3:
                        ax.plot(vx, _smooth(list(vy), smooth_window),
                                color="white", lw=1.0, alpha=0.5, zorder=4)

            try:
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            except Exception:
                break

        # Stay open
        try:
            plt.ioff()
            plt.show(block=True)
        except Exception:
            pass
        self._closed.set()
