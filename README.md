# Emerge UAV Sim

A **multi-agent reinforcement learning environment** for studying emergent role specialization in teams of cooperative UAVs. Built on [PettingZoo](https://pettingzoo.farama.org/) and trained with **MAPPO** (Multi-Agent Proximal Policy Optimization), the simulator lets agents autonomously develop distinct, complementary roles — Explorer, Inspector, and Relay — purely through reward shaping.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluating a Trained Policy](#evaluating-a-trained-policy)
  - [Running a Policy with Visualization](#running-a-policy-with-visualization)
  - [Programmatic API](#programmatic-api)
- [Configuration](#configuration)
  - [YAML Config File](#yaml-config-file)
  - [Key Reward Parameters](#key-reward-parameters)
  - [PPO Hyperparameters](#ppo-hyperparameters)
- [Environment Details](#environment-details)
  - [Observation Space](#observation-space)
  - [Action Space](#action-space)
  - [Reward System](#reward-system)
  - [Task Modes](#task-modes)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Contributing](#contributing)

---

## Overview

Emerge UAV Sim simulates a team of UAVs operating in a 2-D continuous world. The agents must cooperate to:

1. **Explore** the world and build a shared coverage map.
2. **Inspect** Points of Interest (POIs) discovered during exploration.
3. **Relay** gathered information back to the base station through a multi-hop communication network.

The key research question is whether distinct agent roles emerge from a shared policy trained with MAPPO, without any explicit role assignment.

---

## Features

| Category | Details |
|---|---|
| **Framework** | PettingZoo `ParallelEnv` |
| **Agents** | 6 cooperative UAVs (configurable) |
| **Action Space** | Continuous 6-D (2-D acceleration + 4-D learned message) |
| **Observation Space** | Continuous 69-D per agent |
| **Training Algorithm** | MAPPO with shared parameters (CTDE) |
| **Physics** | 2-D continuous dynamics, battery drain, collision detection |
| **Communication** | Range-based multi-hop relay with line-of-sight checks |
| **Visualization** | Pygame renderer + Matplotlib training panel |
| **Parallelization** | N parallel rollout workers for faster training |

---

## Installation

**Requirements:** Python 3.9 or higher.

```bash
# Clone the repository
git clone https://github.com/christianpfitzner/emerge_uav_sim
cd emerge_uav_sim

# Install the base package
pip install -e .

# Install with training dependencies (PyTorch)
pip install -e ".[train]"

# Install with development/testing dependencies
pip install -e ".[dev]"
```

> **Note:** The `[train]` extra pulls in PyTorch (`>=2.0`). If you need a specific CUDA version, install PyTorch manually before running the command above (see [pytorch.org](https://pytorch.org/get-started/locally/)).

### Verify the installation

```bash
python examples/check_env.py
```

This runs the PettingZoo parallel API test (100 cycles) and confirms the environment is compliant.

---

## Quick Start

```bash
# Train for 200 k steps with a single worker
python examples/train_mappo.py --config configs/default.yaml

# Evaluate the resulting checkpoint
python examples/evaluate.py --load checkpoints/final.pt

# Watch the policy run in the Pygame window
python examples/run_policy.py --load checkpoints/final.pt
```

---

## Usage

### Training

#### Single worker (default)

```bash
python examples/train_mappo.py --config configs/default.yaml
```

#### Multiple parallel workers (recommended for speed)

```bash
python examples/train_mappo.py \
  --config configs/default.yaml \
  --n-workers 8 \
  --total-steps 500000
```

Each worker runs an independent environment; rollouts are aggregated before each network update. With 8 workers and `--n-steps 512` you get 4 096 steps per update (~4 episodes averaged).

#### Resume from a checkpoint

```bash
python examples/train_mappo.py \
  --config configs/default.yaml \
  --load checkpoints/ckpt_000100.pt \
  --total-steps 400000
```

#### Enable the delivery mechanic

```bash
python examples/train_mappo.py \
  --require-delivery \
  --reward-delivery-poi 10.0 \
  --connected-bonus 0.005 \
  --penalty-low-battery 5.0 \
  --low-battery-threshold 0.6
```

#### Full option reference

```
python examples/train_mappo.py --help
```

---

### Evaluating a Trained Policy

Run one or more episodes using the deterministic mean action (no sampling):

```bash
# Single episode with visualization
python examples/evaluate.py --load checkpoints/final.pt

# Five episodes, terminal statistics only
python examples/evaluate.py --load checkpoints/final.pt --episodes 5 --no-render
```

The script prints a per-agent summary including role classification, cells discovered, inspection steps, and messages relayed.

---

### Running a Policy with Visualization

```bash
python examples/run_policy.py --load checkpoints/final.pt --episodes 3
```

The Pygame window shows:

- **UAV positions** color-coded by emergent role (Explorer = orange, Inspector = red, Relay = blue)
- **Coverage overlay** (visited cells highlighted)
- **Communication network** (toggle with **N** key — shows range circles and relay links)
- **HUD** with step counter and battery levels

---

### Programmatic API

```python
import numpy as np
from emerge_uav_sim import UAVTeamEnv, SimConfig, WorldConfig, UAVConfig, TaskConfig

cfg = SimConfig(
    world=WorldConfig(
        n_agents=4,
        n_pois=6,
        n_obstacles=3,
        width=100.0,
        height=100.0,
    ),
    uav=UAVConfig(
        max_speed=5.0,
        comm_range=20.0,
        sensor_range=15.0,
    ),
    task=TaskConfig(
        max_steps=500,
        reward_coverage=0.1,
        reward_inspection=5.0,
        require_delivery=True,
    ),
    seed=42,
)

env = UAVTeamEnv(config=cfg, render_mode="human")

observations, infos = env.reset()

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()
```

---

## Configuration

All training parameters can be set via a **YAML file** or individual **CLI flags**. CLI flags always override the YAML file.

### YAML Config File

```bash
python examples/train_mappo.py --config configs/default.yaml
```

The bundled `configs/default.yaml` contains all available parameters with explanatory comments.

### Key Reward Parameters

| Parameter | Default | Description |
|---|---|---|
| `reward_coverage` | `0.1` | Reward per newly discovered grid cell |
| `reward_inspection` | `5.0` | Reward per fully inspected POI |
| `reward_relay` | `1.0` | Reward when a message reaches base via relay |
| `role_shaping_strength` | `0.5` | Scale of role-specific shaping bonuses |
| `require_delivery` | `false` | Only award rewards once data reaches base |
| `reward_delivery_poi` | `10.0` | Extra bonus per POI delivered to base |
| `connected_bonus` | `0.0` | Per-step bonus for being connected to base |
| `penalty_low_battery` | `0.0` | Per-step penalty when battery is below threshold |
| `low_battery_threshold` | `0.3` | Battery level that triggers the low-battery penalty |
| `reward_battery_return` | `2.0` | One-time bonus for docking before battery dies |
| `penalty_dead` | `-5.0` | Penalty when an agent's battery reaches zero |
| `penalty_same_cell` | `0.0` | Per-step penalty when agents share a grid cell |
| `exploration_bonus` | `0.0` | Per-step bonus proportional to distance from base |
| `spread_bonus` | `0.0` | Per-step team bonus for mean pairwise distance |
| `loitering_penalty` | `0.0` | Per-step penalty for idling near base |

### PPO Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `n_steps` | `512` | Rollout length per worker per update |
| `n_epochs` | `4` | PPO update epochs per rollout |
| `batch_size` | `128` | Mini-batch size |
| `lr` | `3e-4` | Adam learning rate |
| `gamma` | `0.99` | Discount factor |
| `gae_lambda` | `0.95` | GAE lambda |
| `clip_eps` | `0.2` | PPO clip epsilon |
| `entropy_coef` | `0.01` | Entropy regularisation coefficient |
| `hidden_dim` | `128` | Actor-critic network hidden layer size |

---

## Environment Details

### Observation Space

Each agent receives a 69-dimensional continuous observation:

| Slice | Dims | Content |
|---|---|---|
| Own state | 6 | `pos_x, pos_y, vel_x, vel_y, battery, connected_to_base` |
| Base direction | 3 | `dx, dy, distance` (normalised) |
| K=5 peers | 25 | `rel_pos_x, rel_pos_y, vel_x, vel_y, battery` × 5 |
| M=4 POIs | 16 | `rel_pos_x, rel_pos_y, inspection_progress, discovered` × 4 |
| Broadcast messages | 11 | 7 structured + 4 learned (mean-pooled from neighbors) |
| Memory | 8 | Personal coverage fraction, direction to nearest unvisited cell, base coverage fraction, direction to nearest cell unknown to base |

### Action Space

Each agent outputs a 6-dimensional continuous action:

| Slice | Dims | Range | Content |
|---|---|---|---|
| Acceleration | 2 | `[-2.0, 2.0]` | `accel_x, accel_y` |
| Learned message | 4 | `[-1.0, 1.0]` | Broadcast to neighbors |

### Reward System

Rewards are a combination of shared team rewards and individual shaping signals:

- **Coverage:** Shared reward per newly visited grid cell.
- **Inspection:** Shared reward when a POI is fully inspected (after `inspection_steps` steps in range).
- **Relay:** Per-agent reward when a message reaches base via the relay chain (with cooldown).
- **Role shaping:** Additional bonuses differentiate explorer, inspector, and relay behavior.
- **Battery management:** Penalties encourage agents to return to base before their battery dies.
- **Coordination penalties:** Discourage crowding on the same cell or redundant POI inspection.

### Task Modes

| Mode | Description |
|---|---|
| **Area Coverage + Inspection** (default) | POIs are pre-discovered. Agents earn rewards on inspection. |
| **Search and Report** (`--require-delivery`) | POIs are hidden until discovered. Rewards only count once data reaches base, driving the formation of a relay network. |

---

## Project Structure

```
emerge_uav_sim/
├── emerge_uav_sim/          # Main Python package
│   ├── envs/
│   │   └── uav_team_env.py  # PettingZoo ParallelEnv implementation
│   ├── core/
│   │   ├── uav.py           # UAV state dataclass
│   │   ├── world.py         # World model (grid, POIs, obstacles)
│   │   └── comm.py          # Communication system and relay logic
│   ├── config/
│   │   └── configs.py       # SimConfig, WorldConfig, UAVConfig, TaskConfig
│   ├── training/
│   │   ├── mappo.py         # MAPPO trainer
│   │   ├── parallel_trainer.py  # Multi-worker rollout collection
│   │   ├── networks.py      # Actor-critic network (PyTorch)
│   │   ├── buffer.py        # Rollout buffer
│   │   └── training_panel.py    # Real-time training visualization
│   ├── analysis/
│   │   └── role_tracker.py  # Role classification and behavior metrics
│   └── rendering/
│       ├── renderer.py      # Pygame renderer
│       └── status_panel.py  # Matplotlib statistics panel
├── examples/
│   ├── train_mappo.py       # Training entry point
│   ├── evaluate.py          # Policy evaluation
│   ├── run_policy.py        # Pygame policy visualization
│   ├── random_policy.py     # Random-agent baseline
│   └── check_env.py         # PettingZoo API validation
├── configs/
│   └── default.yaml         # Default training configuration
├── tests/
│   └── test_env.py          # pytest unit tests
└── pyproject.toml           # Package metadata and dependencies
```

---

## Testing

```bash
# Run the unit test suite
pytest tests/ -v

# Validate PettingZoo API compliance
python examples/check_env.py
```

The test suite covers observation and action shapes, step transition correctness, episode termination, reward finiteness, and battery/collision physics.

---

## Contributing

1. Fork the repository and create a feature branch.
2. Install the development dependencies: `pip install -e ".[dev]"`.
3. Make your changes and add tests where appropriate.
4. Ensure all tests pass: `pytest tests/ -v`.
5. Verify PettingZoo API compliance: `python examples/check_env.py`.
6. Open a pull request with a clear description of the changes.
