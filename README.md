# Hybrid PPO-A2C Reinforcement Learning for Mobile Robot Training

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-v2.1.0-green.svg)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## Overview

**Hybrid PPO-A2C** is a versatile reinforcement learning framework that provides highly efficient mobile robot training with state-of-the-art performance on rough terrain locomotion and adaptive learning dynamics through hybrid PPO-A2C algorithms.

This repo covers the hybrid algorithm implementation and training in Isaac Lab. **You should be able to train any mobile robot locomotion task on rough terrain, without tuning any parameters**.

## Installation

1. **Install Isaac Lab v2.1.0** by following the [installation guide](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/pip_installation.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

2. **Install Isaac Sim 4.5.0** by following the [Isaac Sim installation guide](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_workstation.html).

3. **Clone this repository** "(personally I did inside Isaac Lab directory)":

```bash
# Option 1: SSH
git clone git@github.com:AliKhan93/Hybrid_PPO-A2C.git

# Option 2: HTTPS
git clone https://github.com/AliKhan93/Hybrid_PPO-A2C.git
```

4. **Using a Python interpreter that has Isaac Lab installed**, install the library:

```bash
python -m pip install -e source/isaaclab_rl
# or use
./isaaclab.sh -p
```

## Policy Training

Train policy by the following command:

```bash
python scripts/reinforcement_learning/skrl/train_genuine_hybrid_ppo_a2c.py \
    --task Isaac-Velocity-Rough-H1-v0 \
    --num_envs 16 \
    --headless
```

## Policy Evaluation

Play the trained policy by the following command:

```bash
python scripts/reinforcement_learning/skrl/play_genuine_hybrid_ppo_a2c.py \
    --task Isaac-Velocity-Rough-H1-v0 \
    --checkpoint /path/to/checkpoint.pt \ 
    --num_envs 2
```

The checkpoint path can be found in the `logs/skrl/h1_rough_genuine_hybrid_ppo_a2c/` directory after training.


## Code Structure

Below is an overview of the code structure for this repository:

- **`source/isaaclab_rl/isaaclab_rl/genuine_hybrid_ppo_a2c.py`** This file contains the core hybrid PPO-A2C algorithm implementation. Below is a breakdown of the key components:
  - **`GenuineHybridPPOA2C`** Main hybrid agent class that combines PPO and A2C algorithms with adaptive weighting mechanism
  - **`GenuineHybridModel`** Shared neural network architecture with policy and value heads for efficient parameter usage
  - **`HybridMetrics`** Container for comprehensive training metrics including PPO/A2C losses, policy entropy, and adaptive weights
  - **`create_genuine_hybrid_models`** Factory function to create hybrid models with proper initialization
  - **`create_genuine_hybrid_memory`** Memory buffer creation optimized for both PPO and A2C training

- **`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/agents/skrl_genuine_hybrid_ppo_a2c_cfg.yaml`** Contains the hybrid algorithm hyperparameters configuration for the H1 rough terrain locomotion task.

- **`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/__init__.py`** Contains environment registration for the H1 robot with hybrid PPO-A2C configuration entry points.

- **`scripts/reinforcement_learning/skrl/`** Includes utility scripts for training policies, playing trained models, and evaluating algorithm performance:
  - **`train_genuine_hybrid_ppo_a2c.py`** Main training script with comprehensive CLI arguments and hybrid-specific parameters
  - **`play_genuine_hybrid_ppo_a2c.py`** Policy playback script for visualizing trained models
  - **`evaluate_genuine_hybrid_ppo_a2c.py`** Comprehensive evaluation script with performance metrics and statistical analysis

```bash
Hybrid_PPO_A2C/
├── scripts/reinforcement_learning/skrl/
│   ├── train_genuine_hybrid_ppo_a2c.py      # Main training script
│   ├── play_genuine_hybrid_ppo_a2c.py       # Policy playback
│   └── evaluate_genuine_hybrid_ppo_a2c.py   # Comprehensive evaluation
├── source/isaaclab_rl/isaaclab_rl/
│   └── genuine_hybrid_ppo_a2c.py            # Core algorithm implementation
└── source/isaaclab_tasks/.../config/h1/
    └── skrl_genuine_hybrid_ppo_a2c_cfg.yaml # Training configuration
```


Note: This repository contains only the essential custom files needed for the hybrid algorithm. Standard Isaac Lab files are not included and should be available in your main installation.
