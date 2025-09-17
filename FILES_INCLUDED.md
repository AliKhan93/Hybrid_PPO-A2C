# Files Included in This Backup

This backup contains only the essential custom files needed to run the Genuine Hybrid PPO-A2C training script. Standard Isaac Lab files are not included.

## Directory Structure

```
Hybrid_PPO_A2C/
├── README.md                                    # This documentation
├── requirements.txt                             # Python dependencies
├── install.sh                                   # Installation script
├── FILES_INCLUDED.md                           # This file
├── scripts/
│   └── reinforcement_learning/
│       └── skrl/
│           ├── train_genuine_hybrid_ppo_a2c.py      # Main training script
│           ├── play_genuine_hybrid_ppo_a2c.py       # Playback script
│           └── evaluate_genuine_hybrid_ppo_a2c.py   # Evaluation script
└── source/
    ├── isaaclab_rl/
    │   └── isaaclab_rl/
    │       ├── __init__.py                          # Package initialization
    │       └── genuine_hybrid_ppo_a2c.py            # Core algorithm implementation
    └── isaaclab_tasks/
        └── isaaclab_tasks/
            └── manager_based/
                └── locomotion/
                    └── velocity/
                        └── config/
                            └── h1/
                                ├── __init__.py      # Environment registration
                                └── agents/
                                    └── skrl_genuine_hybrid_ppo_a2c_cfg.yaml  # Training configuration
```

## File Descriptions

### Core Algorithm Files
- **`genuine_hybrid_ppo_a2c.py`**: Main implementation of the hybrid PPO-A2C algorithm
- **`__init__.py`** (isaaclab_rl): Package initialization with exports

### Configuration Files
- **`skrl_genuine_hybrid_ppo_a2c_cfg.yaml`**: Training configuration with hyperparameters
- **`__init__.py`** (h1 config): Environment registration for the H1 robot

### Scripts
- **`train_genuine_hybrid_ppo_a2c.py`**: Main training script with CLI arguments
- **`play_genuine_hybrid_ppo_a2c.py`**: Script to play trained models
- **`evaluate_genuine_hybrid_ppo_a2c.py`**: Comprehensive evaluation script

### Documentation
- **`README.md`**: Complete usage guide and documentation
- **`requirements.txt`**: Python package dependencies
- **`install.sh`**: Automated installation script
- **`FILES_INCLUDED.md`**: This file listing

## What's NOT Included

This backup intentionally excludes:
- Standard Isaac Lab files (available in main installation)
- Training logs and checkpoints
- Temporary files and caches
- Documentation files not essential for deployment
- Test files and development utilities
- Large asset files (URDFs, meshes, etc.)

## Total Size

The backup is minimal and contains only the essential custom code, making it easy to share and deploy.

## Installation

Use the provided `install.sh` script or manually copy files to your Isaac Lab installation following the README.md instructions.
