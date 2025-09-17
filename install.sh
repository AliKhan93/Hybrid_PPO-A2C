#!/bin/bash

# Installation script for Genuine Hybrid PPO-A2C backup
# This script helps users install the custom files into their Isaac Lab installation

set -e

echo "=== Genuine Hybrid PPO-A2C Installation Script ==="
echo ""

# Check if Isaac Lab path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <isaac_lab_path>"
    echo "Example: $0 /path/to/isaaclab"
    echo ""
    echo "Please provide the path to your Isaac Lab installation."
    exit 1
fi

ISAAC_LAB_PATH="$1"

# Check if Isaac Lab path exists
if [ ! -d "$ISAAC_LAB_PATH" ]; then
    echo "Error: Isaac Lab path '$ISAAC_LAB_PATH' does not exist."
    exit 1
fi

# Check if it looks like an Isaac Lab installation
if [ ! -d "$ISAAC_LAB_PATH/source" ] || [ ! -d "$ISAAC_LAB_PATH/scripts" ]; then
    echo "Error: '$ISAAC_LAB_PATH' does not appear to be an Isaac Lab installation."
    echo "Expected directories 'source' and 'scripts' not found."
    exit 1
fi

echo "Installing Genuine Hybrid PPO-A2C files to: $ISAAC_LAB_PATH"
echo ""

# Create backup of existing files (if any)
BACKUP_DIR="$ISAAC_LAB_PATH/backup_$(date +%Y%m%d_%H%M%S)"
echo "Creating backup of existing files in: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Backup existing files
if [ -f "$ISAAC_LAB_PATH/source/isaaclab_rl/isaaclab_rl/genuine_hybrid_ppo_a2c.py" ]; then
    echo "Backing up existing genuine_hybrid_ppo_a2c.py"
    cp "$ISAAC_LAB_PATH/source/isaaclab_rl/isaaclab_rl/genuine_hybrid_ppo_a2c.py" "$BACKUP_DIR/"
fi

if [ -f "$ISAAC_LAB_PATH/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/agents/skrl_genuine_hybrid_ppo_a2c_cfg.yaml" ]; then
    echo "Backing up existing skrl_genuine_hybrid_ppo_a2c_cfg.yaml"
    mkdir -p "$BACKUP_DIR/agents"
    cp "$ISAAC_LAB_PATH/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/agents/skrl_genuine_hybrid_ppo_a2c_cfg.yaml" "$BACKUP_DIR/agents/"
fi

# Install custom RL algorithm
echo "Installing custom RL algorithm..."
cp -r source/isaaclab_rl/isaaclab_rl/* "$ISAAC_LAB_PATH/source/isaaclab_rl/isaaclab_rl/"

# Install configuration files
echo "Installing configuration files..."
mkdir -p "$ISAAC_LAB_PATH/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/agents"
cp source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/agents/skrl_genuine_hybrid_ppo_a2c_cfg.yaml "$ISAAC_LAB_PATH/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/agents/"

# Update the __init__.py file to register the new environment
echo "Updating environment registration..."
cp source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/__init__.py "$ISAAC_LAB_PATH/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/"

# Install training scripts
echo "Installing training scripts..."
mkdir -p "$ISAAC_LAB_PATH/scripts/reinforcement_learning/skrl"
cp scripts/reinforcement_learning/skrl/train_genuine_hybrid_ppo_a2c.py "$ISAAC_LAB_PATH/scripts/reinforcement_learning/skrl/"
cp scripts/reinforcement_learning/skrl/play_genuine_hybrid_ppo_a2c.py "$ISAAC_LAB_PATH/scripts/reinforcement_learning/skrl/"
cp scripts/reinforcement_learning/skrl/evaluate_genuine_hybrid_ppo_a2c.py "$ISAAC_LAB_PATH/scripts/reinforcement_learning/skrl/"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Files installed successfully!"
echo "Backup of existing files created in: $BACKUP_DIR"
echo ""
echo "You can now run the training script:"
echo "  cd $ISAAC_LAB_PATH"
echo "  python scripts/reinforcement_learning/skrl/train_genuine_hybrid_ppo_a2c.py --task Isaac-Velocity-Rough-H1-v0"
echo ""
echo "For more information, see the README.md file."
