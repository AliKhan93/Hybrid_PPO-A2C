# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train genuine hybrid PPO-A2C agent with skrl.

This script implements a comprehensive hybrid reinforcement learning approach that genuinely
combines Proximal Policy Optimization (PPO) and Advantage Actor-Critic (A2C) algorithms
for improved training stability and sample efficiency on the H1 rough terrain
locomotion task.

Key Features:
- Genuine integration of PPO and A2C algorithms
- Multiple training modes (alternating, mixed, adaptive)
- Comprehensive evaluation metrics and monitoring
- Robustness features including gradient clipping and early stopping
- Hyperparameter tuning guidelines and edge case handling
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import logging
from datetime import datetime
from tqdm import tqdm
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train a genuine hybrid PPO-A2C agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="GENUINE_HYBRID_PPO_A2C",
    choices=["GENUINE_HYBRID_PPO_A2C"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument(
    "--training_mode",
    type=str,
    default="adaptive",
    choices=["alternate", "mixed", "adaptive"],
    help="Hybrid training mode: alternate, mixed, or adaptive.",
)
parser.add_argument(
    "--ppo_weight",
    type=float,
    default=0.7,
    help="Initial weight for PPO component (0.0-1.0).",
)
parser.add_argument(
    "--a2c_weight",
    type=float,
    default=0.3,
    help="Initial weight for A2C component (0.0-1.0).",
)
parser.add_argument(
    "--adaptive_weighting",
    action="store_true",
    default=True,
    help="Enable adaptive weight adjustment.",
)
parser.add_argument(
    "--verbose_logging",
    action="store_true",
    default=False,
    help="Enable verbose logging.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import random
import torch
import numpy as np
from typing import Dict, Any

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.genuine_hybrid_ppo_a2c import (
    GenuineHybridPPOA2C,
    create_genuine_hybrid_models,
    create_genuine_hybrid_memory,
    HybridMetrics
)
from skrl.utils.runner.torch import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_genuine_hybrid_ppo_a2c_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with genuine hybrid PPO-A2C agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["ppo_rollouts"]
    
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # Override hybrid-specific parameters from CLI
    if hasattr(args_cli, 'training_mode'):
        agent_cfg["agent"]["training_mode"] = args_cli.training_mode
    if hasattr(args_cli, 'ppo_weight'):
        agent_cfg["agent"]["ppo_weight"] = args_cli.ppo_weight
    if hasattr(args_cli, 'a2c_weight'):
        agent_cfg["agent"]["a2c_weight"] = args_cli.a2c_weight
    if hasattr(args_cli, 'adaptive_weighting'):
        agent_cfg["agent"]["adaptive_weighting"] = args_cli.adaptive_weighting

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["genuine_hybrid_ppo_a2c"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner with proper checkpointing
    runner = Runner(env, agent_cfg)
    
    # Replace the agent with our custom hybrid agent
    agent = GenuineHybridPPOA2C(
        models=runner.agent.models,
        memory=runner.agent.memory,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        cfg=agent_cfg["agent"]
    )
    
    # Set verbose logging flag
    agent.verbose_logging = args_cli.verbose_logging
    
    # Initialize the agent (creates memory tensors)
    agent.init()
    
    
    # Replace the runner's agent with our custom one
    runner._agent = agent

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # Print hybrid training information
    print(f"[INFO] Starting genuine hybrid PPO-A2C training...")
    print(f"[INFO] Training Mode: {agent_cfg['agent'].get('training_mode', 'adaptive')}")
    print(f"[INFO] PPO Weight: {agent_cfg['agent'].get('ppo_weight', 0.7):.3f}")
    print(f"[INFO] A2C Weight: {agent_cfg['agent'].get('a2c_weight', 0.3):.3f}")
    print(f"[INFO] Adaptive Weighting: {agent_cfg['agent'].get('adaptive_weighting', True)}")
    print(f"[INFO] Performance Window: {agent_cfg['agent'].get('performance_window', 10)}")
    print(f"[INFO] Early Stopping Patience: {agent_cfg['agent'].get('early_stopping_patience', 100)}")
    print(f"[INFO] Gradient Clip Norm: {agent_cfg['agent'].get('gradient_clip_norm', 1.0)}")

    # run training
    runner.run()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
   