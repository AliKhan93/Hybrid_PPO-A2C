# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to evaluate genuine hybrid PPO-A2C agent with comprehensive metrics.

This script provides comprehensive evaluation of the genuine hybrid PPO-A2C agent
including performance metrics, training stability analysis, and comparison
with standard PPO and A2C baselines.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate genuine hybrid PPO-A2C agent.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint to evaluate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-H1-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to evaluate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory for results.")
parser.add_argument("--verbose_logging", action="store_true", default=False, help="Enable verbose logging.")

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
import torch
import random
from collections import defaultdict, deque

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

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.genuine_hybrid_ppo_a2c import (
    GenuineHybridPPOA2C,
    create_genuine_hybrid_models,
    create_genuine_hybrid_memory,
    HybridMetrics
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


class HybridEvaluator:
    """Comprehensive evaluator for genuine hybrid PPO-A2C agent."""
    
    def __init__(self, agent, env, num_episodes=100, verbose=False):
        """Initialize the evaluator.
        
        Args:
            agent: Trained hybrid agent
            env: Environment
            num_episodes: Number of episodes to evaluate
            verbose: Enable verbose logging
        """
        self.agent = agent
        self.env = env
        self.num_episodes = num_episodes
        self.verbose = verbose
        
        # Setup logging
        if verbose:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.WARNING)
        
        # Evaluation metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_metrics = []
        self.step_metrics = defaultdict(list)
        
        # Performance tracking
        self.best_episode_reward = float('-inf')
        self.worst_episode_reward = float('inf')
        self.total_steps = 0
    
    def evaluate(self) -> Dict[str, Any]:
        """Run comprehensive evaluation.
        
        Returns:
            Dictionary containing evaluation results
        """
        self.logger.info(f"Starting evaluation for {self.num_episodes} episodes...")
        
        # Reset environment
        obs, _ = self.env.reset()
        
        episode_count = 0
        current_episode_reward = 0
        current_episode_length = 0
        current_episode_metrics = defaultdict(list)
        
        while episode_count < self.num_episodes:
            # Get action from agent
            with torch.no_grad():
                if hasattr(self.agent, 'get_action_and_value'):
                    actions, log_probs, values, entropy = self.agent.get_action_and_value(obs)
                else:
                    # Fallback for standard agents
                    actions = self.agent.act(obs)
                    log_probs = torch.zeros(obs.shape[0])
                    values = torch.zeros(obs.shape[0])
                    entropy = torch.zeros(obs.shape[0])
            
            # Step environment
            next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
            
            # Update metrics
            current_episode_reward += rewards.mean().item()
            current_episode_length += 1
            self.total_steps += 1
            
            # Track step-level metrics
            self.step_metrics['rewards'].append(rewards.mean().item())
            self.step_metrics['values'].append(values.mean().item())
            self.step_metrics['entropy'].append(entropy.mean().item())
            self.step_metrics['log_probs'].append(log_probs.mean().item())
            
            # Check for episode termination
            if terminated.any() or truncated.any():
                # Record episode metrics
                self.episode_rewards.append(current_episode_reward)
                self.episode_lengths.append(current_episode_length)
                
                # Update best/worst rewards
                self.best_episode_reward = max(self.best_episode_reward, current_episode_reward)
                self.worst_episode_reward = min(self.worst_episode_reward, current_episode_reward)
                
                # Reset for next episode
                current_episode_reward = 0
                current_episode_length = 0
                episode_count += 1
                
                if self.verbose and episode_count % 10 == 0:
                    self.logger.info(f"Completed {episode_count}/{self.num_episodes} episodes")
            
            obs = next_obs
        
        # Compute final metrics
        results = self._compute_metrics()
        
        self.logger.info("Evaluation completed!")
        self.logger.info(f"Average reward: {results['average_reward']:.3f}")
        self.logger.info(f"Average episode length: {results['average_episode_length']:.1f}")
        self.logger.info(f"Success rate: {results['success_rate']:.1%}")
        
        return results
    
    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics.
        
        Returns:
            Dictionary of computed metrics
        """
        if not self.episode_rewards:
            return {}
        
        # Basic statistics
        avg_reward = np.mean(self.episode_rewards)
        std_reward = np.std(self.episode_rewards)
        median_reward = np.median(self.episode_rewards)
        
        avg_length = np.mean(self.episode_lengths)
        std_length = np.std(self.episode_lengths)
        
        # Success rate (episodes with positive reward)
        success_rate = np.mean([r > 0 for r in self.episode_rewards])
        
        # Reward distribution percentiles
        reward_percentiles = {
            'p25': np.percentile(self.episode_rewards, 25),
            'p50': np.percentile(self.episode_rewards, 50),
            'p75': np.percentile(self.episode_rewards, 75),
            'p90': np.percentile(self.episode_rewards, 90),
            'p95': np.percentile(self.episode_rewards, 95),
        }
        
        # Step-level metrics
        step_metrics = {}
        for key, values in self.step_metrics.items():
            if values:
                step_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Training stability metrics
        reward_trend = self._compute_trend(self.episode_rewards)
        length_trend = self._compute_trend(self.episode_lengths)
        
        # Hybrid-specific metrics
        hybrid_metrics = {}
        if hasattr(self.agent, 'get_metrics'):
            agent_metrics = self.agent.get_metrics()
            hybrid_metrics = {
                'ppo_weight': agent_metrics.ppo_weight,
                'a2c_weight': agent_metrics.a2c_weight,
                'training_mode': agent_metrics.training_mode,
                'ppo_loss': agent_metrics.ppo_loss,
                'a2c_loss': agent_metrics.a2c_loss,
                'policy_entropy': agent_metrics.policy_entropy,
                'kl_divergence': agent_metrics.kl_divergence,
            }
        
        return {
            'average_reward': avg_reward,
            'std_reward': std_reward,
            'median_reward': median_reward,
            'best_episode_reward': self.best_episode_reward,
            'worst_episode_reward': self.worst_episode_reward,
            'average_episode_length': avg_length,
            'std_episode_length': std_length,
            'success_rate': success_rate,
            'reward_percentiles': reward_percentiles,
            'step_metrics': step_metrics,
            'reward_trend': reward_trend,
            'length_trend': length_trend,
            'hybrid_metrics': hybrid_metrics,
            'total_episodes': len(self.episode_rewards),
            'total_steps': self.total_steps,
        }
    
    def _compute_trend(self, values: List[float]) -> Dict[str, float]:
        """Compute trend analysis for a series of values.
        
        Args:
            values: List of values to analyze
            
        Returns:
            Dictionary containing trend metrics
        """
        if len(values) < 2:
            return {'slope': 0.0, 'correlation': 0.0, 'stability': 0.0}
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)
        correlation = np.corrcoef(x, y)[0, 1]
        
        # Stability (inverse of variance)
        stability = 1.0 / (np.var(y) + 1e-8)
        
        return {
            'slope': slope,
            'correlation': correlation,
            'stability': stability
        }
    
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """Save evaluation results to files.
        
        Args:
            results: Evaluation results
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON results
        json_path = os.path.join(output_dir, "evaluation_results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save episode data
        episode_data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
        }
        episode_path = os.path.join(output_dir, "episode_data.json")
        with open(episode_path, 'w') as f:
            json.dump(episode_data, f, indent=2)
        
        # Create plots
        self._create_plots(output_dir)
        
        self.logger.info(f"Results saved to {output_dir}")
    
    def _create_plots(self, output_dir: str):
        """Create evaluation plots.
        
        Args:
            output_dir: Output directory
        """
        # Episode rewards plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.hist(self.episode_rewards, bins=20, alpha=0.7)
        plt.title('Reward Distribution')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(self.episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Length')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        if self.step_metrics['rewards']:
            # Moving average of step rewards
            window_size = min(100, len(self.step_metrics['rewards']) // 10)
            if window_size > 1:
                moving_avg = np.convolve(self.step_metrics['rewards'], 
                                       np.ones(window_size)/window_size, mode='valid')
                plt.plot(moving_avg)
                plt.title(f'Step Rewards (Moving Average, window={window_size})')
                plt.xlabel('Step')
                plt.ylabel('Reward')
                plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'evaluation_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()


@hydra_task_config(args_cli.task, "skrl_genuine_hybrid_ppo_a2c_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Evaluate genuine hybrid PPO-A2C agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # set the agent and environment seed from command line
    agent_cfg["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["seed"]

    # get checkpoint path
    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(args_cli.output_dir, "videos"),
            "step_trigger": lambda step: step % 1000 == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # Create genuine hybrid models and memory
    models = create_genuine_hybrid_models(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        cfg=agent_cfg["agent"]
    )
    
    memory = create_genuine_hybrid_memory(agent_cfg["agent"])
    
    # Create genuine hybrid agent
    agent = GenuineHybridPPOA2C(
        models=models,
        memory=memory,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        cfg=agent_cfg["agent"]
    )

    # Load checkpoint
    print(f"[INFO] Loading model checkpoint from: {checkpoint_path}")
    agent.load_checkpoint(checkpoint_path)

    # Create evaluator
    evaluator = HybridEvaluator(
        agent=agent,
        env=env,
        num_episodes=args_cli.num_episodes,
        verbose=args_cli.verbose_logging
    )

    # Run evaluation
    print(f"[INFO] Starting evaluation for {args_cli.num_episodes} episodes...")
    results = evaluator.evaluate()

    # Save results
    evaluator.save_results(results, args_cli.output_dir)

    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Average Reward: {results['average_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"Median Reward: {results['median_reward']:.3f}")
    print(f"Best Episode Reward: {results['best_episode_reward']:.3f}")
    print(f"Worst Episode Reward: {results['worst_episode_reward']:.3f}")
    print(f"Average Episode Length: {results['average_episode_length']:.1f} ± {results['std_episode_length']:.1f}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Total Episodes: {results['total_episodes']}")
    print(f"Total Steps: {results['total_steps']}")
    
    if results['hybrid_metrics']:
        print("\nHybrid Metrics:")
        print(f"  PPO Weight: {results['hybrid_metrics']['ppo_weight']:.3f}")
        print(f"  A2C Weight: {results['hybrid_metrics']['a2c_weight']:.3f}")
        print(f"  Training Mode: {results['hybrid_metrics']['training_mode']}")
        print(f"  Policy Entropy: {results['hybrid_metrics']['policy_entropy']:.6f}")
        print(f"  KL Divergence: {results['hybrid_metrics']['kl_divergence']:.6f}")
    
    print(f"\nResults saved to: {args_cli.output_dir}")
    print("="*50)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
