# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Rough-H1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:H1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
        "skrl_hybrid_cfg_entry_point": f"{agents.__name__}:skrl_hybrid_ppo_sac_cfg.yaml",
        "skrl_hybrid_ppo_a2c_cfg_entry_point": f"{agents.__name__}:skrl_hybrid_ppo_a2c_cfg.yaml",
        "skrl_hybrid_ppo_a2c_stable_cfg_entry_point": f"{agents.__name__}:skrl_hybrid_ppo_a2c_stable_cfg.yaml",
        "skrl_genuine_hybrid_ppo_a2c_cfg_entry_point": f"{agents.__name__}:skrl_genuine_hybrid_ppo_a2c_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-H1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:H1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
        "skrl_hybrid_ppo_a2c_cfg_entry_point": f"{agents.__name__}:skrl_hybrid_ppo_a2c_cfg.yaml",
        "skrl_hybrid_ppo_a2c_stable_cfg_entry_point": f"{agents.__name__}:skrl_hybrid_ppo_a2c_stable_cfg.yaml",
        "skrl_genuine_hybrid_ppo_a2c_cfg_entry_point": f"{agents.__name__}:skrl_genuine_hybrid_ppo_a2c_cfg.yaml",

    },
)


gym.register(
    id="Isaac-Velocity-Flat-H1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:H1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "skrl_genuine_hybrid_ppo_a2c_cfg_entry_point": f"{agents.__name__}:skrl_genuine_hybrid_ppo_a2c_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-H1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:H1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "skrl_hybrid_ppo_a2c_cfg_entry_point": f"{agents.__name__}:skrl_hybrid_ppo_a2c_cfg.yaml",
        "skrl_hybrid_ppo_a2c_stable_cfg_entry_point": f"{agents.__name__}:skrl_hybrid_ppo_a2c_stable_cfg.yaml",
    },
)


# Memory-optimized environments for 8GB RTX 4070 GPU
gym.register(
    id="Isaac-Velocity-Rough-H1-MemoryOptimized-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg_memory_optimized:H1RoughEnvCfg_MemoryOptimized",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
        "skrl_hybrid_ppo_a2c_cfg_entry_point": f"{agents.__name__}:skrl_hybrid_ppo_a2c_memory_optimized_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-H1-MemoryOptimized-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg_memory_optimized:H1RoughEnvCfg_MemoryOptimized_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
        "skrl_hybrid_ppo_a2c_cfg_entry_point": f"{agents.__name__}:skrl_hybrid_ppo_a2c_memory_optimized_cfg.yaml",
    },
)
