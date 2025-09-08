"""Configuration management for MADDPG Tennis training."""

import os
from dataclasses import dataclass


@dataclass
class MADDPGConfig:
    """Configuration class for MADDPG agent and training."""
    
    # Environment settings
    env_path: str = "./envs/Tennis_Linux.x86_64"
    no_graphics: bool = True
    
    # Network architecture
    actor_fc1_units: int = 256
    actor_fc2_units: int = 128
    critic_fc1_units: int = 256
    critic_fc2_units: int = 128
    
    # Training hyperparameters
    lr_actor: float = 5e-4
    lr_critic: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 5e-3
    update_every: int = 1
    num_updates: int = 5
    
    # Exploration
    noise_sigma: float = 0.1
    noise_theta: float = 0.15
    noise_mu: float = 0.0
    
    # Training settings
    max_episodes: int = 1000
    warmup_episodes: int = 300
    target_score: float = 0.5
    score_window: int = 100
    reward_amplification: float = 5.0
    
    # Checkpointing
    checkpoint_dir: str = "./models"
    save_every: int = 100
    
    # Logging
    log_dir: str = "./logs"
    plot_dir: str = "./plots"
    
    # Random seed
    random_seed: int = 42
