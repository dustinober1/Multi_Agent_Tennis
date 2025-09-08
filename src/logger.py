"""Logging utilities for MADDPG Tennis training."""

import logging
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


class TrainingLogger:
    """Logging system for training metrics."""
    
    def __init__(self, log_dir="./logs", experiment_name=None):
        self.log_dir = log_dir
        self.experiment_name = experiment_name or f"maddpg_tennis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = os.path.join(log_dir, self.experiment_name)
        
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Training metrics
        self.episode_scores = []
        self.episode_rewards = []
        self.loss_history = {"actor": [], "critic": []}
        
    def log_episode(self, episode, scores, avg_score):
        """Log episode results."""
        self.episode_scores.append(scores)
        print(f"Episode {episode:4d} | Scores: {scores} | Avg Score: {avg_score:.3f}")
        
    def plot_training_progress(self):
        """Create training progress plot."""
        episodes = range(1, len(self.episode_scores) + 1)
        max_scores = [max(scores) for scores in self.episode_scores]
        
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, max_scores, alpha=0.6, label='Max Score per Episode')
        
        # Moving average
        if len(max_scores) >= 100:
            moving_avg = [np.mean(max_scores[max(0, i-99):i+1]) for i in range(len(max_scores))]
            plt.plot(episodes, moving_avg, 'r-', linewidth=2, label='100-Episode Moving Average')
            plt.axhline(y=0.5, color='g', linestyle='--', label='Target Score (0.5)')
        
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('MADDPG Tennis Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(self.experiment_dir, "training_progress.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training plot saved to {plot_path}")
        plt.show()
