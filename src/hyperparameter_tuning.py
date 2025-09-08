"""Hyperparameter tuning for MADDPG Tennis."""

import json
import os


def define_search_space():
    """Define hyperparameter search space."""
    return {
        'lr_actor': [1e-4, 5e-4, 1e-3],
        'lr_critic': [5e-4, 1e-3, 2e-3],
        'batch_size': [32, 64, 128],
        'tau': [1e-3, 5e-3, 1e-2],
        'noise_sigma': [0.05, 0.1, 0.2]
    }


def save_best_config(config, score, filename='best_config.json'):
    """Save best configuration found."""
    result = {
        'config': config,
        'score': score,
        'timestamp': '2024-01-01'
    }
    
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Best config saved: {filename}")


if __name__ == '__main__':
    print("Hyperparameter tuning utilities")
