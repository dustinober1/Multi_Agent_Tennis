"""Evaluation script for trained MADDPG agents."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os
from collections import deque


def evaluate_agents(env, actors, num_episodes=10):
    """Evaluate trained agents."""
    brain_name = env.brain_names[0]
    scores_deque = deque(maxlen=100)
    episode_scores = []
    
    for episode in range(num_episodes):
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(len(actors))
        
        while True:
            actions = []
            for i, actor in enumerate(actors):
                state = torch.from_numpy(states[i]).float().unsqueeze(0)
                with torch.no_grad():
                    action = actor(state).cpu().data.numpy()
                actions.append(action)
            
            actions = np.concatenate(actions, axis=0)
            env_info = env.step(actions)[brain_name]
            
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            
            scores += rewards
            states = next_states
            
            if np.any(dones):
                break
        
        max_score = np.max(scores)
        scores_deque.append(max_score)
        episode_scores.append(scores)
        
        print(f'Episode {episode+1}: Agent Scores: {scores}, Max: {max_score:.3f}')
    
    avg_score = np.mean(scores_deque)
    print(f'Average Score: {avg_score:.3f}')
    
    return episode_scores, avg_score


if __name__ == '__main__':
    print("Use this script to evaluate trained MADDPG agents")
