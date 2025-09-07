# Multi-Agent Tennis Environment with MADDPG

This repository contains a solution to the Unity ML-Agents Tennis environment using Multi-Agent Deep Deterministic Policy Gradient (MADDPG).

## Project Details

In this environment, two agents control rackets to bounce a ball over a net. The goal is to train the agents to collaborate and keep the ball in play as long as possible.

### Environment Specifications
- **State Space**: 24 continuous variables per agent (position and velocity of ball and racket)
- **Action Space**: 2 continuous actions per agent (movement toward/away from net, and jumping)
- **Rewards**: 
  - +0.1 for hitting the ball over the net
  - -0.01 for letting the ball hit the ground or out of bounds
- **Solved Criteria**: Average score of +0.5 over 100 consecutive episodes

## Getting Started

### Dependencies

1. Python 3.6 or higher
2. Install required packages:
```bash
pip install torch numpy matplotlib unityagents
```

### Download the Tennis environment:
Extract the environment file and update the path in the notebook.

##  Instructions

Open Tennis.ipynb in Jupyter Notebook
Update the environment path in the Config class
Run all cells to train the agent
The trained model weights will be saved as:

solved_actor_0.pth, solved_actor_1.pth (actor networks)
solved_critic_0.pth, solved_critic_1.pth (critic networks)

## Results

Training Episodes to Solve: 404
Final Average Score: 0.50
Test Performance: 1.28 average over 10 episodes

The agents successfully learned to collaborate, demonstrating sustained rallies and strategic positioning.