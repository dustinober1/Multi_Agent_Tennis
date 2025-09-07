# Project Report: Multi-Agent Tennis with MADDPG

## Learning Algorithm

### MADDPG Overview

The Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm extends DDPG to multi-agent environments by using:
- **Centralized Training**: Critics have access to all agents' observations and actions
- **Decentralized Execution**: Actors only use local observations during execution
- **Experience Replay**: Shared buffer storing experiences from all agents
- **Soft Target Updates**: Gradual updates to target networks for stability

### Neural Network Architectures

#### Actor Network (per agent)
- Input: 24 (state dimensions)
- Hidden Layer 1: 256 units with ReLU activation and Batch Normalization
- Hidden Layer 2: 128 units with ReLU activation
- Output: 2 units with Tanh activation (action dimensions)

#### Critic Network (per agent)
- Input: 48 (all agents' states) + 4 (all agents' actions)
- Hidden Layer 1: 256 units with ReLU activation and Batch Normalization
- Hidden Layer 2: 128 units with ReLU activation
- Output: 1 unit (Q-value)

### Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Learning Rate (Actor) | 5e-4 | Higher rate needed for sparse rewards |
| Learning Rate (Critic) | 1e-3 | Faster value learning |
| Batch Size | 64 | Smaller batches for more frequent updates |
| Replay Buffer Size | 1e6 | Large buffer for diverse experiences |
| Discount Factor (γ) | 0.99 | Standard for continuous tasks |
| Soft Update Rate (τ) | 5e-3 | Faster target network updates |
| Updates per Step | 5 | Multiple updates to extract more learning |
| Noise Sigma | 0.1 | Moderate exploration |
| Warmup Episodes | 300 | Extensive random exploration initially |
| Reward Amplification | 5x | Addresses sparse reward problem |

## Plot of Rewards

![Training Progress](training_progress.png)

The training showed distinct phases:
1. **Episodes 1-200**: Slow initial learning (avg score ~0.05)
2. **Episodes 200-300**: Rapid improvement (avg score 0.05→0.13)
3. **Episodes 300-400**: Acceleration phase (avg score 0.13→0.50)
4. **Episode 404**: Environment solved with average score of 0.50

Key observations:
- Sharp improvement after episode 270 indicates emergence of collaborative behavior
- Increasing variance in scores (up to 2.6) shows agents learned complex strategies
- Test performance (1.28 average) demonstrates robust learned policy

## Ideas for Future Work

### 1. Algorithm Enhancements
- **Prioritized Experience Replay**: Sample important transitions more frequently based on TD error
- **Multi-Agent PPO**: Try on-policy methods which might be more stable
- **SAC (Soft Actor-Critic)**: Incorporate entropy regularization for better exploration

### 2. Architecture Improvements
- **Attention Mechanisms**: Help agents focus on relevant information about opponent
- **LSTM/GRU Layers**: Handle partial observability and temporal dependencies
- **Parameter Sharing**: Share some layers between agents for faster learning

### 3. Training Strategies
- **Curriculum Learning**: Gradually increase task difficulty (ball speed, net height)
- **Self-Play**: Train against past versions for robustness
- **League Play**: Create diverse population of opponents

### 4. Hyperparameter Optimization
- **Automated Tuning**: Use Optuna or Ray Tune for systematic search
- **Adaptive Learning Rates**: Implement schedules or adaptive optimizers
- **Dynamic Noise**: Adjust exploration based on performance

### 5. Generalization
- **Transfer Learning**: Apply learned skills to different racket sports
- **Multi-Task Learning**: Train on multiple environments simultaneously
- **Zero-Shot Transfer**: Test on environments with different physics

## Conclusion

The MADDPG implementation successfully solved the Tennis environment in 404 episodes, demonstrating effective multi-agent learning. The key to success was:
1. Higher learning rates (5e-4, 1e-3) to handle sparse rewards
2. Extensive warmup (300 episodes) for exploration
3. Reward amplification (5x) to strengthen learning signal
4. Multiple updates per step (5) to maximize learning

The final agent shows robust collaborative behavior with an average test score of 1.28, significantly exceeding the requirement of 0.5.