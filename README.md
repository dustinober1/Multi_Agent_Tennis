# 🎾 Multi-Agent Tennis with MADDPG

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.6+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated implementation of **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)** for training cooperative agents in Unity's Tennis environment.

## 🎯 Project Overview

Two AI agents learn to play tennis cooperatively using MADDPG, a state-of-the-art multi-agent reinforcement learning algorithm. The agents must keep a ball in play as long as possible, requiring sophisticated coordination and strategy.

### 🏆 Key Achievements
- **Environment Solved**: Target average score of 0.5+ achieved in 404 episodes
- **Peak Performance**: Test evaluation average score of 1.28
- **Professional Architecture**: Production-ready codebase with comprehensive tooling

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/dustinober1/Multi_Agent_Tennis.git
cd Multi_Agent_Tennis

# Setup environment
chmod +x setup.sh
./setup.sh

# Install dependencies
pip install -r requirements.txt
```

### Run Training
```bash
# Interactive training
jupyter notebook notebooks/Tennis.ipynb

# Direct training
python src/maddpg_agent.py

# Launch interactive web demo
./launch_demo.sh
# OR: streamlit run src/web_demo.py
```

### 📚 Documentation
```bash
# Build comprehensive documentation
./build_docs.sh

# View documentation
open docs/_build/html/index.html
```

## 🏗️ Project Architecture

```
Multi_Agent_Tennis/
├── 📊 notebooks/
│   └── Tennis.ipynb                 # Interactive training & analysis
├── 🧠 src/
│   ├── maddpg_agent.py             # Core MADDPG implementation
│   ├── config.py                   # Configuration management
│   ├── logger.py                   # Professional logging system
│   ├── evaluate.py                 # Model evaluation tools
│   └── hyperparameter_tuning.py   # HPO framework
├── 🧪 tests/
│   └── test_maddpg.py              # Unit tests
├── 🎯 models/
│   ├── best_actor_0.pth            # Trained agent policies
│   └── best_critic_0.pth           # Trained critics
└── 📋 reports/                     # Documentation & reports
```

## 🤖 Algorithm Details

### MADDPG (Multi-Agent Deep Deterministic Policy Gradient)

**Key Innovation**: Centralized training with decentralized execution
- **Training**: Critics observe all agents' states and actions for stable learning
- **Execution**: Actors only use local observations for deployment

### Network Architecture

#### Actor Network (Policy)
```
Input: 24 (state) → FC(256) + BatchNorm + ReLU → FC(128) + ReLU → FC(2) + Tanh
Output: 2 continuous actions bounded in [-1, 1]
```

#### Critic Network (Value Function)  
```
Input: 48 (all states) + 4 (all actions) → FC(256) + BatchNorm + ReLU → FC(128) + ReLU → FC(1)
Output: Q-value for state-action pair
```

## 📈 Training Results

### Performance Metrics
- **Solved in**: 404 episodes (target: avg score ≥ 0.5 over 100 episodes)
- **Test performance**: 1.28 average score (156% above target)
- **Training phases**:
  - Episodes 1-200: Initial exploration (avg score ~0.05)
  - Episodes 200-300: Rapid improvement (0.05 → 0.13) 
  - Episodes 300-404: Mastery phase (0.13 → 0.50+)

### Key Hyperparameters
| Parameter | Value | Purpose |
|-----------|-------|---------|
| Actor Learning Rate | 5e-4 | Policy optimization speed |
| Critic Learning Rate | 1e-3 | Value function learning |
| Batch Size | 64 | Training stability |
| Soft Update (τ) | 5e-3 | Target network updates |
| Noise Sigma | 0.1 | Exploration level |

## 🔧 Advanced Features

### 1. **Interactive Web Demo** 🌐
- **Real-time visualization** of agent gameplay
- **Performance metrics** dashboard with live updates
- **Hyperparameter exploration** with instant feedback
- **Agent behavior analysis** and cooperation metrics
- **Policy visualization** with interactive heatmaps

### 2. **Professional Documentation** 📖
- **Sphinx-generated** API documentation with auto-extraction
- **Mathematical formulations** with MathJax rendering
- **Code examples** and tutorials for easy onboarding
- **Interactive notebooks** embedded in documentation
- **Cross-referenced** modules and functions

### 3. **Configuration Management**
```python
from src.config import MADDPGConfig

config = MADDPGConfig()
config.lr_actor = 1e-3
config.batch_size = 128
```

### 2. Professional Logging
```python
from src.logger import TrainingLogger

logger = TrainingLogger(experiment_name="experiment_1")
logger.log_episode(episode, scores, avg_score)
logger.plot_training_progress()
```

### 3. Model Evaluation
```bash
python src/evaluate.py --env-path ./Tennis.app --episodes 100
```

### 4. Unit Testing
```bash
python -m pytest tests/ -v
```

## 🎮 Environment Details

**Unity Tennis Environment**
- **Observation Space**: 24 continuous variables per agent
- **Action Space**: 2 continuous actions per agent (move, jump)
- **Reward Structure**: +0.1 for hitting ball over net, -0.01 for missing
- **Success Criteria**: Average score ≥ 0.5 over 100 consecutive episodes

## 🔬 Future Research Directions

### Algorithm Enhancements
- Prioritized Experience Replay for better sample efficiency
- Multi-Agent PPO comparison studies
- Attention mechanisms for agent communication

### Advanced Training
- Curriculum learning with progressive difficulty
- Self-play against historical agent versions  
- Transfer learning to related environments

## 📊 Performance Comparison

| Method | Episodes to Solve | Peak Score | Stability |
|--------|------------------|------------|-----------|
| **MADDPG (Ours)** | **404** | **1.28** | **High** |
| DDPG Independent | 800+ | 0.8 | Medium |
| Random Policy | Never | 0.01 | N/A |

## 🛠️ Development Features

### Code Quality
- Comprehensive unit testing framework
- Professional logging and monitoring
- Modular, extensible architecture
- Type hints and documentation

### MLOps Pipeline
- Automated experiment tracking
- Hyperparameter optimization tools
- Model evaluation and comparison
- Reproducible training workflows

## 👥 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Udacity Deep Reinforcement Learning Nanodegree
- OpenAI for MADDPG algorithm research
- Unity ML-Agents for simulation environment

## 📞 Contact

- **Repository**: [https://github.com/dustinober1/Multi_Agent_Tennis](https://github.com/dustinober1/Multi_Agent_Tennis)
- **Author**: Dustin Ober

---

⭐ **If you found this project helpful, please consider giving it a star!**
