# Multi-Agent Tennis (MADDPG)

This repository contains an implementation of Multi-Agent Deep Deterministic Policy Gradient (MADDPG) applied to the Unity Tennis environment (two cooperative agents). The codebase includes training logic, a Jupyter notebook for walkthrough and experiments, and pre-trained model weights.

This README documents project layout, setup, how to reproduce training/evaluation, and where to find model artifacts and results.

## Repository layout

- `src/` - implementation and scripts (e.g. `maddpg_agent.py`)
- `models/` - saved PyTorch weights (.pth)
- `notebooks/` - `Tennis.ipynb` — training / analysis notebook
- `reports/` - documentation and project reports (this file)
- `requirements.txt` - Python dependencies for quick setup

Quick note: if you open the project in an editor, the main training script is `src/maddpg_agent.py` and trained weights are in `models/`.

## Project overview

The Unity Tennis environment simulates two racket agents that must learn to keep a ball in play; each agent receives observations (position, velocity, etc.) and outputs continuous actions. The MADDPG algorithm uses centralized critics and decentralized actors to train multiple agents in a cooperative (or mixed) setting.

Core ideas implemented here:
- Per-agent actor networks (policy)
- Per-agent critic networks with access to joint observations/actions for stable multi-agent learning
- Replay buffers, Ornstein-Uhlenbeck / action noise, and soft target updates

## Environment details (summary)

- State vector: ~24 continuous values per agent (environment-specific)
- Action vector: 2 continuous actions per agent (move, jump)
- Reward shaping: small positive reward for scoring, small negative for losing the ball
- Solved criteria (typical): average score >= 0.5 across 100 consecutive episodes

Note: exact observation and action shapes depend on the Unity environment build you use. Inspect the environment (or the notebook) to confirm shapes before training.

## Quickstart — reproducible setup

1. Create a clean Python environment (recommended):

```bash
# from project root
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Obtain the Unity Tennis environment binary (not included):
- Download the environment executable that matches your OS from the assignment or Unity package. Place it under `envs/` or any folder you prefer.
- You can also use the Unity Editor + ML-Agents to generate a build.

3. Configure the environment path used by the notebook or scripts:
- Open `notebooks/Tennis.ipynb` and the `Config` section, or set an environment variable `UNITY_ENV_PATH` pointing to the executable.

4. Run the notebook for a guided walkthrough or run the trainer directly:

```bash
# Guided: open the notebook
jupyter notebook notebooks/Tennis.ipynb

# Direct training (if script supports CLI):
python src/maddpg_agent.py --env-path ./envs/Tennis_Linux.x86_64
```

If `maddpg_agent.py` expects a hard-coded path, edit the path in the top-level configuration variables in `src/maddpg_agent.py`.

## Models and artifacts

- Saved actor/critic weights are in `models/` with filenames like `best_actor_0.pth`, `best_critic_1.pth`.
- During training the code will also write intermediate checkpoints to the `models/` folder.

To evaluate a saved model, load the `.pth` files with the network classes defined in `src/` and run the environment in evaluation mode (no exploration noise).

## Reproducing the reported results

1. Install dependencies (see Quickstart).
2. Place the Unity environment binary and point the notebook or script to it.
3. Run training for the same number of episodes reported (the notebook records the number of episodes used in the experiments):

```bash
python src/maddpg_agent.py --episodes 1000 --env-path ./envs/Tennis_Linux.x86_64
```

4. After training finishes, inspect `models/` and the training logs (the notebook plots scores over time). The training notebook also contains code to reproduce the plots shown in the project report.

## Evaluation

Use the notebook evaluation cells or a small script that:
1. Loads actor networks from `models/`.
2. Steps the environment for N episodes without noise.
3. Logs per-episode and mean scores.

Example (conceptual):

```python
# pseudo-code
# load actor networks
# for episode in range(10):
#   reset env
#   while not done:
#       act = actor(obs)
#       obs, reward, done, _ = env.step(act)
#   record score
```

## Notes & tips

- Unity environment: make sure the executable's bitness/OS matches your machine.
- GPU support: PyTorch will use CUDA if available. Install the proper `torch` wheel for your CUDA version if you want GPU acceleration.
- If you see shape mismatches, check observation/action sizes printed by the notebook when you connect to the environment.

## Results (from experiments)

- Training episodes to solve (example): 404
- Final average score recorded: ~0.50
- Test performance example: 1.28 average over 10 episodes

Your results may vary depending on environment binary version, randomness seeds, and hyperparameters.

## Troubleshooting

- If the notebook cannot connect to the environment, confirm the `env` path and executable permissions (`chmod +x`).
- If packages fail to import, check your virtual environment and the `requirements.txt` installation.

## License & contact

This repository is provided for educational purposes. See `report.md` for details about experiments and parameters. For questions, contact the repo owner.

---

If you want, I can also:
- Add a top-level `README.md` that points to this `reports/README.md` and includes a one-line summary.
- Add a short evaluation script `src/evaluate.py` that loads `models/` and runs N episodes.

Replace this file with updated README content.