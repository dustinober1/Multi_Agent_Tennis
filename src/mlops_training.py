"""
MLOps-enabled Training Script for MADDPG Tennis
===============================================

This script provides production-ready training with comprehensive experiment tracking,
model versioning, automated hyperparameter optimization, and monitoring.
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config import MADDPGConfig
from experiment_tracker import ExperimentTracker
from hyperparameter_tuning import run_hyperparameter_optimization, create_objective_function
from maddpg_agent import MADDPGAgent

# Try to import Unity environment
try:
    from unityagents import UnityEnvironment
    UNITY_AVAILABLE = True
except ImportError:
    UNITY_AVAILABLE = False
    print("Unity environment not available. Using mock environment for testing.")


class MLOpsTrainer:
    """
    Production MLOps trainer with comprehensive tracking and optimization.
    
    Features:
    - Experiment tracking with MLflow/Weights & Biases
    - Automated hyperparameter optimization
    - Model versioning and artifact storage
    - Performance monitoring and alerts
    - Automated model validation and testing
    """
    
    def __init__(
        self,
        config: MADDPGConfig,
        experiment_name: str = "maddpg_tennis_training",
        tracking_platforms: List[str] = ["mlflow"],
        enable_optimization: bool = False
    ):
        """
        Initialize MLOps trainer.
        
        Args:
            config: MADDPG configuration
            experiment_name: Name for experiment tracking
            tracking_platforms: List of tracking platforms to use
            enable_optimization: Whether to run hyperparameter optimization
        """
        self.config = config
        self.experiment_name = experiment_name
        self.tracking_platforms = tracking_platforms
        self.enable_optimization = enable_optimization
        
        # Initialize environment
        self.env = self._init_environment()
        self.state_size = 24  # Tennis environment state size
        self.action_size = 2  # Tennis environment action size
        self.num_agents = 2
        
        # Initialize experiment tracker
        self.tracker = ExperimentTracker(
            experiment_name=experiment_name,
            config=config,
            platforms=tracking_platforms
        )
        
        # Initialize agent
        self.agent = MADDPGAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            num_agents=self.num_agents,
            random_seed=config.random_seed
        )
        
        print(f"✅ MLOps Trainer initialized")
        print(f"   - Experiment: {experiment_name}")
        print(f"   - Tracking: {tracking_platforms}")
        print(f"   - Optimization: {enable_optimization}")
    
    def _init_environment(self):
        """Initialize Unity Tennis environment or mock for testing."""
        if UNITY_AVAILABLE and os.path.exists(self.config.env_path):
            try:
                env = UnityEnvironment(
                    file_name=self.config.env_path,
                    no_graphics=self.config.no_graphics
                )
                return env
            except Exception as e:
                print(f"Failed to initialize Unity environment: {e}")
                return self._create_mock_environment()
        else:
            return self._create_mock_environment()
    
    def _create_mock_environment(self):
        """Create mock environment for testing without Unity."""
        class MockEnvironment:
            def reset(self, train_mode=True):
                # Return mock states for 2 agents
                return np.random.uniform(-1, 1, (2, 24))
            
            def step(self, actions):
                # Return mock next_states, rewards, dones
                next_states = np.random.uniform(-1, 1, (2, 24))
                rewards = np.random.uniform(0, 0.1, 2)
                dones = np.random.choice([True, False], 2, p=[0.05, 0.95])
                return next_states, rewards, dones, {}
            
            def close(self):
                pass
        
        print("⚠️  Using mock environment for testing")
        return MockEnvironment()
    
    def train_episode(self, episode: int) -> Tuple[List[float], float]:
        """
        Train one episode and return scores.
        
        Args:
            episode: Episode number
        
        Returns:
            Tuple of (individual_scores, average_score)
        """
        # Reset environment
        states = self.env.reset(train_mode=True)
        if hasattr(states, 'vector_observations'):
            states = states.vector_observations
        
        # Initialize episode variables
        scores = np.zeros(self.num_agents)
        step_count = 0
        max_steps = 1000  # Prevent infinite episodes
        
        while step_count < max_steps:
            # Get actions from agent
            actions = self.agent.act(states)
            
            # Take step in environment
            next_states, rewards, dones, _ = self.env.step(actions)
            if hasattr(next_states, 'vector_observations'):
                next_states = next_states.vector_observations
            
            # Store experience and learn
            self.agent.step(states, actions, rewards, next_states, dones)
            
            # Update states and scores
            states = next_states
            scores += rewards
            step_count += 1
            
            # Check if episode is done
            if np.any(dones):
                break
        
        avg_score = np.mean(scores)
        
        # Log episode metrics
        self.tracker.log_episode_metrics(
            episode=episode,
            scores=scores.tolist(),
            avg_score=avg_score,
            additional_metrics={
                "episode_length": step_count,
                "exploration_noise": self.agent.noise.sigma if hasattr(self.agent, 'noise') else 0.1
            }
        )
        
        return scores.tolist(), avg_score
    
    def train(
        self,
        episodes: Optional[int] = None,
        target_score: Optional[float] = None,
        save_every: Optional[int] = None
    ) -> Tuple[float, List[float]]:
        """
        Main training loop with comprehensive tracking.
        
        Args:
            episodes: Number of episodes to train (defaults to config)
            target_score: Target score to reach (defaults to config)
            save_every: Save models every N episodes (defaults to config)
        
        Returns:
            Tuple of (final_score, all_episode_scores)
        """
        # Use config defaults if not specified
        episodes = episodes or self.config.max_episodes
        target_score = target_score or self.config.target_score
        save_every = save_every or self.config.save_every
        
        print(f"🚀 Starting training for {episodes} episodes")
        print(f"   Target score: {target_score}")
        print(f"   Save every: {save_every} episodes")
        
        # Training variables
        all_scores = []
        scores_window = []
        best_score = 0.0
        solved_episode = None
        
        for episode in range(1, episodes + 1):
            # Train one episode
            episode_scores, avg_score = self.train_episode(episode)
            all_scores.append(avg_score)
            scores_window.append(avg_score)
            
            # Maintain rolling window
            if len(scores_window) > self.config.score_window:
                scores_window.pop(0)
            
            # Calculate metrics
            rolling_avg = np.mean(scores_window)
            is_best = avg_score > best_score
            
            if is_best:
                best_score = avg_score
            
            # Log progress
            if episode % 10 == 0:
                print(f"Episode {episode:4d} | Score: {avg_score:.3f} | "
                      f"Rolling Avg: {rolling_avg:.3f} | Best: {best_score:.3f}")
            
            # Log additional metrics
            self.tracker.log_metrics({
                "rolling_average": rolling_avg,
                "best_score": best_score,
                "is_best_episode": is_best,
                "episodes_completed": episode
            }, step=episode)
            
            # Check if environment is solved
            if rolling_avg >= target_score and len(scores_window) >= self.config.score_window:
                if solved_episode is None:
                    solved_episode = episode
                    print(f"🎉 Environment solved in {episode} episodes!")
                    print(f"   Rolling average: {rolling_avg:.3f}")
                    
                    # Log solving metrics
                    self.tracker.log_metrics({
                        "environment_solved": True,
                        "episodes_to_solve": episode,
                        "solving_score": rolling_avg
                    })
                    
                    # Save best models
                    self._save_models(episode, "solved")
            
            # Save models periodically
            if episode % save_every == 0:
                self._save_models(episode, "checkpoint")
            
            # Save best model
            if is_best:
                self._save_models(episode, "best")
        
        # Training completed
        final_score = np.mean(scores_window) if scores_window else 0.0
        
        print(f"✅ Training completed!")
        print(f"   Final average score: {final_score:.3f}")
        print(f"   Best score: {best_score:.3f}")
        if solved_episode:
            print(f"   Solved in: {solved_episode} episodes")
        
        # Log final metrics
        self.tracker.log_metrics({
            "final_score": final_score,
            "training_completed": True,
            "total_episodes": episodes
        })
        
        # Save final models
        self._save_models(episodes, "final")
        
        return final_score, all_scores
    
    def _save_models(self, episode: int, model_type: str):
        """Save actor and critic models with versioning."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save actor models
        for i in range(self.num_agents):
            actor_name = f"actor_{i}_{model_type}_{episode}_{timestamp}"
            self.tracker.log_model(
                model=self.agent.actor_local if hasattr(self.agent, 'actor_local') else None,
                model_name=actor_name
            )
        
        # Save critic models  
        for i in range(self.num_agents):
            critic_name = f"critic_{i}_{model_type}_{episode}_{timestamp}"
            self.tracker.log_model(
                model=self.agent.critic_local if hasattr(self.agent, 'critic_local') else None,
                model_name=critic_name
            )
        
        print(f"💾 Saved {model_type} models for episode {episode}")
    
    def evaluate(self, episodes: int = 100) -> Dict[str, float]:
        """
        Evaluate trained models.
        
        Args:
            episodes: Number of episodes to evaluate
        
        Returns:
            Evaluation metrics
        """
        print(f"🧪 Evaluating model for {episodes} episodes...")
        
        eval_scores = []
        
        for episode in range(episodes):
            # Evaluation episode (no training)
            states = self.env.reset(train_mode=False)
            if hasattr(states, 'vector_observations'):
                states = states.vector_observations
            
            scores = np.zeros(self.num_agents)
            step_count = 0
            max_steps = 1000
            
            while step_count < max_steps:
                # Get actions (no noise during evaluation)
                actions = self.agent.act(states, add_noise=False)
                next_states, rewards, dones, _ = self.env.step(actions)
                
                if hasattr(next_states, 'vector_observations'):
                    next_states = next_states.vector_observations
                
                states = next_states
                scores += rewards
                step_count += 1
                
                if np.any(dones):
                    break
            
            eval_scores.append(np.mean(scores))
        
        # Calculate evaluation metrics
        metrics = {
            "eval_mean_score": np.mean(eval_scores),
            "eval_std_score": np.std(eval_scores),
            "eval_min_score": np.min(eval_scores),
            "eval_max_score": np.max(eval_scores),
            "eval_episodes": episodes
        }
        
        # Log evaluation results
        self.tracker.log_metrics(metrics)
        
        print(f"📊 Evaluation Results:")
        print(f"   Mean Score: {metrics['eval_mean_score']:.3f} ± {metrics['eval_std_score']:.3f}")
        print(f"   Score Range: [{metrics['eval_min_score']:.3f}, {metrics['eval_max_score']:.3f}]")
        
        return metrics
    
    def run_hyperparameter_optimization(
        self,
        n_trials: int = 50,
        episodes_per_trial: int = 300
    ) -> Dict[str, Any]:
        """
        Run automated hyperparameter optimization.
        
        Args:
            n_trials: Number of optimization trials
            episodes_per_trial: Episodes per trial
        
        Returns:
            Best hyperparameters and results
        """
        print(f"🔧 Starting hyperparameter optimization with {n_trials} trials...")
        
        # Create training function for optimization
        def train_function(config: MADDPGConfig, tracker: ExperimentTracker, episodes: int):
            # Create temporary trainer
            temp_trainer = MLOpsTrainer(
                config=config,
                experiment_name=f"{self.experiment_name}_opt_trial",
                tracking_platforms=["mlflow"],  # Lightweight tracking
                enable_optimization=False
            )
            
            # Train and return final score
            final_score, episode_scores = temp_trainer.train(episodes=episodes)
            temp_trainer.cleanup()
            
            return final_score, episode_scores
        
        # Run optimization
        results = run_hyperparameter_optimization(
            train_function=train_function,
            backend="optuna",
            n_trials=n_trials,
            episodes_per_trial=episodes_per_trial
        )
        
        # Log optimization results
        self.tracker.log_metrics({
            "optimization_completed": True,
            "best_optimized_score": results["best_value"],
            "optimization_trials": results["n_trials"]
        })
        
        return results
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            self.env.close()
        except:
            pass
        
        try:
            self.tracker.finish_experiment()
        except:
            pass


def main():
    """Main training function with CLI interface."""
    parser = argparse.ArgumentParser(description="MLOps MADDPG Tennis Training")
    
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--target-score", type=float, default=0.5, help="Target score to reach")
    parser.add_argument("--experiment-name", type=str, default="maddpg_tennis", help="Experiment name")
    parser.add_argument("--platforms", nargs="+", default=["mlflow"], help="Tracking platforms")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--optimize-trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after training")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Number of evaluation episodes")
    
    args = parser.parse_args()
    
    # Create configuration
    config = MADDPGConfig()
    config.max_episodes = args.episodes
    config.target_score = args.target_score
    
    # Create trainer
    trainer = MLOpsTrainer(
        config=config,
        experiment_name=args.experiment_name,
        tracking_platforms=args.platforms,
        enable_optimization=args.optimize
    )
    
    try:
        # Run hyperparameter optimization if requested
        if args.optimize:
            opt_results = trainer.run_hyperparameter_optimization(
                n_trials=args.optimize_trials,
                episodes_per_trial=300
            )
            
            # Update config with best parameters
            for key, value in opt_results["best_params"].items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            print(f"🏆 Using optimized hyperparameters: {opt_results['best_params']}")
        
        # Train model
        final_score, all_scores = trainer.train()
        
        # Evaluate if requested
        if args.evaluate:
            eval_results = trainer.evaluate(episodes=args.eval_episodes)
        
        print(f"✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print(f"⚠️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        raise
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()

