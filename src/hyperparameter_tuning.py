"""
Hyperparameter Optimization for MADDPG Tennis
==============================================

This module provides automated hyperparameter optimization using multiple
optimization frameworks including Optuna, Ray Tune, and Hyperopt.
"""

import os
import json
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from dataclasses import asdict

# Optimization libraries
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Install with: pip install optuna")

try:
    import ray
    from ray import tune
    from ray.tune.schedulers import AsyncHyperBandScheduler
    from ray.tune.suggest.optuna import OptunaSearch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Ray Tune not available. Install with: pip install ray[tune]")

try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    print("Hyperopt not available. Install with: pip install hyperopt")

from config import MADDPGConfig
from experiment_tracker import ExperimentTracker


class HyperparameterOptimizer:
    """
    Comprehensive hyperparameter optimization system.
    
    Supports multiple optimization backends and provides advanced features like
    early stopping, multi-objective optimization, and parallel execution.
    """
    
    def __init__(
        self,
        optimization_backend: str = "optuna",
        direction: str = "maximize",
        n_trials: int = 100,
        timeout: Optional[int] = None,
        pruning: bool = True
    ):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            optimization_backend: Backend to use ("optuna", "ray", "hyperopt")
            direction: "maximize" or "minimize" the objective
            n_trials: Number of trials to run
            timeout: Maximum time in seconds
            pruning: Whether to use early stopping/pruning
        """
        self.backend = optimization_backend
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        self.pruning = pruning
        
        # Results storage
        self.results_dir = Path("hyperparameter_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Best trial tracking
        self.best_trial = None
        self.best_score = float('-inf') if direction == "maximize" else float('inf')
        
        # Validation
        self._validate_backend()
    
    def _validate_backend(self):
        """Validate that the selected backend is available."""
        if self.backend == "optuna" and not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")
        elif self.backend == "ray" and not RAY_AVAILABLE:
            raise ImportError("Ray Tune not available. Install with: pip install ray[tune]")
        elif self.backend == "hyperopt" and not HYPEROPT_AVAILABLE:
            raise ImportError("Hyperopt not available. Install with: pip install hyperopt")
    
    def define_search_space(self) -> Dict[str, Any]:
        """
        Define the hyperparameter search space for MADDPG.
        
        Returns:
            Dictionary defining the search space
        """
        if self.backend == "optuna":
            return {
                # Learning rates
                'lr_actor': ('log_uniform', 1e-5, 1e-2),
                'lr_critic': ('log_uniform', 1e-5, 1e-2),
                
                # Network architecture
                'actor_fc1_units': ('categorical', [128, 256, 512]),
                'actor_fc2_units': ('categorical', [64, 128, 256]),
                'critic_fc1_units': ('categorical', [128, 256, 512]),
                'critic_fc2_units': ('categorical', [64, 128, 256]),
                
                # Training parameters
                'batch_size': ('categorical', [32, 64, 128, 256]),
                'gamma': ('uniform', 0.9, 0.999),
                'tau': ('log_uniform', 1e-4, 1e-1),
                
                # Exploration
                'noise_sigma': ('uniform', 0.01, 0.3),
                'noise_theta': ('uniform', 0.1, 0.5),
                
                # Training schedule
                'update_every': ('categorical', [1, 2, 4, 8]),
                'num_updates': ('categorical', [1, 3, 5, 10]),
                'warmup_episodes': ('categorical', [100, 200, 300, 500])
            }
        
        elif self.backend == "ray":
            return {
                'lr_actor': tune.loguniform(1e-5, 1e-2),
                'lr_critic': tune.loguniform(1e-5, 1e-2),
                'actor_fc1_units': tune.choice([128, 256, 512]),
                'actor_fc2_units': tune.choice([64, 128, 256]),
                'critic_fc1_units': tune.choice([128, 256, 512]),
                'critic_fc2_units': tune.choice([64, 128, 256]),
                'batch_size': tune.choice([32, 64, 128, 256]),
                'gamma': tune.uniform(0.9, 0.999),
                'tau': tune.loguniform(1e-4, 1e-1),
                'noise_sigma': tune.uniform(0.01, 0.3),
                'noise_theta': tune.uniform(0.1, 0.5),
                'update_every': tune.choice([1, 2, 4, 8]),
                'num_updates': tune.choice([1, 3, 5, 10]),
                'warmup_episodes': tune.choice([100, 200, 300, 500])
            }
        
        elif self.backend == "hyperopt":
            return {
                'lr_actor': hp.loguniform('lr_actor', np.log(1e-5), np.log(1e-2)),
                'lr_critic': hp.loguniform('lr_critic', np.log(1e-5), np.log(1e-2)),
                'actor_fc1_units': hp.choice('actor_fc1_units', [128, 256, 512]),
                'actor_fc2_units': hp.choice('actor_fc2_units', [64, 128, 256]),
                'critic_fc1_units': hp.choice('critic_fc1_units', [128, 256, 512]),
                'critic_fc2_units': hp.choice('critic_fc2_units', [64, 128, 256]),
                'batch_size': hp.choice('batch_size', [32, 64, 128, 256]),
                'gamma': hp.uniform('gamma', 0.9, 0.999),
                'tau': hp.loguniform('tau', np.log(1e-4), np.log(1e-1)),
                'noise_sigma': hp.uniform('noise_sigma', 0.01, 0.3),
                'noise_theta': hp.uniform('noise_theta', 0.1, 0.5),
                'update_every': hp.choice('update_every', [1, 2, 4, 8]),
                'num_updates': hp.choice('num_updates', [1, 3, 5, 10]),
                'warmup_episodes': hp.choice('warmup_episodes', [100, 200, 300, 500])
            }
    
    def optimize_with_optuna(
        self,
        objective_function: Callable,
        study_name: str = "maddpg_optimization"
    ) -> Dict[str, Any]:
        """
        Run optimization using Optuna.
        
        Args:
            objective_function: Function to optimize
            study_name: Name for the study
        
        Returns:
            Best hyperparameters and results
        """
        # Create study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner() if self.pruning else None
        
        study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
            study_name=study_name
        )
        
        # Define objective wrapper
        def optuna_objective(trial):
            # Sample hyperparameters
            params = {}
            search_space = self.define_search_space()
            
            for param_name, (method, *args) in search_space.items():
                if method == 'log_uniform':
                    params[param_name] = trial.suggest_loguniform(param_name, args[0], args[1])
                elif method == 'uniform':
                    params[param_name] = trial.suggest_uniform(param_name, args[0], args[1])
                elif method == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, args[0])
                elif method == 'int':
                    params[param_name] = trial.suggest_int(param_name, args[0], args[1])
            
            # Evaluate objective
            score = objective_function(params, trial)
            
            # Track best result
            if self._is_better_score(score):
                self.best_score = score
                self.best_trial = {
                    'params': params,
                    'score': score,
                    'trial_number': trial.number
                }
            
            return score
        
        # Run optimization
        study.optimize(
            optuna_objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Save results
        results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'study_name': study_name,
            'optimization_backend': 'optuna'
        }
        
        self._save_results(results, f"{study_name}_optuna_results.json")
        
        return results
    
    def optimize_with_ray_tune(
        self,
        objective_function: Callable,
        num_samples: Optional[int] = None,
        max_concurrent_trials: int = 4
    ) -> Dict[str, Any]:
        """
        Run optimization using Ray Tune.
        
        Args:
            objective_function: Function to optimize
            num_samples: Number of samples (defaults to self.n_trials)
            max_concurrent_trials: Maximum concurrent trials
        
        Returns:
            Best hyperparameters and results
        """
        if not num_samples:
            num_samples = self.n_trials
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Define search space
        search_space = self.define_search_space()
        
        # Create scheduler
        scheduler = AsyncHyperBandScheduler(
            time_attr="training_iteration",
            metric="score",
            mode="max" if self.direction == "maximize" else "min"
        )
        
        # Run optimization
        analysis = tune.run(
            objective_function,
            config=search_space,
            num_samples=num_samples,
            scheduler=scheduler,
            max_concurrent_trials=max_concurrent_trials,
            verbose=1
        )
        
        # Get best result
        best_trial = analysis.get_best_trial(
            metric="score",
            mode="max" if self.direction == "maximize" else "min"
        )
        
        results = {
            'best_params': best_trial.config,
            'best_value': best_trial.last_result["score"],
            'n_trials': len(analysis.trials),
            'optimization_backend': 'ray_tune'
        }
        
        self._save_results(results, "ray_tune_results.json")
        
        return results
    
    def _is_better_score(self, score: float) -> bool:
        """Check if score is better than current best."""
        if self.direction == "maximize":
            return score > self.best_score
        else:
            return score < self.best_score
    
    def _save_results(self, results: Dict[str, Any], filename: str):
        """Save optimization results to file."""
        results_path = self.results_dir / filename
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved to {results_path}")


def create_objective_function(
    train_function: Callable,
    episodes: int = 500,
    target_score: float = 0.5
) -> Callable:
    """
    Create objective function for hyperparameter optimization.
    
    Args:
        train_function: Function that trains MADDPG agent
        episodes: Number of episodes to train
        target_score: Target score for early stopping
    
    Returns:
        Objective function compatible with optimization frameworks
    """
    def objective(params: Dict[str, Any], trial=None) -> float:
        """
        Objective function that trains agent and returns performance score.
        
        Args:
            params: Hyperparameters to evaluate
            trial: Optional trial object for pruning
        
        Returns:
            Performance score (higher is better)
        """
        # Create config with suggested parameters
        config = MADDPGConfig()
        
        # Update config with suggested parameters
        for key, value in params.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Create experiment tracker
        experiment_name = f"hyperparam_opt_trial_{trial.number if trial else 'unknown'}"
        tracker = ExperimentTracker(
            experiment_name=experiment_name,
            config=config,
            platforms=["mlflow"]  # Use lightweight tracking during optimization
        )
        
        try:
            # Train agent
            final_score, episode_scores = train_function(config, tracker, episodes)
            
            # Log final results
            tracker.log_metrics({
                "final_score": final_score,
                "trials_to_solve": len(episode_scores),
                "max_score": max(episode_scores) if episode_scores else 0
            })
            
            # Finish experiment
            tracker.finish_experiment()
            
            return final_score
            
        except Exception as e:
            print(f"Trial failed with error: {e}")
            tracker.finish_experiment()
            return 0.0  # Return poor score for failed trials
    
    return objective


# Example usage functions
def run_hyperparameter_optimization(
    train_function: Callable,
    backend: str = "optuna",
    n_trials: int = 50,
    episodes_per_trial: int = 300
) -> Dict[str, Any]:
    """
    Run complete hyperparameter optimization.
    
    Args:
        train_function: Function that trains MADDPG agent
        backend: Optimization backend to use
        n_trials: Number of trials to run
        episodes_per_trial: Episodes per trial
    
    Returns:
        Best hyperparameters and optimization results
    """
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        optimization_backend=backend,
        direction="maximize",
        n_trials=n_trials,
        pruning=True
    )
    
    # Create objective function
    objective = create_objective_function(
        train_function=train_function,
        episodes=episodes_per_trial
    )
    
    # Run optimization
    if backend == "optuna":
        results = optimizer.optimize_with_optuna(objective)
    elif backend == "ray":
        results = optimizer.optimize_with_ray_tune(objective)
    else:
        raise ValueError(f"Backend {backend} not supported")
    
    print(f"🏆 Optimization completed!")
    print(f"Best score: {results['best_value']:.4f}")
    print(f"Best parameters: {results['best_params']}")
    
    return results

