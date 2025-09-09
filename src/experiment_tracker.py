"""
MLOps Experiment Tracking for Multi-Agent Tennis
=================================================
"""

import os
import json
import time
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

# Import experiment tracking libraries
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available. Install with: pip install mlflow")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Weights & Biases not available. Install with: pip install wandb")

from dataclasses import asdict
from config import MADDPGConfig


class ExperimentTracker:
    """Comprehensive experiment tracking system."""
    
    def __init__(self, experiment_name: str, config: MADDPGConfig, platforms: List[str] = ["mlflow"]):
        self.experiment_name = experiment_name
        self.config = config
        self.platforms = platforms
        
        # Generate unique run ID
        self.run_id = self._generate_run_id()
        self.start_time = datetime.now()
        
        # Initialize tracking platforms
        self.trackers = {}
        self._init_trackers()
        
        # Log initial experiment info
        self.log_experiment_start()
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        config_str = json.dumps(asdict(self.config), sort_keys=True)
        timestamp = str(time.time())
        return hashlib.md5((config_str + timestamp).encode()).hexdigest()[:12]
    
    def _init_trackers(self):
        """Initialize tracking platforms."""
        if "mlflow" in self.platforms and MLFLOW_AVAILABLE:
            try:
                mlflow.set_experiment(self.experiment_name)
                mlflow.start_run(run_name=f"run_{self.run_id}")
                self.trackers["mlflow"] = True
                print(f"✅ MLflow tracking initialized - Run ID: {self.run_id}")
            except Exception as e:
                print(f"❌ MLflow initialization failed: {e}")
                self.trackers["mlflow"] = False
        
        if "wandb" in self.platforms and WANDB_AVAILABLE:
            try:
                wandb.init(
                    project=self.experiment_name,
                    name=f"run_{self.run_id}",
                    config=asdict(self.config)
                )
                self.trackers["wandb"] = True
                print(f"✅ Weights & Biases tracking initialized")
            except Exception as e:
                print(f"❌ Weights & Biases initialization failed: {e}")
                self.trackers["wandb"] = False
    
    def log_experiment_start(self):
        """Log initial experiment information."""
        self.log_hyperparameters(asdict(self.config))
        print(f"🚀 Experiment '{self.experiment_name}' started with run ID: {self.run_id}")
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to all tracking platforms."""
        if self.trackers.get("mlflow"):
            try:
                mlflow.log_params(params)
            except Exception as e:
                print(f"MLflow hyperparameter logging failed: {e}")
        
        if self.trackers.get("wandb"):
            try:
                wandb.config.update(params)
            except Exception as e:
                print(f"Weights & Biases hyperparameter logging failed: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to all tracking platforms."""
        if self.trackers.get("mlflow"):
            try:
                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                print(f"MLflow metrics logging failed: {e}")
        
        if self.trackers.get("wandb"):
            try:
                log_dict = metrics.copy()
                if step is not None:
                    log_dict["step"] = step
                wandb.log(log_dict, step=step)
            except Exception as e:
                print(f"Weights & Biases metrics logging failed: {e}")
    
    def log_episode_metrics(self, episode: int, scores: List[float], avg_score: float):
        """Log episode-specific metrics."""
        metrics = {
            "episode": episode,
            "agent_0_score": scores[0] if len(scores) > 0 else 0,
            "agent_1_score": scores[1] if len(scores) > 1 else 0,
            "average_score": avg_score,
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "score_std": np.std(scores) if scores else 0
        }
        self.log_metrics(metrics, step=episode)
    
    def log_model(self, model: torch.nn.Module, model_name: str):
        """Log and version PyTorch models."""
        model_path = f"models/{model_name}_{self.run_id}.pth"
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        
        if self.trackers.get("mlflow"):
            try:
                mlflow.log_artifact(model_path)
            except Exception as e:
                print(f"MLflow model logging failed: {e}")
        
        if self.trackers.get("wandb"):
            try:
                artifact = wandb.Artifact(name=f"{model_name}_{self.run_id}", type="model")
                artifact.add_file(model_path)
                wandb.log_artifact(artifact)
            except Exception as e:
                print(f"Weights & Biases model logging failed: {e}")
    
    def finish_experiment(self):
        """Finish experiment and cleanup tracking."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        self.log_metrics({"experiment_duration_seconds": duration})
        
        if self.trackers.get("mlflow"):
            try:
                mlflow.end_run()
            except Exception as e:
                print(f"MLflow cleanup failed: {e}")
        
        if self.trackers.get("wandb"):
            try:
                wandb.finish()
            except Exception as e:
                print(f"Weights & Biases cleanup failed: {e}")
        
        print(f"✅ Experiment completed in {duration:.2f} seconds")
