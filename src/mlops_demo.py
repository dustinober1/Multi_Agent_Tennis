"""
MLOps Demo Script
================

This script demonstrates the complete MLOps pipeline including:
- Model registry and versioning
- Performance monitoring  
- Model comparison and promotion
"""

import os
import sys
import time
import json
import random
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

try:
    import numpy as np
except ImportError:
    # Mock numpy for demo
    class MockNumpy:
        def random(self):
            class MockRandom:
                def uniform(self, low, high):
                    return random.uniform(low, high)
                def normal(self, mean, std):
                    return random.gauss(mean, std)
            return MockRandom()
        def mean(self, data):
            return sum(data) / len(data)
        def percentile(self, data, p):
            sorted_data = sorted(data)
            index = int(p * len(sorted_data) / 100)
            return sorted_data[min(index, len(sorted_data) - 1)]
    np = MockNumpy()

# Mock PyTorch for demo
class MockTorch:
    class nn:
        class Module:
            def __init__(self):
                pass
            def state_dict(self):
                return {"layer1.weight": "mock_tensor", "layer1.bias": "mock_tensor"}
        class Sequential:
            def __init__(self, *args):
                pass
        class Linear:
            def __init__(self, *args):
                pass
        class ReLU:
            def __init__(self):
                pass
        class Tanh:
            def __init__(self):
                pass
    
    def save(self, state_dict, path):
        # Mock save function
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(state_dict, f)
            
torch = MockTorch()

from model_registry import ModelRegistry
from monitoring import PerformanceMonitor


class DemoActor(torch.nn.Module):
    """Demo actor network for demonstration."""
    
    def __init__(self, state_size=33, action_size=4, hidden_size=256):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, action_size),
            torch.nn.Tanh()
        )
    
    def forward(self, state):
        return self.network(state)


class DemoCritic(torch.nn.Module):
    """Demo critic network for demonstration."""
    
    def __init__(self, state_size=33, action_size=4, hidden_size=256):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_size + action_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)


def demo_experiment_tracking():
    """Demonstrate experiment tracking capabilities."""
    print("\n" + "="*50)
    print("🧪 EXPERIMENT TRACKING DEMO")
    print("="*50)
    
    # Create demo config
    from config import MADDPGConfig
    demo_config = MADDPGConfig(
        learning_rate=0.001,
        batch_size=128,
        hidden_size=256,
        gamma=0.99
    )
    
    # Initialize tracker
    tracker = ExperimentTracker(
        experiment_name="demo_experiment",
        config=demo_config,
        platforms=["mlflow"]
    )
    
    # Get experiment ID
    experiment_id = tracker.run_id
    
    print(f"📊 Started experiment: {experiment_id}")
    
    # Simulate training metrics
    for epoch in range(5):
        metrics = {
            "actor_loss": np.random.uniform(0.1, 0.5),
            "critic_loss": np.random.uniform(0.2, 0.8),
            "episode_reward": np.random.uniform(0.1, 0.9),
            "exploration_rate": max(0.1, 1.0 - epoch * 0.1)
        }
        
        tracker.log_metrics(metrics, step=epoch)
        print(f"   Epoch {epoch}: Reward = {metrics['episode_reward']:.3f}")
        time.sleep(0.5)
    
    # Create demo models
    actor = DemoActor()
    critic = DemoCritic()
    
    # Save models
    model_dir = Path("demo_models")
    model_dir.mkdir(exist_ok=True)
    
    torch.save(actor.state_dict(), model_dir / "demo_actor.pth")
    torch.save(critic.state_dict(), model_dir / "demo_critic.pth")
    
    # Log artifacts
    tracker.log_artifact(str(model_dir / "demo_actor.pth"), "model")
    tracker.log_artifact(str(model_dir / "demo_critic.pth"), "model")
    
    # End experiment
    tracker.end_experiment()
    
    print("✅ Experiment tracking demo completed")
    return experiment_id


def demo_hyperparameter_optimization():
    """Demonstrate hyperparameter optimization."""
    print("\n" + "="*50)
    print("🔧 HYPERPARAMETER OPTIMIZATION DEMO")
    print("="*50)
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(
        optimization_backend="optuna",
        direction="maximize",
        n_trials=5
    )
    
    # Define objective function
    def objective(trial):
        # Sample hyperparameters
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
        hidden_size = trial.suggest_int('hidden_size', 128, 512, step=64)
        
        # Simulate training with these hyperparameters
        # In real scenario, this would train the actual model
        score = np.random.uniform(0.1, 1.0)
        
        # Add some logic to make certain combinations better
        if lr < 0.01 and batch_size >= 128:
            score += 0.2
        if hidden_size >= 256:
            score += 0.1
        
        print(f"   Trial: lr={lr:.4f}, batch={batch_size}, hidden={hidden_size}, score={score:.3f}")
        
        return score
    
    # Run optimization
    print("🚀 Starting optimization (5 trials)...")
    best_params, best_score = optimizer.optimize_hyperparameters(
        objective_function=objective,
        search_space={
            'learning_rate': ('float', 1e-5, 1e-2, True),
            'batch_size': ('choice', [64, 128, 256]),
            'hidden_size': ('int', 128, 512, 64)
        }
    )
    
    # Get best parameters
    print(f"\n✅ Best parameters: {best_params}")
    print(f"   Best score: {best_score:.3f}")
    
    return best_params


def demo_model_registry():
    """Demonstrate model registry and versioning."""
    print("\n" + "="*50)
    print("📚 MODEL REGISTRY DEMO")
    print("="*50)
    
    # Initialize registry
    registry = ModelRegistry("demo_registry")
    
    # Register multiple model versions
    models_to_register = [
        ("tennis_actor", "v1.0.0", {"learning_rate": 0.001, "batch_size": 128}, {"reward": 0.5}),
        ("tennis_actor", "v1.1.0", {"learning_rate": 0.0005, "batch_size": 256}, {"reward": 0.7}),
        ("tennis_actor", "v2.0.0", {"learning_rate": 0.002, "batch_size": 128}, {"reward": 0.9}),
    ]
    
    for name, version, params, metrics in models_to_register:
        # Create a demo model
        model = DemoActor()
        
        # Register model
        model_id = registry.register_model(
            model=model,
            name=name,
            version=version,
            description=f"Demo model {version}",
            tags=["demo", "tennis", "actor"],
            hyperparameters=params,
            metrics=metrics,
            author="mlops_demo"
        )
        
        print(f"   Registered: {model_id}")
    
    # List models
    print("\n📋 All models in registry:")
    models = registry.list_models()
    for model_info in models:
        print(f"   - {model_info['name']} (v{model_info['latest_version']}) - Reward: {model_info['metrics'].get('reward', 'N/A')}")
    
    # Compare models
    print("\n🔍 Comparing model versions:")
    comparison = registry.compare_models([
        ("tennis_actor", "v1.0.0"),
        ("tennis_actor", "v1.1.0"),
        ("tennis_actor", "v2.0.0")
    ])
    
    for metric, results in comparison["metrics_comparison"].items():
        print(f"   {metric}:")
        for model_id, value in results.items():
            print(f"     {model_id}: {value}")
    
    # Promote best model
    best_version = "v2.0.0"
    print(f"\n🚀 Promoting {best_version} to production...")
    registry.promote_to_production("tennis_actor", best_version)
    
    print("✅ Model registry demo completed")
    return registry


def demo_monitoring():
    """Demonstrate performance monitoring."""
    print("\n" + "="*50)
    print("📊 MONITORING DEMO")
    print("="*50)
    
    # Initialize monitor
    monitor = PerformanceMonitor(
        model_name="demo_tennis_model",
        monitoring_interval=5,  # Short interval for demo
        enable_prometheus=False  # Disable for demo
    )
    
    # Simulate model predictions
    print("🔍 Simulating model predictions...")
    
    for i in range(10):
        # Simulate prediction metrics
        inference_time = np.random.uniform(0.01, 0.1)
        prediction_score = np.random.uniform(0.3, 0.9)
        input_size = 33
        
        # Simulate occasional high inference time (should trigger alert)
        if i == 7:
            inference_time = 1.5  # This should trigger alert
        
        # Log prediction
        monitor.log_prediction(
            model_version="v2.0.0",
            inference_time=inference_time,
            prediction_score=prediction_score,
            input_size=input_size,
            success=True
        )
        
        print(f"   Prediction {i+1}: {inference_time:.3f}s, score: {prediction_score:.3f}")
        time.sleep(0.2)
    
    # Get performance summary
    print("\n📈 Performance Summary:")
    summary = monitor.get_performance_summary(hours=1)
    
    print(f"   Total requests: {summary['total_requests']}")
    print(f"   Average inference time: {summary['avg_inference_time']:.3f}s")
    print(f"   Average prediction score: {summary['avg_prediction_score']:.3f}")
    print(f"   Active alerts: {summary['active_alerts']}")
    
    print("✅ Monitoring demo completed")
    return monitor


def demo_complete_mlops_pipeline():
    """Demonstrate complete MLOps pipeline integration."""
    print("\n" + "="*60)
    print("🎯 COMPLETE MLOPS PIPELINE DEMO")
    print("="*60)
    
    # 1. Start with experiment tracking
    print("\n1️⃣ Initializing experiment tracking...")
    tracker = ExperimentTracker(
        project_name="complete_mlops_demo",
        experiment_name="end_to_end_demo"
    )
    
    # 2. Optimize hyperparameters
    print("\n2️⃣ Running hyperparameter optimization...")
    optimizer = HyperparameterOptimizer(
        study_name="complete_demo_optimization"
    )
    
    def quick_objective(trial):
        lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        hidden = trial.suggest_int('hidden_size', 128, 512, step=64)
        return np.random.uniform(0.3, 0.8) + (0.1 if lr < 0.01 else 0)
    
    study = optimizer.optimize(quick_objective, n_trials=3, direction="maximize")
    best_params = optimizer.get_best_params()
    print(f"   Best params: {best_params}")
    
    # 3. Train model with best parameters
    print("\n3️⃣ Training model with optimized parameters...")
    experiment_id = tracker.start_experiment(
        config=best_params,
        description="End-to-end MLOps demo"
    )
    
    # Simulate training
    best_score = 0
    for epoch in range(3):
        score = np.random.uniform(0.5, 0.9)
        tracker.log_metrics({"episode_reward": score}, step=epoch)
        best_score = max(best_score, score)
        print(f"   Epoch {epoch}: {score:.3f}")
    
    tracker.end_experiment()
    
    # 4. Register trained model
    print("\n4️⃣ Registering model in registry...")
    registry = ModelRegistry("complete_demo_registry")
    
    model = DemoActor(hidden_size=best_params.get('hidden_size', 256))
    model_id = registry.register_model(
        model=model,
        name="optimized_tennis_actor",
        description="Model trained with optimized hyperparameters",
        hyperparameters=best_params,
        metrics={"best_reward": best_score},
        experiment_id=experiment_id
    )
    
    print(f"   Registered: {model_id}")
    
    # 5. Setup monitoring
    print("\n5️⃣ Setting up production monitoring...")
    monitor = PerformanceMonitor(
        model_name="optimized_tennis_actor",
        enable_prometheus=False
    )
    
    # Simulate production usage
    print("   Simulating production predictions...")
    for i in range(5):
        monitor.log_prediction(
            model_version=model_id.split(':')[1],
            inference_time=np.random.uniform(0.01, 0.05),
            prediction_score=np.random.uniform(0.7, 0.95),
            input_size=33
        )
        print(f"     Prediction {i+1} logged")
    
    # 6. Model promotion workflow
    print("\n6️⃣ Promoting model to production...")
    registry.promote_to_production("optimized_tennis_actor", model_id.split(':')[1])
    
    # 7. Generate final report
    print("\n7️⃣ Generating MLOps pipeline report...")
    
    report = {
        "pipeline_completion": "success",
        "experiment_id": experiment_id,
        "best_hyperparameters": best_params,
        "best_score": best_score,
        "model_id": model_id,
        "monitoring_setup": "active",
        "deployment_status": "production",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save report
    with open("mlops_pipeline_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n✅ Complete MLOps pipeline demo finished!")
    print(f"   Report saved to: mlops_pipeline_report.json")
    print(f"   Final model: {model_id} (Production)")
    print(f"   Best reward: {best_score:.3f}")
    
    return report


def main():
    """Run the complete MLOps demo."""
    print("🚀 Starting MLOps Demo")
    print("This demo showcases all MLOps features:")
    print("- Experiment tracking")
    print("- Hyperparameter optimization") 
    print("- Model registry and versioning")
    print("- Performance monitoring")
    print("- Complete pipeline integration")
    
    try:
        # Run individual demos
        demo_experiment_tracking()
        demo_hyperparameter_optimization()
        demo_model_registry()
        demo_monitoring()
        
        # Run complete pipeline
        report = demo_complete_mlops_pipeline()
        
        print("\n" + "="*60)
        print("🎉 MLOps DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nAll MLOps components demonstrated:")
        print("✅ Experiment tracking with MLflow/W&B")
        print("✅ Hyperparameter optimization with Optuna")
        print("✅ Model registry and versioning")
        print("✅ Real-time performance monitoring")
        print("✅ Complete MLOps pipeline integration")
        print("\nYour MADDPG Tennis project now has production-ready MLOps capabilities!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
