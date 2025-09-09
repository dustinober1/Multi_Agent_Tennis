"""
Simple MLOps Demo Script
========================

This script demonstrates the core MLOps features that work without external dependencies:
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

from model_registry import ModelRegistry
from monitoring import PerformanceMonitor


class MockModel:
    """Mock model for demonstration."""
    
    def __init__(self, name="demo_model"):
        self.name = name
        self.data = {"weights": [random.random() for _ in range(100)]}
    
    def state_dict(self):
        return self.data
    
    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)


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
        model = MockModel(f"{name}_{version}")
        
        # Save model to temporary file
        model_dir = Path("temp_models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / f"{name}_{version}.pkl"
        model.save(model_path)
        
        # Register model
        model_id = registry.register_model(
            model=str(model_path),
            name=name,
            version=version,
            description=f"Demo model {version}",
            tags=["demo", "tennis", "actor"],
            hyperparameters=params,
            metrics=metrics,
            author="mlops_demo"
        )
        
        print(f"   ✅ Registered: {model_id}")
    
    # List models
    print("\n📋 All models in registry:")
    models = registry.list_models()
    for model_info in models:
        reward = model_info['metrics'].get('reward', 'N/A')
        print(f"   - {model_info['name']} (latest: {model_info['latest_version']}) - Reward: {reward}")
    
    # List versions for a specific model
    print(f"\n📝 Versions of tennis_actor:")
    versions = registry.list_versions("tennis_actor")
    for version_info in versions:
        print(f"   - {version_info['version']}: reward={version_info['metrics'].get('reward', 'N/A')}")
    
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
    success = registry.promote_to_production("tennis_actor", best_version)
    
    if success:
        print("✅ Model successfully promoted to production")
    else:
        print("❌ Failed to promote model")
    
    print("✅ Model registry demo completed")
    return registry


def demo_monitoring():
    """Demonstrate performance monitoring."""
    print("\n" + "="*50)
    print("�� MONITORING DEMO")
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
        inference_time = random.uniform(0.01, 0.1)
        prediction_score = random.uniform(0.3, 0.9)
        input_size = 33
        
        # Simulate occasional high inference time (should trigger alert)
        if i == 7:
            inference_time = 1.5  # This should trigger alert
            print("   ⚠️  Simulating high inference time...")
        
        # Log prediction
        monitor.log_prediction(
            model_version="v2.0.0",
            inference_time=inference_time,
            prediction_score=prediction_score,
            input_size=input_size,
            success=True
        )
        
        print(f"   Prediction {i+1}: {inference_time:.3f}s, score: {prediction_score:.3f}")
        time.sleep(0.1)  # Small delay for realism
    
    # Get performance summary
    print("\n📈 Performance Summary:")
    summary = monitor.get_performance_summary(hours=1)
    
    print(f"   Total requests: {summary['total_requests']}")
    print(f"   Average inference time: {summary['avg_inference_time']:.3f}s")
    print(f"   Average prediction score: {summary['avg_prediction_score']:.3f}")
    print(f"   Max inference time: {summary['p95_inference_time']:.3f}s")
    print(f"   Active alerts: {summary['active_alerts']}")
    
    # Export metrics
    print("\n💾 Exporting metrics...")
    json_metrics = monitor.export_metrics("json")
    
    with open("demo_metrics.json", "w") as f:
        f.write(json_metrics)
    
    print("   ✅ Metrics exported to demo_metrics.json")
    
    print("✅ Monitoring demo completed")
    return monitor


def demo_complete_workflow():
    """Demonstrate complete MLOps workflow."""
    print("\n" + "="*60)
    print("🎯 COMPLETE MLOPS WORKFLOW DEMO")
    print("="*60)
    
    # 1. Model development and registration
    print("\n1️⃣ Model Development & Registration...")
    registry = ModelRegistry("workflow_registry")
    
    # Simulate training multiple models
    models = []
    for i in range(3):
        version = f"v1.{i}.0"
        reward = random.uniform(0.4, 0.9)
        
        model = MockModel(f"workflow_actor_{version}")
        model_path = Path("temp_models") / f"workflow_actor_{version}.pkl"
        model_path.parent.mkdir(exist_ok=True)
        model.save(model_path)
        
        model_id = registry.register_model(
            model=str(model_path),
            name="workflow_actor",
            version=version,
            description=f"Workflow model {version}",
            hyperparameters={"lr": 0.001 * (i+1), "batch_size": 128},
            metrics={"reward": reward, "episodes": 1000},
            author="workflow_demo"
        )
        
        models.append((version, reward))
        print(f"   ✅ Trained and registered {model_id} (reward: {reward:.3f})")
    
    # 2. Model comparison and selection
    print("\n2️⃣ Model Comparison & Selection...")
    comparison = registry.compare_models([
        ("workflow_actor", "v1.0.0"),
        ("workflow_actor", "v1.1.0"),
        ("workflow_actor", "v1.2.0")
    ], metrics=["reward"])
    
    best_model = comparison["best_model"]["reward"]["model"]
    best_version = best_model.split(":")[1]
    print(f"   🏆 Best model: {best_model}")
    
    # 3. Model promotion
    print("\n3️⃣ Model Promotion...")
    registry.promote_to_staging("workflow_actor", best_version)
    print(f"   ✅ Promoted {best_version} to staging")
    
    registry.promote_to_production("workflow_actor", best_version)
    print(f"   ✅ Promoted {best_version} to production")
    
    # 4. Production monitoring
    print("\n4️⃣ Production Monitoring...")
    monitor = PerformanceMonitor(
        model_name="workflow_actor_prod",
        enable_prometheus=False
    )
    
    # Simulate production usage
    print("   🔍 Simulating production predictions...")
    for i in range(5):
        monitor.log_prediction(
            model_version=best_version,
            inference_time=random.uniform(0.01, 0.05),
            prediction_score=random.uniform(0.7, 0.95),
            input_size=33
        )
        print(f"     ✅ Prediction {i+1} logged successfully")
    
    # 5. Performance analysis
    print("\n5️⃣ Performance Analysis...")
    summary = monitor.get_performance_summary(hours=1)
    
    report = {
        "workflow_status": "completed",
        "best_model": best_model,
        "total_models_trained": len(models),
        "production_model": f"workflow_actor:{best_version}",
        "production_metrics": summary,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save workflow report
    with open("mlops_workflow_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"   📊 Performance: {summary['avg_prediction_score']:.3f} avg score")
    print(f"   📊 Latency: {summary['avg_inference_time']:.3f}s avg time")
    print("   ✅ Workflow report saved to mlops_workflow_report.json")
    
    print("\n✅ Complete MLOps workflow finished successfully!")
    return report


def main():
    """Run the simplified MLOps demo."""
    print("🚀 Starting Simple MLOps Demo")
    print("This demo showcases core MLOps features:")
    print("- ✅ Model registry and versioning")
    print("- ✅ Performance monitoring") 
    print("- ✅ Model comparison and promotion")
    print("- ✅ Complete MLOps workflow")
    
    try:
        # Run demos
        registry = demo_model_registry()
        monitor = demo_monitoring()
        workflow_report = demo_complete_workflow()
        
        print("\n" + "="*60)
        print("🎉 SIMPLE MLOPS DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nMLOps components demonstrated:")
        print("✅ Model registry with versioning and metadata")
        print("✅ Real-time performance monitoring with alerts")
        print("✅ Model comparison and selection")
        print("✅ Automated promotion workflows (staging → production)")
        print("✅ Complete end-to-end MLOps pipeline")
        
        print("\n�� Generated files:")
        print("   - demo_registry/: Model registry with versions")
        print("   - monitoring/: Performance monitoring data")
        print("   - demo_metrics.json: Exported monitoring metrics")
        print("   - mlops_workflow_report.json: Complete workflow report")
        
        print("\nYour MADDPG Tennis project now has core MLOps capabilities!")
        print("🔗 Next steps: Install MLOps dependencies to unlock advanced features")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
