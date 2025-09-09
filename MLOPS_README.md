# MLOps-Enhanced Multi-Agent Tennis 🎾🤖

A production-ready Multi-Agent Deep Deterministic Policy Gradient (MADDPG) implementation with comprehensive MLOps capabilities for the Unity Tennis environment.

## 🎯 MLOps Features

This project showcases enterprise-level MLOps practices including:

- **🧪 Experiment Tracking**: MLflow, Weights & Biases, Neptune integration
- **🔧 Hyperparameter Optimization**: Optuna, Ray Tune, Hyperopt support  
- **📚 Model Registry**: Versioning, lineage tracking, promotion workflows
- **📊 Performance Monitoring**: Real-time metrics, alerting, drift detection
- **🐳 Containerization**: Docker services with MLflow, Jupyter, databases
- **🚀 CI/CD Pipeline**: GitHub Actions with automated training and deployment
- **📈 Visualization**: Interactive dashboards and performance analytics

## 🏗️ Architecture

```
├── src/                          # Core MLOps modules
│   ├── experiment_tracker.py     # Multi-platform experiment tracking
│   ├── hyperparameter_tuning.py  # Automated hyperparameter optimization
│   ├── model_registry.py         # Model versioning and lifecycle management
│   ├── monitoring.py             # Real-time performance monitoring
│   ├── mlops_training.py         # Production training pipeline
│   └── mlops_demo.py             # Complete MLOps demonstration
├── mlops/                        # Infrastructure and deployment
│   ├── docker-compose.yml        # MLOps services (MLflow, Jupyter, DBs)
│   ├── Dockerfile                # Containerized training environment
│   └── run_mlops_pipeline.sh     # Complete automation script
├── .github/workflows/            # CI/CD pipeline
│   └── mlops-pipeline.yml        # Automated training and deployment
└── monitoring/                   # Monitoring data and alerts
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and navigate to repository
git clone <repository-url>
cd Multi_Agent_Tennis

# Install dependencies
pip install -r requirements.txt

# Setup MLOps infrastructure
chmod +x mlops/run_mlops_pipeline.sh
./mlops/run_mlops_pipeline.sh
```

### 2. Run MLOps Demo

```bash
# Comprehensive demo of all MLOps features
python src/mlops_demo.py
```

### 3. Start MLOps Services

```bash
# Start all MLOps infrastructure
cd mlops
docker-compose up -d

# Access services:
# - MLflow UI: http://localhost:5000
# - Jupyter Lab: http://localhost:8888
# - Optuna Dashboard: http://localhost:8080
```

## 📊 Experiment Tracking

Track experiments across multiple platforms:

```python
from src.experiment_tracker import ExperimentTracker

# Initialize tracker with multiple backends
tracker = ExperimentTracker(
    project_name="maddpg_tennis",
    use_mlflow=True,
    use_wandb=True,
    use_neptune=True
)

# Start experiment
experiment_id = tracker.start_experiment(
    config={
        "learning_rate": 0.001,
        "batch_size": 128,
        "hidden_size": 256
    },
    description="MADDPG training with optimized hyperparameters"
)

# Log metrics during training
tracker.log_metrics({
    "actor_loss": actor_loss,
    "critic_loss": critic_loss,
    "episode_reward": episode_reward
}, step=episode)

# Log model artifacts
tracker.log_artifact("models/best_actor.pth", "model")
tracker.end_experiment()
```

## 🔧 Hyperparameter Optimization

Automated hyperparameter tuning with multiple backends:

```python
from src.hyperparameter_tuning import HyperparameterOptimizer

# Initialize optimizer
optimizer = HyperparameterOptimizer(
    study_name="maddpg_optimization",
    backend="optuna"  # or "ray_tune", "hyperopt"
)

# Define objective function
def objective(trial):
    # Sample hyperparameters
    config = {
        "learning_rate": trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical('batch_size', [64, 128, 256]),
        "hidden_size": trial.suggest_int('hidden_size', 128, 512, step=64)
    }
    
    # Train model with sampled hyperparameters
    trainer = MLOpsTrainer(config)
    best_reward = trainer.train()
    
    return best_reward

# Run optimization
study = optimizer.optimize(
    objective=objective,
    n_trials=100,
    direction="maximize"
)

# Get best parameters
best_params = optimizer.get_best_params()
```

## 📚 Model Registry

Comprehensive model lifecycle management:

```python
from src.model_registry import ModelRegistry

# Initialize registry
registry = ModelRegistry("model_registry")

# Register new model
model_id = registry.register_model(
    model=trained_actor,
    name="tennis_actor",
    version="v2.1.0",
    description="MADDPG actor with optimized hyperparameters",
    tags=["maddpg", "tennis", "production"],
    hyperparameters=best_params,
    metrics={"best_reward": 0.95, "avg_reward": 0.87},
    author="mlops_engineer"
)

# Load model
actor_model = registry.load_model("tennis_actor", version="v2.1.0")

# Compare models
comparison = registry.compare_models([
    ("tennis_actor", "v2.0.0"),
    ("tennis_actor", "v2.1.0")
])

# Promote to production
registry.promote_to_production("tennis_actor", "v2.1.0")
```

## 📊 Performance Monitoring

Real-time monitoring and alerting:

```python
from src.monitoring import PerformanceMonitor

# Setup monitoring
monitor = PerformanceMonitor(
    model_name="tennis_actor_prod",
    monitoring_interval=30,
    enable_prometheus=True
)

# Start monitoring
monitor.start_monitoring()

# Log predictions
monitor.log_prediction(
    model_version="v2.1.0",
    inference_time=0.05,
    prediction_score=0.92,
    input_size=33,
    success=True
)

# Get performance summary
summary = monitor.get_performance_summary(hours=24)
```

## 🐳 Containerized Infrastructure

The MLOps infrastructure runs in Docker containers:

```yaml
# mlops/docker-compose.yml
services:
  mlflow:
    image: mlflow-server
    ports: ["5000:5000"]
    environment:
      - BACKEND_STORE_URI=postgresql://user:pass@postgres:5432/mlflow
      - DEFAULT_ARTIFACT_ROOT=/artifacts
      
  jupyter:
    image: jupyter/tensorflow-notebook
    ports: ["8888:8888"]
    volumes: ["../notebooks:/home/jovyan/work"]
    
  optuna-dashboard:
    image: optuna/optuna-dashboard
    ports: ["8080:8080"]
    
  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=mlflow
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      
  redis:
    image: redis:alpine
    ports: ["6379:6379"]
```

## 🚀 CI/CD Pipeline

Automated training and deployment with GitHub Actions:

```yaml
# .github/workflows/mlops-pipeline.yml
name: MLOps Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  validate-data:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/
          
  train-model:
    needs: validate-data
    strategy:
      matrix:
        training_type: [baseline, optimized]
    runs-on: ubuntu-latest
    steps:
      - name: Train MADDPG
        run: python src/mlops_training.py --mode ${{ matrix.training_type }}
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: models-${{ matrix.training_type }}
          path: models/
          
  evaluate-model:
    needs: train-model
    runs-on: ubuntu-latest
    steps:
      - name: Evaluate performance
        run: python src/evaluate.py --compare-all
      - name: Generate report
        run: python scripts/generate_report.py
```

## 📈 Monitoring Dashboard

Access real-time metrics and alerts:

### MLflow UI (http://localhost:5000)
- Experiment comparison
- Model versioning
- Artifact management
- Parameter tracking

### Optuna Dashboard (http://localhost:8080)  
- Hyperparameter optimization progress
- Trial history and analysis
- Parameter importance plots
- Optimization history

### Jupyter Lab (http://localhost:8888)
- Interactive analysis
- Model debugging
- Custom visualizations
- Experiment notebooks

## 🔍 Model Comparison

Compare different model versions:

```python
# Compare multiple models
comparison = registry.compare_models([
    ("tennis_actor", "v1.0.0"),
    ("tennis_actor", "v2.0.0"), 
    ("tennis_actor", "v2.1.0")
], metrics=["best_reward", "avg_reward", "training_time"])

# Best model by metric
best_models = comparison["best_model"]
print(f"Best reward: {best_models['best_reward']['model']}")
print(f"Fastest training: {best_models['training_time']['model']}")
```

## 🚨 Alerting System

Automated alerts for model performance:

```python
# Custom alert rules
monitor.alert_rules.append(AlertRule(
    name="Low Episode Reward",
    metric="episode_reward", 
    operator="lt",
    threshold=0.5,
    duration=300,  # 5 minutes
    severity="high",
    description="Model performance degraded"
))

# Email/Slack notifications
monitor.setup_notifications(
    email_recipients=["team@company.com"],
    slack_webhook="https://hooks.slack.com/...",
    discord_webhook="https://discord.com/api/webhooks/..."
)
```

## 📊 Performance Metrics

Track comprehensive model performance:

- **Training Metrics**: Actor/Critic loss, episode rewards, exploration rate
- **Inference Metrics**: Prediction time, model accuracy, resource usage  
- **System Metrics**: CPU/GPU utilization, memory usage, disk I/O
- **Business Metrics**: Model availability, error rates, user satisfaction

## 🎯 Production Deployment

Deploy models to production:

```python
# Automated model promotion
if model_performance > threshold:
    registry.promote_to_staging("tennis_actor", "v2.1.0")
    
    # Run staging tests
    staging_results = run_staging_tests()
    
    if staging_results.passed:
        registry.promote_to_production("tennis_actor", "v2.1.0")
        
        # Setup production monitoring
        prod_monitor = PerformanceMonitor(
            model_name="tennis_actor_prod",
            enable_prometheus=True,
            alert_endpoints=["oncall@company.com"]
        )
```

## 🛠️ Development Workflow

1. **Experiment**: Track experiments with MLflow/W&B
2. **Optimize**: Use Optuna for hyperparameter tuning
3. **Register**: Version models in registry
4. **Monitor**: Setup real-time monitoring
5. **Deploy**: Promote through staging to production
6. **Iterate**: Continuous improvement cycle

## 📚 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [Ray Tune Documentation](https://docs.ray.io/en/latest/tune/index.html)
- [Prometheus Monitoring](https://prometheus.io/docs/)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/mlops-enhancement`)
3. Commit changes (`git commit -am 'Add MLOps feature'`)
4. Push to branch (`git push origin feature/mlops-enhancement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Unity ML-Agents team for the Tennis environment
- OpenAI for MADDPG algorithm research
- MLOps community for best practices and tools
- Contributors and maintainers

---

**Ready for Production** 🚀

This MLOps-enhanced MADDPG implementation provides enterprise-grade machine learning operations capabilities, making it ready for production deployment with comprehensive monitoring, versioning, and automation.
