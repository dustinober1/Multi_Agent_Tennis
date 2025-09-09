#!/bin/bash
# Complete MLOps Pipeline for MADDPG Tennis

set -e

echo "🚀 Starting MLOps Pipeline for Multi-Agent Tennis"
echo "=================================================="

# Configuration
EXPERIMENT_NAME="maddpg_production_$(date +%Y%m%d_%H%M%S)"
EPISODES=1000
OPTIMIZE_TRIALS=20
PLATFORMS="mlflow wandb"

# Function to check if service is running
check_service() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $service to be ready on port $port..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:$port > /dev/null 2>&1; then
            echo "✅ $service is ready!"
            return 0
        fi
        
        echo "   Attempt $attempt/$max_attempts - waiting..."
        sleep 2
        ((attempt++))
    done
    
    echo "❌ $service failed to start within expected time"
    return 1
}

# Step 1: Start MLOps infrastructure
echo "�� Step 1: Starting MLOps Infrastructure"
echo "----------------------------------------"

cd mlops
docker-compose up -d

# Wait for services
check_service "MLflow" 5000
check_service "Optuna Dashboard" 8080

cd ..

# Step 2: Setup environment
echo "�� Step 2: Setting up Environment"
echo "---------------------------------"

# Install dependencies if not already installed
pip install -r requirements.txt

# Create necessary directories
mkdir -p experiments logs artifacts hyperparameter_results

# Step 3: Run hyperparameter optimization
echo "🔧 Step 3: Hyperparameter Optimization"
echo "--------------------------------------"

python src/mlops_training.py \
    --experiment-name "${EXPERIMENT_NAME}_optimization" \
    --optimize \
    --optimize-trials $OPTIMIZE_TRIALS \
    --episodes 300 \
    --platforms mlflow

# Step 4: Train final model with best hyperparameters
echo "🏋️ Step 4: Training Final Model"
echo "-------------------------------"

python src/mlops_training.py \
    --experiment-name "${EXPERIMENT_NAME}_final" \
    --episodes $EPISODES \
    --evaluate \
    --eval-episodes 100 \
    --platforms $PLATFORMS

# Step 5: Model validation and testing
echo "🧪 Step 5: Model Validation"
echo "---------------------------"

python src/mlops_training.py \
    --experiment-name "${EXPERIMENT_NAME}_validation" \
    --episodes 500 \
    --evaluate \
    --eval-episodes 200

# Step 6: Generate reports and artifacts
echo "📊 Step 6: Generating Reports"
echo "-----------------------------"

python -c "
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Connect to MLflow
mlflow.set_tracking_uri('http://localhost:5000')

# Get experiment data
experiment = mlflow.get_experiment_by_name('${EXPERIMENT_NAME}_final')
if experiment:
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    # Generate performance report
    print('📈 Performance Summary:')
    print(f'   Total Runs: {len(runs)}')
    
    if 'metrics.final_score' in runs.columns:
        best_score = runs['metrics.final_score'].max()
        avg_score = runs['metrics.final_score'].mean()
        print(f'   Best Score: {best_score:.4f}')
        print(f'   Average Score: {avg_score:.4f}')
    
    # Save report
    runs.to_csv('experiments/training_report.csv', index=False)
    print('💾 Report saved to experiments/training_report.csv')
else:
    print('⚠️  No experiment data found')
"

# Step 7: Deploy model artifacts
echo "🚀 Step 7: Model Deployment Preparation"
echo "---------------------------------------"

# Copy best models to deployment directory
mkdir -p deployment/models
if [ -d "models" ]; then
    cp models/best_*.pth deployment/models/ 2>/dev/null || echo "   No best models found"
fi

# Create deployment configuration
cat > deployment/config.yaml << 'DEPLOY_EOF'
model:
  name: maddpg_tennis
  version: "1.0.0"
  framework: pytorch
  
serving:
  port: 8501
  workers: 1
  
inference:
  batch_size: 1
  timeout: 30
  
monitoring:
  enable_metrics: true
  log_requests: true
DEPLOY_EOF

echo "✅ Deployment configuration created"

# Step 8: Cleanup and summary
echo "🧹 Step 8: Pipeline Summary"
echo "---------------------------"

echo "📋 MLOps Pipeline Completed Successfully!"
echo ""
echo "🔗 Access Points:"
echo "   MLflow UI:      http://localhost:5000"
echo "   Optuna Dashboard: http://localhost:8080"
echo "   Jupyter Lab:    http://localhost:8888"
echo ""
echo "📁 Generated Artifacts:"
echo "   Training Reports: experiments/"
echo "   Model Artifacts:  artifacts/"
echo "   Deployment Ready: deployment/"
echo ""
echo "🎯 Next Steps:"
echo "   1. Review results in MLflow UI"
echo "   2. Analyze hyperparameter optimization in Optuna"
echo "   3. Deploy best model from deployment/ directory"
echo "   4. Monitor model performance in production"

# Optional: Open browsers
if command -v open &> /dev/null; then
    echo ""
    echo "🌐 Opening dashboards..."
    open http://localhost:5000  # MLflow
    open http://localhost:8080  # Optuna
fi

