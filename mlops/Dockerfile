# Production Docker image for MADDPG Tennis training
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY models/ ./models/
COPY notebooks/ ./notebooks/

# Create directories for MLOps
RUN mkdir -p /app/logs /app/experiments /app/artifacts

# Set environment variables
ENV PYTHONPATH=/app/src
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python", "src/mlops_training.py", "--episodes", "1000", "--experiment-name", "production_training"]

