"""
Model Registry and Versioning System
===================================

This module provides comprehensive model lifecycle management including
versioning, registry, deployment tracking, and model comparison.
"""

import os
import json
import shutil
import hashlib
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import pickle

# ML libraries
import torch
import numpy as np

# Experiment tracking
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


@dataclass
class ModelMetadata:
    """Comprehensive model metadata."""
    model_id: str
    name: str
    version: str
    created_at: str
    author: str
    description: str
    tags: List[str]
    framework: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: Dict[str, str]  # artifact_name -> path
    dependencies: Dict[str, str]  # package -> version
    model_size_mb: float
    training_duration: Optional[float] = None
    dataset_info: Optional[Dict[str, Any]] = None
    experiment_id: Optional[str] = None
    parent_models: Optional[List[str]] = None
    deployment_status: str = "not_deployed"
    performance_benchmark: Optional[Dict[str, float]] = None


@dataclass
class ModelVersion:
    """Model version information."""
    version: str
    model_path: str
    metadata_path: str
    checksum: str
    created_at: str
    is_active: bool = False
    is_production: bool = False


class ModelRegistry:
    """
    Comprehensive model registry and versioning system.
    
    Features:
    - Model versioning and lineage tracking
    - Metadata management
    - Performance comparison
    - Deployment tracking
    - Model promotion workflows
    - Artifact management
    """
    
    def __init__(self, registry_path: str = "model_registry"):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to store registry data
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        
        # Registry structure
        self.models_dir = self.registry_path / "models"
        self.metadata_dir = self.registry_path / "metadata"
        self.artifacts_dir = self.registry_path / "artifacts"
        self.staging_dir = self.registry_path / "staging"
        self.production_dir = self.registry_path / "production"
        
        # Create directories
        for dir_path in [self.models_dir, self.metadata_dir, self.artifacts_dir,
                        self.staging_dir, self.production_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Load existing registry
        self.registry_file = self.registry_path / "registry.json"
        self.models: Dict[str, Dict[str, ModelVersion]] = self._load_registry()
        
        print(f"✅ Model Registry initialized at {registry_path}")
        print(f"   - Total models: {len(self.models)}")
        print(f"   - Total versions: {self._count_total_versions()}")
    
    def _load_registry(self) -> Dict[str, Dict[str, ModelVersion]]:
        """Load existing registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                
                # Convert to ModelVersion objects
                registry = {}
                for model_name, versions in data.items():
                    registry[model_name] = {
                        version: ModelVersion(**version_data)
                        for version, version_data in versions.items()
                    }
                return registry
            except Exception as e:
                print(f"⚠️  Error loading registry: {e}")
                return {}
        return {}
    
    def _save_registry(self):
        """Save registry to file."""
        # Convert ModelVersion objects to dicts
        data = {}
        for model_name, versions in self.models.items():
            data[model_name] = {
                version: asdict(version_obj)
                for version, version_obj in versions.items()
            }
        
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _count_total_versions(self) -> int:
        """Count total number of model versions."""
        return sum(len(versions) for versions in self.models.values())
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_model_size(self, model_path: Path) -> float:
        """Get model size in MB."""
        if model_path.is_file():
            return model_path.stat().st_size / (1024 * 1024)
        elif model_path.is_dir():
            total_size = sum(
                f.stat().st_size for f in model_path.rglob('*') if f.is_file()
            )
            return total_size / (1024 * 1024)
        return 0.0
    
    def register_model(
        self,
        model: Union[torch.nn.Module, str],
        name: str,
        version: Optional[str] = None,
        description: str = "",
        tags: List[str] = None,
        hyperparameters: Dict[str, Any] = None,
        metrics: Dict[str, float] = None,
        author: str = "unknown",
        experiment_id: Optional[str] = None,
        artifacts: Dict[str, str] = None,
        parent_models: List[str] = None
    ) -> str:
        """
        Register a new model version.
        
        Args:
            model: PyTorch model or path to saved model
            name: Model name
            version: Version string (auto-generated if None)
            description: Model description
            tags: List of tags
            hyperparameters: Model hyperparameters
            metrics: Performance metrics
            author: Model author
            experiment_id: Associated experiment ID
            artifacts: Additional artifacts (paths)
            parent_models: Parent model IDs for lineage
        
        Returns:
            Model ID (name:version)
        """
        # Auto-generate version if not provided
        if version is None:
            version = self._generate_version(name)
        
        # Create model directory
        model_dir = self.models_dir / name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if isinstance(model, torch.nn.Module):
            model_path = model_dir / "model.pth"
            torch.save(model.state_dict(), model_path)
        elif isinstance(model, str):
            # Copy existing model file
            model_path = model_dir / "model.pth"
            shutil.copy2(model, model_path)
        else:
            raise ValueError("Model must be PyTorch module or path to saved model")
        
        # Calculate checksum
        checksum = self._calculate_checksum(model_path)
        
        # Create metadata
        model_id = f"{name}:{version}"
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            created_at=datetime.now().isoformat(),
            author=author,
            description=description,
            tags=tags or [],
            framework="pytorch",
            algorithm="maddpg",
            hyperparameters=hyperparameters or {},
            metrics=metrics or {},
            artifacts=artifacts or {},
            dependencies=self._get_dependencies(),
            model_size_mb=self._get_model_size(model_path),
            experiment_id=experiment_id,
            parent_models=parent_models
        )
        
        # Save metadata
        metadata_path = self.metadata_dir / f"{name}_{version}.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        # Create version record
        model_version = ModelVersion(
            version=version,
            model_path=str(model_path),
            metadata_path=str(metadata_path),
            checksum=checksum,
            created_at=datetime.now().isoformat()
        )
        
        # Update registry
        if name not in self.models:
            self.models[name] = {}
        self.models[name][version] = model_version
        
        # Save registry
        self._save_registry()
        
        # Log to MLflow if available
        if MLFLOW_AVAILABLE and experiment_id:
            self._log_to_mlflow(metadata, model_path)
        
        print(f"✅ Registered model {model_id}")
        print(f"   - Size: {metadata.model_size_mb:.2f} MB")
        print(f"   - Checksum: {checksum[:8]}...")
        
        return model_id
    
    def _generate_version(self, name: str) -> str:
        """Generate next version number for a model."""
        if name not in self.models or not self.models[name]:
            return "v1.0.0"
        
        # Get latest version
        versions = list(self.models[name].keys())
        versions.sort(key=lambda x: [int(i) for i in x.replace('v', '').split('.')])
        latest = versions[-1]
        
        # Increment patch version
        parts = latest.replace('v', '').split('.')
        parts[-1] = str(int(parts[-1]) + 1)
        return f"v{'.'.join(parts)}"
    
    def _get_dependencies(self) -> Dict[str, str]:
        """Get current Python package versions."""
        import pkg_resources
        
        key_packages = ['torch', 'numpy', 'mlflow', 'wandb', 'optuna']
        dependencies = {}
        
        for package in key_packages:
            try:
                version = pkg_resources.get_distribution(package).version
                dependencies[package] = version
            except pkg_resources.DistributionNotFound:
                dependencies[package] = "not_installed"
        
        return dependencies
    
    def _log_to_mlflow(self, metadata: ModelMetadata, model_path: Path):
        """Log model to MLflow."""
        try:
            with mlflow.start_run(run_id=metadata.experiment_id):
                # Log model
                mlflow.pytorch.log_model(
                    pytorch_model=None,  # We have state dict
                    artifact_path="model",
                    registered_model_name=metadata.name
                )
                
                # Log metadata
                mlflow.log_params(metadata.hyperparameters)
                mlflow.log_metrics(metadata.metrics)
                mlflow.set_tags({
                    "version": metadata.version,
                    "author": metadata.author,
                    "algorithm": metadata.algorithm
                })
        except Exception as e:
            print(f"⚠️  Error logging to MLflow: {e}")
    
    def load_model(
        self,
        name: str,
        version: Optional[str] = None,
        model_class: Optional[type] = None
    ) -> torch.nn.Module:
        """
        Load a model from the registry.
        
        Args:
            name: Model name
            version: Version (latest if None)
            model_class: Model class for initialization
        
        Returns:
            Loaded PyTorch model
        """
        if name not in self.models:
            raise ValueError(f"Model {name} not found in registry")
        
        if version is None:
            # Get latest version
            version = self.get_latest_version(name)
        
        if version not in self.models[name]:
            raise ValueError(f"Version {version} not found for model {name}")
        
        model_version = self.models[name][version]
        model_path = Path(model_version.model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load state dict
        state_dict = torch.load(model_path, map_location='cpu')
        
        if model_class:
            # Initialize model and load state
            model = model_class()
            model.load_state_dict(state_dict)
            return model
        else:
            # Return state dict
            return state_dict
    
    def get_latest_version(self, name: str) -> str:
        """Get latest version of a model."""
        if name not in self.models or not self.models[name]:
            raise ValueError(f"Model {name} not found")
        
        versions = list(self.models[name].keys())
        versions.sort(key=lambda x: [int(i) for i in x.replace('v', '').split('.')])
        return versions[-1]
    
    def get_model_metadata(self, name: str, version: Optional[str] = None) -> ModelMetadata:
        """Get model metadata."""
        if version is None:
            version = self.get_latest_version(name)
        
        if name not in self.models or version not in self.models[name]:
            raise ValueError(f"Model {name}:{version} not found")
        
        model_version = self.models[name][version]
        metadata_path = Path(model_version.metadata_path)
        
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        return ModelMetadata(**metadata_dict)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all models in the registry."""
        models_list = []
        
        for name, versions in self.models.items():
            latest_version = self.get_latest_version(name)
            latest_metadata = self.get_model_metadata(name, latest_version)
            
            models_list.append({
                "name": name,
                "latest_version": latest_version,
                "total_versions": len(versions),
                "size_mb": latest_metadata.model_size_mb,
                "created_at": latest_metadata.created_at,
                "metrics": latest_metadata.metrics,
                "tags": latest_metadata.tags,
                "deployment_status": latest_metadata.deployment_status
            })
        
        return models_list
    
    def list_versions(self, name: str) -> List[Dict[str, Any]]:
        """List all versions of a model."""
        if name not in self.models:
            return []
        
        versions_list = []
        for version, model_version in self.models[name].items():
            metadata = self.get_model_metadata(name, version)
            
            versions_list.append({
                "version": version,
                "created_at": model_version.created_at,
                "size_mb": metadata.model_size_mb,
                "metrics": metadata.metrics,
                "checksum": model_version.checksum[:8],
                "is_active": model_version.is_active,
                "is_production": model_version.is_production
            })
        
        # Sort by creation time
        versions_list.sort(key=lambda x: x["created_at"], reverse=True)
        return versions_list
    
    def compare_models(
        self,
        model_specs: List[tuple],  # [(name, version), ...]
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple model versions.
        
        Args:
            model_specs: List of (name, version) tuples
            metrics: Specific metrics to compare
        
        Returns:
            Comparison results
        """
        comparison = {
            "models": [],
            "metrics_comparison": {},
            "best_model": {},
            "summary": {}
        }
        
        model_data = []
        
        for name, version in model_specs:
            try:
                metadata = self.get_model_metadata(name, version)
                model_data.append({
                    "id": f"{name}:{version}",
                    "name": name,
                    "version": version,
                    "metadata": metadata
                })
            except ValueError as e:
                print(f"⚠️  Skipping {name}:{version} - {e}")
        
        comparison["models"] = model_data
        
        # Compare metrics
        if metrics is None:
            # Get all available metrics
            all_metrics = set()
            for data in model_data:
                all_metrics.update(data["metadata"].metrics.keys())
            metrics = list(all_metrics)
        
        for metric in metrics:
            comparison["metrics_comparison"][metric] = {}
            
            for data in model_data:
                model_id = data["id"]
                metric_value = data["metadata"].metrics.get(metric, None)
                comparison["metrics_comparison"][metric][model_id] = metric_value
            
            # Find best model for this metric
            valid_values = {
                k: v for k, v in comparison["metrics_comparison"][metric].items()
                if v is not None
            }
            
            if valid_values:
                # Assume higher is better (customize as needed)
                best_model_id = max(valid_values.keys(), key=lambda k: valid_values[k])
                comparison["best_model"][metric] = {
                    "model": best_model_id,
                    "value": valid_values[best_model_id]
                }
        
        # Overall summary
        comparison["summary"] = {
            "total_models": len(model_data),
            "metrics_compared": len(metrics),
            "comparison_date": datetime.now().isoformat()
        }
        
        return comparison
    
    def promote_to_staging(self, name: str, version: str) -> bool:
        """Promote model to staging environment."""
        try:
            model_version = self.models[name][version]
            staging_path = self.staging_dir / f"{name}_{version}.pth"
            
            # Copy model to staging
            shutil.copy2(model_version.model_path, staging_path)
            
            # Update metadata
            metadata = self.get_model_metadata(name, version)
            metadata.deployment_status = "staging"
            
            # Save updated metadata
            with open(model_version.metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            
            print(f"✅ Promoted {name}:{version} to staging")
            return True
            
        except Exception as e:
            print(f"❌ Error promoting to staging: {e}")
            return False
    
    def promote_to_production(self, name: str, version: str) -> bool:
        """Promote model to production environment."""
        try:
            model_version = self.models[name][version]
            production_path = self.production_dir / f"{name}_production.pth"
            
            # Copy model to production
            shutil.copy2(model_version.model_path, production_path)
            
            # Update registry
            model_version.is_production = True
            
            # Update metadata
            metadata = self.get_model_metadata(name, version)
            metadata.deployment_status = "production"
            
            # Save updated metadata
            with open(model_version.metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            
            # Update registry
            self._save_registry()
            
            print(f"✅ Promoted {name}:{version} to production")
            return True
            
        except Exception as e:
            print(f"❌ Error promoting to production: {e}")
            return False
    
    def delete_model(self, name: str, version: Optional[str] = None) -> bool:
        """Delete a model or specific version."""
        try:
            if version is None:
                # Delete entire model
                if name in self.models:
                    # Delete all files
                    model_dir = self.models_dir / name
                    if model_dir.exists():
                        shutil.rmtree(model_dir)
                    
                    # Delete metadata files
                    for version_obj in self.models[name].values():
                        metadata_path = Path(version_obj.metadata_path)
                        if metadata_path.exists():
                            metadata_path.unlink()
                    
                    # Remove from registry
                    del self.models[name]
                    self._save_registry()
                    
                    print(f"✅ Deleted model {name} and all versions")
                    return True
            else:
                # Delete specific version
                if name in self.models and version in self.models[name]:
                    model_version = self.models[name][version]
                    
                    # Delete model file
                    model_path = Path(model_version.model_path)
                    if model_path.exists():
                        model_path.unlink()
                    
                    # Delete metadata
                    metadata_path = Path(model_version.metadata_path)
                    if metadata_path.exists():
                        metadata_path.unlink()
                    
                    # Remove from registry
                    del self.models[name][version]
                    
                    # Remove model entirely if no versions left
                    if not self.models[name]:
                        del self.models[name]
                    
                    self._save_registry()
                    
                    print(f"✅ Deleted model version {name}:{version}")
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error deleting model: {e}")
            return False
    
    def export_model(
        self,
        name: str,
        version: str,
        export_path: str,
        format: str = "pytorch"
    ) -> bool:
        """Export model in specified format."""
        try:
            model_version = self.models[name][version]
            export_path = Path(export_path)
            export_path.mkdir(parents=True, exist_ok=True)
            
            if format == "pytorch":
                # Copy PyTorch model
                shutil.copy2(model_version.model_path, export_path / "model.pth")
                
                # Copy metadata
                metadata = self.get_model_metadata(name, version)
                with open(export_path / "metadata.json", 'w') as f:
                    json.dump(asdict(metadata), f, indent=2)
                
            elif format == "onnx":
                # TODO: Implement ONNX export
                print("⚠️  ONNX export not yet implemented")
                return False
            
            print(f"✅ Exported {name}:{version} to {export_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error exporting model: {e}")
            return False


# Convenience function for easy registry setup
def create_model_registry(registry_path: str = "model_registry") -> ModelRegistry:
    """
    Create and initialize a model registry.
    
    Args:
        registry_path: Path for registry storage
    
    Returns:
        Initialized ModelRegistry instance
    """
    return ModelRegistry(registry_path)

