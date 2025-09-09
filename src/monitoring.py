"""
Model Performance Monitoring and Alerting System
===============================================

This module provides real-time monitoring, alerting, and performance
tracking for deployed MADDPG models.
"""

import time
import json
import threading
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from dataclasses import dataclass, asdict

# Monitoring libraries
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not available. Install with: pip install psutil")

try:
    import prometheus_client
    from prometheus_client import Gauge, Counter, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Prometheus client not available. Install with: pip install prometheus-client")


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""
    timestamp: str
    model_version: str
    inference_time: float
    prediction_score: float
    input_size: int
    memory_usage: float
    cpu_usage: float
    gpu_usage: Optional[float] = None
    error_count: int = 0
    success_count: int = 0


@dataclass
class AlertRule:
    """Configuration for monitoring alerts."""
    name: str
    metric: str
    operator: str  # 'gt', 'lt', 'eq', 'gte', 'lte'
    threshold: float
    duration: int  # seconds
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    enabled: bool = True


class PerformanceMonitor:
    """
    Real-time performance monitoring system for MADDPG models.
    
    Features:
    - Real-time metric collection
    - Performance alerting
    - Drift detection
    - Resource monitoring
    - Custom dashboard generation
    """
    
    def __init__(
        self,
        model_name: str = "maddpg_tennis",
        monitoring_interval: int = 30,
        retention_days: int = 30,
        enable_prometheus: bool = True
    ):
        """
        Initialize performance monitor.
        
        Args:
            model_name: Name of the model being monitored
            monitoring_interval: Metrics collection interval in seconds
            retention_days: How long to keep metrics data
            enable_prometheus: Whether to export Prometheus metrics
        """
        self.model_name = model_name
        self.monitoring_interval = monitoring_interval
        self.retention_days = retention_days
        self.enable_prometheus = enable_prometheus
        
        # Metrics storage
        self.metrics_history: List[ModelMetrics] = []
        self.alerts_history: List[Dict[str, Any]] = []
        
        # Alert rules
        self.alert_rules: List[AlertRule] = []
        self._setup_default_alert_rules()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Prometheus metrics (if enabled)
        if self.enable_prometheus and PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()
        
        # Create monitoring directory
        self.monitoring_dir = Path("monitoring")
        self.monitoring_dir.mkdir(exist_ok=True)
        
        print(f"✅ Performance Monitor initialized for {model_name}")
        print(f"   - Monitoring interval: {monitoring_interval}s")
        print(f"   - Retention: {retention_days} days")
        print(f"   - Prometheus: {'Enabled' if enable_prometheus else 'Disabled'}")
    
    def _setup_default_alert_rules(self):
        """Setup default alerting rules."""
        self.alert_rules = [
            AlertRule(
                name="High Inference Time",
                metric="inference_time",
                operator="gt",
                threshold=1.0,  # 1 second
                duration=60,
                severity="medium",
                description="Model inference time is too high"
            ),
            AlertRule(
                name="Low Prediction Score",
                metric="prediction_score",
                operator="lt",
                threshold=0.1,
                duration=120,
                severity="high",
                description="Model prediction scores are consistently low"
            ),
            AlertRule(
                name="High Memory Usage",
                metric="memory_usage",
                operator="gt",
                threshold=80.0,  # 80%
                duration=300,
                severity="high",
                description="System memory usage is critically high"
            ),
            AlertRule(
                name="High Error Rate",
                metric="error_rate",
                operator="gt",
                threshold=0.05,  # 5%
                duration=180,
                severity="critical",
                description="Model error rate is too high"
            )
        ]
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics collectors."""
        self.prometheus_metrics = {
            'inference_time': Histogram(
                'model_inference_time_seconds',
                'Model inference time in seconds',
                ['model_name', 'version']
            ),
            'prediction_score': Gauge(
                'model_prediction_score',
                'Model prediction score',
                ['model_name', 'version']
            ),
            'memory_usage': Gauge(
                'system_memory_usage_percent',
                'System memory usage percentage'
            ),
            'cpu_usage': Gauge(
                'system_cpu_usage_percent',
                'System CPU usage percentage'
            ),
            'error_count': Counter(
                'model_errors_total',
                'Total number of model errors',
                ['model_name', 'version', 'error_type']
            ),
            'success_count': Counter(
                'model_predictions_total',
                'Total number of successful predictions',
                ['model_name', 'version']
            )
        }
    
    def log_prediction(
        self,
        model_version: str,
        inference_time: float,
        prediction_score: float,
        input_size: int,
        success: bool = True,
        error_type: Optional[str] = None
    ):
        """
        Log a model prediction for monitoring.
        
        Args:
            model_version: Version of the model used
            inference_time: Time taken for inference
            prediction_score: Score/confidence of prediction
            input_size: Size of input data
            success: Whether prediction was successful
            error_type: Type of error if prediction failed
        """
        # Collect system metrics
        memory_usage = self._get_memory_usage()
        cpu_usage = self._get_cpu_usage()
        gpu_usage = self._get_gpu_usage()
        
        # Create metrics object
        metrics = ModelMetrics(
            timestamp=datetime.now().isoformat(),
            model_version=model_version,
            inference_time=inference_time,
            prediction_score=prediction_score,
            input_size=input_size,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            gpu_usage=gpu_usage,
            error_count=0 if success else 1,
            success_count=1 if success else 0
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Update Prometheus metrics
        if self.enable_prometheus and PROMETHEUS_AVAILABLE:
            self._update_prometheus_metrics(metrics, error_type)
        
        # Check for alerts
        self._check_alerts(metrics)
        
        # Clean up old metrics
        self._cleanup_old_metrics()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        if PSUTIL_AVAILABLE:
            return psutil.virtual_memory().percent
        return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        if PSUTIL_AVAILABLE:
            return psutil.cpu_percent(interval=1)
        return 0.0
    
    def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage if available."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUtil * 100
        except ImportError:
            pass
        return None
    
    def _update_prometheus_metrics(self, metrics: ModelMetrics, error_type: Optional[str]):
        """Update Prometheus metrics."""
        labels = {'model_name': self.model_name, 'version': metrics.model_version}
        
        self.prometheus_metrics['inference_time'].labels(**labels).observe(metrics.inference_time)
        self.prometheus_metrics['prediction_score'].labels(**labels).set(metrics.prediction_score)
        self.prometheus_metrics['memory_usage'].set(metrics.memory_usage)
        self.prometheus_metrics['cpu_usage'].set(metrics.cpu_usage)
        
        if metrics.success_count > 0:
            self.prometheus_metrics['success_count'].labels(**labels).inc()
        
        if metrics.error_count > 0:
            error_labels = {**labels, 'error_type': error_type or 'unknown'}
            self.prometheus_metrics['error_count'].labels(**error_labels).inc()
    
    def _check_alerts(self, metrics: ModelMetrics):
        """Check if any alert rules are triggered."""
        current_time = datetime.now()
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            # Calculate current metric value
            metric_value = self._get_metric_value(metrics, rule.metric)
            
            # Check if threshold is exceeded
            if self._evaluate_condition(metric_value, rule.operator, rule.threshold):
                # Check if condition persists for required duration
                if self._check_alert_duration(rule, current_time):
                    self._trigger_alert(rule, metric_value, current_time)
    
    def _get_metric_value(self, metrics: ModelMetrics, metric_name: str) -> float:
        """Get metric value by name."""
        if metric_name == "error_rate":
            # Calculate error rate from recent metrics
            recent_metrics = self.get_recent_metrics(minutes=5)
            if not recent_metrics:
                return 0.0
            
            total_requests = sum(m.success_count + m.error_count for m in recent_metrics)
            total_errors = sum(m.error_count for m in recent_metrics)
            
            return (total_errors / total_requests) if total_requests > 0 else 0.0
        
        return getattr(metrics, metric_name, 0.0)
    
    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        if operator == "gt":
            return value > threshold
        elif operator == "lt":
            return value < threshold
        elif operator == "gte":
            return value >= threshold
        elif operator == "lte":
            return value <= threshold
        elif operator == "eq":
            return value == threshold
        return False
    
    def _check_alert_duration(self, rule: AlertRule, current_time: datetime) -> bool:
        """Check if alert condition has persisted for required duration."""
        # Simplified duration check - in production, implement proper state tracking
        return True
    
    def _trigger_alert(self, rule: AlertRule, metric_value: float, timestamp: datetime):
        """Trigger an alert."""
        alert = {
            'rule_name': rule.name,
            'metric': rule.metric,
            'value': metric_value,
            'threshold': rule.threshold,
            'severity': rule.severity,
            'description': rule.description,
            'timestamp': timestamp.isoformat(),
            'model_name': self.model_name
        }
        
        self.alerts_history.append(alert)
        
        # Log alert
        print(f"🚨 ALERT [{rule.severity.upper()}]: {rule.name}")
        print(f"   {rule.description}")
        print(f"   Current value: {metric_value:.4f}, Threshold: {rule.threshold:.4f}")
        print(f"   Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save alert to file
        self._save_alert(alert)
        
        # Send notifications (implement as needed)
        self._send_alert_notification(alert)
    
    def _save_alert(self, alert: Dict[str, Any]):
        """Save alert to file."""
        alerts_file = self.monitoring_dir / "alerts.jsonl"
        with open(alerts_file, "a") as f:
            f.write(json.dumps(alert) + "\n")
    
    def _send_alert_notification(self, alert: Dict[str, Any]):
        """Send alert notification (email, Slack, etc.)."""
        # Implement notification logic here
        # For now, just save to a notifications file
        notifications_file = self.monitoring_dir / "notifications.jsonl"
        notification = {
            **alert,
            'notification_sent': datetime.now().isoformat(),
            'channels': ['console', 'file']  # Add email, slack, etc.
        }
        
        with open(notifications_file, "a") as f:
            f.write(json.dumps(notification) + "\n")
    
    def get_recent_metrics(self, minutes: int = 60) -> List[ModelMetrics]:
        """Get metrics from the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        return [
            m for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp) >= cutoff_time
        ]
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours."""
        recent_metrics = self.get_recent_metrics(minutes=hours * 60)
        
        if not recent_metrics:
            return {"error": "No metrics available"}
        
        inference_times = [m.inference_time for m in recent_metrics]
        prediction_scores = [m.prediction_score for m in recent_metrics]
        memory_usage = [m.memory_usage for m in recent_metrics]
        
        total_requests = len(recent_metrics)
        total_errors = sum(m.error_count for m in recent_metrics)
        error_rate = (total_errors / total_requests) if total_requests > 0 else 0.0
        
        return {
            "period_hours": hours,
            "total_requests": total_requests,
            "error_rate": error_rate,
            "avg_inference_time": np.mean(inference_times),
            "p95_inference_time": np.percentile(inference_times, 95),
            "avg_prediction_score": np.mean(prediction_scores),
            "min_prediction_score": np.min(prediction_scores),
            "max_prediction_score": np.max(prediction_scores),
            "avg_memory_usage": np.mean(memory_usage),
            "max_memory_usage": np.max(memory_usage),
            "active_alerts": len([a for a in self.alerts_history if self._is_recent_alert(a, hours)])
        }
    
    def _is_recent_alert(self, alert: Dict[str, Any], hours: int) -> bool:
        """Check if alert is recent."""
        alert_time = datetime.fromisoformat(alert['timestamp'])
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return alert_time >= cutoff_time
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff_time = datetime.now() - timedelta(days=self.retention_days)
        
        self.metrics_history = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp) >= cutoff_time
        ]
        
        self.alerts_history = [
            a for a in self.alerts_history
            if datetime.fromisoformat(a['timestamp']) >= cutoff_time
        ]
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        if self.is_monitoring:
            print("⚠️  Monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        print(f"🔍 Started monitoring {self.model_name}")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        print(f"⏹️  Stopped monitoring {self.model_name}")
    
    def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                metrics = ModelMetrics(
                    timestamp=datetime.now().isoformat(),
                    model_version="monitoring",
                    inference_time=0.0,
                    prediction_score=0.0,
                    input_size=0,
                    memory_usage=self._get_memory_usage(),
                    cpu_usage=self._get_cpu_usage(),
                    gpu_usage=self._get_gpu_usage(),
                    error_count=0,
                    success_count=0
                )
                
                # Check system-level alerts
                self._check_alerts(metrics)
                
                # Sleep until next monitoring interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        if format == "json":
            return json.dumps([asdict(m) for m in self.metrics_history], indent=2)
        elif format == "csv":
            # Implement CSV export
            import pandas as pd
            df = pd.DataFrame([asdict(m) for m in self.metrics_history])
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Convenience function for easy monitoring setup
def setup_monitoring(
    model_name: str = "maddpg_tennis",
    enable_prometheus: bool = True,
    start_immediately: bool = True
) -> PerformanceMonitor:
    """
    Quick setup for model monitoring.
    
    Args:
        model_name: Name of the model to monitor
        enable_prometheus: Whether to enable Prometheus metrics
        start_immediately: Whether to start monitoring immediately
    
    Returns:
        Configured PerformanceMonitor instance
    """
    monitor = PerformanceMonitor(
        model_name=model_name,
        enable_prometheus=enable_prometheus
    )
    
    if start_immediately:
        monitor.start_monitoring()
    
    return monitor

