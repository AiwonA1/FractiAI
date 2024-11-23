"""
FractiMetrics.py

Implements comprehensive performance monitoring and metrics collection for FractiAI,
tracking system health, efficiency, and harmony across all components.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import time
from collections import deque
import json

logger = logging.getLogger(__name__)

@dataclass
class MetricsConfig:
    """Configuration for FractiAI metrics"""
    history_size: int = 1000
    update_interval: float = 1.0  # seconds
    alert_threshold: float = 0.7
    save_interval: int = 100
    dimension_weights: Dict[str, float] = None

class MetricCollector:
    """Collects and processes specific types of metrics"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.values = deque(maxlen=capacity)
        self.timestamps = deque(maxlen=capacity)
        self.statistics = {}
        
    def add_value(self, value: float) -> None:
        """Add new metric value"""
        self.values.append(value)
        self.timestamps.append(time.time())
        self._update_statistics()
        
    def get_recent(self, window: int = None) -> List[float]:
        """Get recent metric values"""
        if window is None or window > len(self.values):
            return list(self.values)
        return list(self.values)[-window:]
    
    def _update_statistics(self) -> None:
        """Update metric statistics"""
        if not self.values:
            return
            
        values = np.array(self.values)
        self.statistics = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'trend': self._compute_trend(values)
        }
    
    def _compute_trend(self, values: np.ndarray) -> float:
        """Compute metric trend"""
        if len(values) < 2:
            return 0.0
        return float(np.mean(np.diff(values)))

class ComponentMetrics:
    """Tracks metrics for a specific component"""
    
    def __init__(self, component_id: str, config: MetricsConfig):
        self.component_id = component_id
        self.config = config
        self.collectors = {
            'performance': MetricCollector(config.history_size),
            'efficiency': MetricCollector(config.history_size),
            'harmony': MetricCollector(config.history_size),
            'resource_usage': MetricCollector(config.history_size)
        }
        self.last_update = time.time()
        
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update component metrics"""
        current_time = time.time()
        if current_time - self.last_update < self.config.update_interval:
            return
            
        for metric_type, value in metrics.items():
            if metric_type in self.collectors:
                self.collectors[metric_type].add_value(value)
                
        self.last_update = current_time
        
    def get_summary(self) -> Dict[str, Any]:
        """Get component metrics summary"""
        return {
            'component_id': self.component_id,
            'metrics': {
                metric_type: collector.statistics
                for metric_type, collector in self.collectors.items()
            },
            'health_score': self._compute_health_score(),
            'last_update': self.last_update
        }
    
    def _compute_health_score(self) -> float:
        """Compute overall component health score"""
        scores = []
        
        for collector in self.collectors.values():
            if collector.statistics:
                # Normalize metric to [0, 1] range
                mean = collector.statistics['mean']
                std = collector.statistics['std']
                min_val = collector.statistics['min']
                max_val = collector.statistics['max']
                
                if max_val > min_val:
                    normalized_score = (mean - min_val) / (max_val - min_val)
                    scores.append(normalized_score)
                    
        return float(np.mean(scores)) if scores else 0.0

class FractiMetrics:
    """Main metrics system for FractiAI"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.component_metrics = {}
        self.system_metrics = ComponentMetrics('system', config)
        self.alerts = []
        self.last_save = time.time()
        
    def update_component_metrics(self, component_id: str, 
                               metrics: Dict[str, float]) -> None:
        """Update metrics for specific component"""
        if component_id not in self.component_metrics:
            self.component_metrics[component_id] = ComponentMetrics(
                component_id, self.config
            )
            
        self.component_metrics[component_id].update_metrics(metrics)
        self._check_alerts(component_id)
        
        # Update system-wide metrics
        self._update_system_metrics()
        
        # Save metrics if needed
        self._auto_save()
        
    def get_component_metrics(self, component_id: str) -> Optional[Dict]:
        """Get metrics for specific component"""
        if component_id in self.component_metrics:
            return self.component_metrics[component_id].get_summary()
        return None
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics"""
        return {
            'system_summary': self.system_metrics.get_summary(),
            'component_health': self._get_component_health(),
            'alerts': self.alerts[-10:],  # Last 10 alerts
            'global_status': self._compute_global_status()
        }
    
    def _update_system_metrics(self) -> None:
        """Update system-wide metrics"""
        system_metrics = {
            'performance': self._compute_system_performance(),
            'efficiency': self._compute_system_efficiency(),
            'harmony': self._compute_system_harmony(),
            'resource_usage': self._compute_resource_usage()
        }
        
        self.system_metrics.update_metrics(system_metrics)
    
    def _compute_system_performance(self) -> float:
        """Compute overall system performance"""
        performances = []
        
        for component in self.component_metrics.values():
            if 'performance' in component.collectors:
                stats = component.collectors['performance'].statistics
                if 'mean' in stats:
                    performances.append(stats['mean'])
                    
        return float(np.mean(performances)) if performances else 0.0
    
    def _compute_system_efficiency(self) -> float:
        """Compute overall system efficiency"""
        efficiencies = []
        
        for component in self.component_metrics.values():
            if 'efficiency' in component.collectors:
                stats = component.collectors['efficiency'].statistics
                if 'mean' in stats:
                    efficiencies.append(stats['mean'])
                    
        return float(np.mean(efficiencies)) if efficiencies else 0.0
    
    def _compute_system_harmony(self) -> float:
        """Compute overall system harmony"""
        harmonies = []
        
        for component in self.component_metrics.values():
            if 'harmony' in component.collectors:
                stats = component.collectors['harmony'].statistics
                if 'mean' in stats:
                    harmonies.append(stats['mean'])
                    
        return float(np.mean(harmonies)) if harmonies else 0.0
    
    def _compute_resource_usage(self) -> float:
        """Compute overall resource usage"""
        usages = []
        
        for component in self.component_metrics.values():
            if 'resource_usage' in component.collectors:
                stats = component.collectors['resource_usage'].statistics
                if 'mean' in stats:
                    usages.append(stats['mean'])
                    
        return float(np.mean(usages)) if usages else 0.0
    
    def _check_alerts(self, component_id: str) -> None:
        """Check for metric alerts"""
        component = self.component_metrics[component_id]
        health_score = component._compute_health_score()
        
        if health_score < self.config.alert_threshold:
            alert = {
                'timestamp': time.time(),
                'component_id': component_id,
                'health_score': health_score,
                'message': f"Component health below threshold: {health_score:.2f}"
            }
            self.alerts.append(alert)
            logger.warning(f"Alert: {alert['message']}")
    
    def _get_component_health(self) -> Dict[str, float]:
        """Get health scores for all components"""
        return {
            component_id: metrics._compute_health_score()
            for component_id, metrics in self.component_metrics.items()
        }
    
    def _compute_global_status(self) -> Dict[str, Any]:
        """Compute global system status"""
        component_health = self._get_component_health()
        
        return {
            'health': float(np.mean(list(component_health.values()))),
            'component_count': len(self.component_metrics),
            'alert_count': len(self.alerts),
            'status': 'healthy' if np.mean(list(component_health.values())) >= 
                                  self.config.alert_threshold else 'degraded'
        }
    
    def _auto_save(self) -> None:
        """Automatically save metrics to file"""
        current_time = time.time()
        if current_time - self.last_save > self.config.save_interval:
            self.save_metrics()
            self.last_save = current_time
    
    def save_metrics(self, filename: Optional[str] = None) -> None:
        """Save metrics to file"""
        if filename is None:
            filename = f"fractiai_metrics_{int(time.time())}.json"
            
        try:
            metrics_data = {
                'timestamp': time.time(),
                'system_metrics': self.get_system_metrics(),
                'component_metrics': {
                    component_id: metrics.get_summary()
                    for component_id, metrics in self.component_metrics.items()
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}") 