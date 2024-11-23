"""
FractiMonitor.py

Implements comprehensive monitoring framework for FractiAI, enabling real-time
monitoring and observability across scales and dimensions through fractal patterns.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import time
from collections import deque
import pandas as pd
from scipy.stats import entropy
import networkx as nx
from prometheus_client import Counter, Gauge, Histogram, Summary

logger = logging.getLogger(__name__)

# Prometheus metrics
PATTERN_COUNT = Counter('fractiai_patterns_total', 'Total pattern count')
SYSTEM_HEALTH = Gauge('fractiai_system_health', 'Overall system health')
COMPONENT_LATENCY = Histogram('fractiai_component_latency', 'Component latency')
RESOURCE_USAGE = Summary('fractiai_resource_usage', 'Resource usage metrics')

@dataclass
class MonitorConfig:
    """Configuration for monitoring framework"""
    sampling_interval: float = 1.0
    history_length: int = 1000
    alert_threshold: float = 0.7
    pattern_threshold: float = 0.8
    fractal_depth: int = 3

class PatternMonitor:
    """Monitors fractal patterns in system behavior"""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.pattern_history = deque(maxlen=config.history_length)
        self.alerts = deque(maxlen=config.history_length)
        
    def monitor_pattern(self, pattern: np.ndarray) -> Dict[str, Any]:
        """Monitor and analyze pattern"""
        # Store pattern
        self.pattern_history.append(pattern)
        
        # Analyze pattern
        metrics = self._compute_pattern_metrics(pattern)
        
        # Check for alerts
        alerts = self._check_pattern_alerts(metrics)
        if alerts:
            self.alerts.extend(alerts)
            
        # Update Prometheus metrics
        PATTERN_COUNT.inc()
        
        return {
            'metrics': metrics,
            'alerts': alerts
        }
        
    def _compute_pattern_metrics(self, pattern: np.ndarray) -> Dict[str, float]:
        """Compute pattern metrics"""
        return {
            'complexity': float(entropy(pattern / np.sum(pattern))),
            'stability': self._compute_stability(),
            'fractal_dimension': self._compute_fractal_dimension(pattern)
        }
        
    def _compute_stability(self) -> float:
        """Compute pattern stability"""
        if len(self.pattern_history) < 2:
            return 1.0
            
        patterns = np.array(list(self.pattern_history))
        return float(1.0 - np.std(patterns) / (np.mean(np.abs(patterns)) + 1e-7))
        
    def _compute_fractal_dimension(self, pattern: np.ndarray) -> float:
        """Compute fractal dimension"""
        scales = np.logspace(0, 2, num=10, base=2, dtype=int)
        counts = []
        
        for scale in scales:
            scaled = self._rescale_pattern(pattern, scale)
            counts.append(len(np.unique(scaled)))
            
        if len(counts) > 1:
            coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
            return float(-coeffs[0])
        return 1.0
        
    def _rescale_pattern(self, pattern: np.ndarray, scale: int) -> np.ndarray:
        """Rescale pattern to given scale"""
        if len(pattern) < scale:
            return pattern
            
        return np.array([
            np.mean(chunk) 
            for chunk in np.array_split(pattern, scale)
        ])
        
    def _check_pattern_alerts(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for pattern-related alerts"""
        alerts = []
        
        if metrics['complexity'] > self.config.pattern_threshold:
            alerts.append({
                'type': 'pattern_complexity',
                'level': 'warning',
                'message': 'High pattern complexity detected',
                'value': metrics['complexity']
            })
            
        if metrics['stability'] < self.config.alert_threshold:
            alerts.append({
                'type': 'pattern_stability',
                'level': 'warning',
                'message': 'Low pattern stability detected',
                'value': metrics['stability']
            })
            
        return alerts

class ComponentMonitor:
    """Monitors individual system components"""
    
    def __init__(self, component_id: str, config: MonitorConfig):
        self.component_id = component_id
        self.config = config
        self.metrics_history = deque(maxlen=config.history_length)
        self.latency_history = deque(maxlen=config.history_length)
        self.error_history = deque(maxlen=config.history_length)
        
    def record_metrics(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Record component metrics"""
        # Store metrics
        timestamp = time.time()
        record = {
            'timestamp': timestamp,
            'metrics': metrics
        }
        self.metrics_history.append(record)
        
        # Analyze metrics
        analysis = self._analyze_metrics(metrics)
        
        # Check for alerts
        alerts = self._check_component_alerts(analysis)
        
        return {
            'analysis': analysis,
            'alerts': alerts
        }
        
    def record_latency(self, duration: float) -> None:
        """Record operation latency"""
        self.latency_history.append(duration)
        COMPONENT_LATENCY.observe(duration)
        
    def record_error(self, error: Exception) -> None:
        """Record component error"""
        self.error_history.append({
            'timestamp': time.time(),
            'error': str(error)
        })
        
    def _analyze_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Analyze component metrics"""
        if not self.metrics_history:
            return {}
            
        df = pd.DataFrame([m['metrics'] for m in self.metrics_history])
        
        return {
            'stability': self._compute_metric_stability(df),
            'trend': self._compute_metric_trends(df),
            'health': self._compute_component_health(metrics)
        }
        
    def _compute_metric_stability(self, df: pd.DataFrame) -> float:
        """Compute metric stability"""
        if df.empty:
            return 1.0
            
        stabilities = []
        for column in df.columns:
            values = df[column].values
            stability = 1.0 - np.std(values) / (np.mean(np.abs(values)) + 1e-7)
            stabilities.append(stability)
            
        return float(np.mean(stabilities))
        
    def _compute_metric_trends(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute metric trends"""
        if df.empty:
            return {}
            
        trends = {}
        for column in df.columns:
            values = df[column].values
            if len(values) > 1:
                trend = np.mean(np.diff(values))
                trends[column] = float(trend)
                
        return trends
        
    def _compute_component_health(self, metrics: Dict[str, float]) -> float:
        """Compute overall component health"""
        if not metrics:
            return 0.0
            
        # Combine multiple health indicators
        indicators = [
            self._compute_metric_stability(pd.DataFrame([metrics])),
            1.0 - len(self.error_history) / self.config.history_length,
            self._compute_latency_health()
        ]
        
        health = np.mean(indicators)
        return float(health)
        
    def _compute_latency_health(self) -> float:
        """Compute latency-based health"""
        if not self.latency_history:
            return 1.0
            
        latencies = np.array(list(self.latency_history))
        return float(1.0 - np.std(latencies) / (np.mean(latencies) + 1e-7))
        
    def _check_component_alerts(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for component-related alerts"""
        alerts = []
        
        if analysis['health'] < self.config.alert_threshold:
            alerts.append({
                'type': 'component_health',
                'level': 'warning',
                'message': f'Low health detected for component {self.component_id}',
                'value': analysis['health']
            })
            
        if analysis['stability'] < self.config.alert_threshold:
            alerts.append({
                'type': 'component_stability',
                'level': 'warning',
                'message': f'Low stability detected for component {self.component_id}',
                'value': analysis['stability']
            })
            
        return alerts

class FractiMonitor:
    """Main monitoring system with fractal awareness"""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.pattern_monitor = PatternMonitor(config)
        self.component_monitors: Dict[str, ComponentMonitor] = {}
        self.system_metrics = deque(maxlen=config.history_length)
        
    def add_component(self, component_id: str) -> None:
        """Add component to monitoring"""
        self.component_monitors[component_id] = ComponentMonitor(
            component_id, 
            self.config
        )
        
    def monitor(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor system state"""
        # Monitor patterns
        patterns = self._extract_patterns(state)
        pattern_results = {
            pattern_id: self.pattern_monitor.monitor_pattern(pattern)
            for pattern_id, pattern in patterns.items()
        }
        
        # Monitor components
        component_results = {}
        for component_id, data in state.items():
            if component_id in self.component_monitors:
                if isinstance(data, dict) and 'metrics' in data:
                    result = self.component_monitors[component_id].record_metrics(
                        data['metrics']
                    )
                    component_results[component_id] = result
                    
        # Compute system metrics
        system_metrics = self._compute_system_metrics(
            pattern_results,
            component_results
        )
        
        # Store system metrics
        self.system_metrics.append(system_metrics)
        
        # Update Prometheus metrics
        SYSTEM_HEALTH.set(system_metrics['health'])
        
        return {
            'patterns': pattern_results,
            'components': component_results,
            'system': system_metrics,
            'alerts': self._aggregate_alerts(pattern_results, component_results)
        }
        
    def _extract_patterns(self, state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract patterns from state"""
        patterns = {}
        
        def extract_recursive(data: Any, path: str = '') -> None:
            if isinstance(data, np.ndarray):
                patterns[path] = data
            elif isinstance(data, dict):
                for key, value in data.items():
                    new_path = f"{path}_{key}" if path else key
                    extract_recursive(value, new_path)
                    
        extract_recursive(state)
        return patterns
        
    def _compute_system_metrics(self,
                              pattern_results: Dict[str, Dict[str, Any]],
                              component_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Compute system-wide metrics"""
        # Pattern metrics
        pattern_metrics = [
            result['metrics']
            for result in pattern_results.values()
        ]
        
        pattern_health = np.mean([
            metrics['stability']
            for metrics in pattern_metrics
        ]) if pattern_metrics else 1.0
        
        # Component metrics
        component_health = np.mean([
            result['analysis']['health']
            for result in component_results.values()
        ]) if component_results else 1.0
        
        # System metrics
        return {
            'health': float((pattern_health + component_health) / 2),
            'pattern_health': float(pattern_health),
            'component_health': float(component_health),
            'stability': self._compute_system_stability()
        }
        
    def _compute_system_stability(self) -> float:
        """Compute overall system stability"""
        if len(self.system_metrics) < 2:
            return 1.0
            
        metrics = list(self.system_metrics)
        health_values = [m['health'] for m in metrics]
        
        return float(1.0 - np.std(health_values) / (np.mean(health_values) + 1e-7))
        
    def _aggregate_alerts(self,
                         pattern_results: Dict[str, Dict[str, Any]],
                         component_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate alerts from all sources"""
        alerts = []
        
        # Pattern alerts
        for pattern_id, result in pattern_results.items():
            for alert in result['alerts']:
                alert['source'] = f"pattern_{pattern_id}"
                alerts.append(alert)
                
        # Component alerts
        for component_id, result in component_results.items():
            for alert in result['alerts']:
                alert['source'] = f"component_{component_id}"
                alerts.append(alert)
                
        return sorted(alerts, key=lambda x: x['level'], reverse=True) 