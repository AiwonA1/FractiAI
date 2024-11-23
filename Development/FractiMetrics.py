"""
FractiMetrics.py

Implements comprehensive metrics and monitoring framework for FractiAI, enabling
pattern-aware metrics collection and analysis across scales and dimensions.
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
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics
PATTERN_COMPLEXITY = Gauge('fractiai_pattern_complexity', 'Pattern complexity metric')
SYSTEM_COHERENCE = Gauge('fractiai_system_coherence', 'System coherence metric')
PROCESSING_TIME = Histogram('fractiai_processing_time', 'Processing time in seconds')
MEMORY_USAGE = Gauge('fractiai_memory_usage_bytes', 'Memory usage in bytes')
ERROR_COUNT = Counter('fractiai_errors_total', 'Total error count')

@dataclass
class MetricsConfig:
    """Configuration for metrics framework"""
    collection_interval: float = 1.0
    history_length: int = 1000
    pattern_threshold: float = 0.7
    fractal_depth: int = 3
    aggregation_window: int = 60

class PatternMetrics:
    """Collects and analyzes pattern-related metrics"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.pattern_history = deque(maxlen=config.history_length)
        self.complexity_history = deque(maxlen=config.history_length)
        
    def record_pattern(self, pattern: np.ndarray) -> Dict[str, float]:
        """Record and analyze pattern"""
        # Store pattern
        self.pattern_history.append(pattern)
        
        # Compute metrics
        metrics = self._compute_pattern_metrics(pattern)
        
        # Update Prometheus metrics
        PATTERN_COMPLEXITY.set(metrics['complexity'])
        
        return metrics
        
    def _compute_pattern_metrics(self, pattern: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive pattern metrics"""
        # Compute base metrics
        complexity = self._compute_complexity(pattern)
        self.complexity_history.append(complexity)
        
        return {
            'complexity': float(complexity),
            'stability': self._compute_stability(),
            'fractal_dimension': self._compute_fractal_dimension(pattern),
            'pattern_strength': self._compute_pattern_strength(pattern)
        }
        
    def _compute_complexity(self, pattern: np.ndarray) -> float:
        """Compute pattern complexity"""
        return float(entropy(pattern / np.sum(pattern)))
        
    def _compute_stability(self) -> float:
        """Compute pattern stability over time"""
        if len(self.complexity_history) < 2:
            return 1.0
        return float(1.0 - np.std(self.complexity_history) / 
                    (np.mean(self.complexity_history) + 1e-7))
        
    def _compute_fractal_dimension(self, pattern: np.ndarray) -> float:
        """Compute fractal dimension of pattern"""
        scales = np.logspace(0, 2, num=10, base=2, dtype=int)
        counts = []
        
        for scale in scales:
            scaled = self._rescale_pattern(pattern, scale)
            counts.append(len(np.unique(scaled)))
            
        if len(counts) > 1:
            coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
            return float(-coeffs[0])
        return 1.0
        
    def _compute_pattern_strength(self, pattern: np.ndarray) -> float:
        """Compute pattern strength metric"""
        if len(self.pattern_history) < 2:
            return 0.0
            
        similarities = []
        for hist_pattern in self.pattern_history:
            similarity = np.corrcoef(pattern, hist_pattern)[0,1]
            similarities.append(abs(similarity))
            
        return float(np.mean(similarities))
        
    def _rescale_pattern(self, pattern: np.ndarray, scale: int) -> np.ndarray:
        """Rescale pattern to given scale"""
        if len(pattern) < scale:
            return pattern
            
        return np.array([
            np.mean(chunk) 
            for chunk in np.array_split(pattern, scale)
        ])

class SystemMetrics:
    """Collects and analyzes system-wide metrics"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.metrics_history = deque(maxlen=config.history_length)
        self.component_metrics = {}
        
    def record_metrics(self, metrics: Dict[str, float],
                      component: Optional[str] = None) -> Dict[str, Any]:
        """Record and analyze system metrics"""
        # Store metrics
        timestamp = time.time()
        record = {
            'timestamp': timestamp,
            'metrics': metrics,
            'component': component
        }
        self.metrics_history.append(record)
        
        # Update component metrics
        if component:
            if component not in self.component_metrics:
                self.component_metrics[component] = []
            self.component_metrics[component].append(record)
            
        # Compute analysis
        analysis = self._analyze_metrics(metrics)
        
        # Update Prometheus metrics
        SYSTEM_COHERENCE.set(analysis['coherence'])
        
        return analysis
        
    def _analyze_metrics(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze system metrics"""
        return {
            'coherence': self._compute_coherence(metrics),
            'stability': self._compute_stability(),
            'trends': self._compute_trends(),
            'anomalies': self._detect_anomalies(metrics)
        }
        
    def _compute_coherence(self, metrics: Dict[str, float]) -> float:
        """Compute system coherence metric"""
        if not metrics:
            return 0.0
            
        # Compute correlation between metrics
        values = np.array(list(metrics.values()))
        if len(values) > 1:
            correlations = np.corrcoef(values)
            return float(np.mean(np.abs(correlations)))
        return 1.0
        
    def _compute_stability(self) -> float:
        """Compute system stability over time"""
        if len(self.metrics_history) < 2:
            return 1.0
            
        recent = list(self.metrics_history)[-self.config.aggregation_window:]
        values = [list(r['metrics'].values()) for r in recent]
        
        if not values:
            return 1.0
            
        values = np.array(values)
        return float(1.0 - np.mean(np.std(values, axis=0)) / 
                    (np.mean(np.abs(values)) + 1e-7))
        
    def _compute_trends(self) -> Dict[str, float]:
        """Compute metric trends"""
        if len(self.metrics_history) < 2:
            return {}
            
        recent = list(self.metrics_history)[-self.config.aggregation_window:]
        df = pd.DataFrame([r['metrics'] for r in recent])
        
        trends = {}
        for column in df.columns:
            values = df[column].values
            if len(values) > 1:
                trend = np.mean(np.diff(values))
                trends[column] = float(trend)
                
        return trends
        
    def _detect_anomalies(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect metric anomalies"""
        anomalies = []
        
        if len(self.metrics_history) < self.config.aggregation_window:
            return anomalies
            
        # Compute historical statistics
        recent = list(self.metrics_history)[-self.config.aggregation_window:]
        df = pd.DataFrame([r['metrics'] for r in recent])
        
        for metric, value in metrics.items():
            if metric in df.columns:
                mean = df[metric].mean()
                std = df[metric].std()
                
                # Check for anomaly
                if abs(value - mean) > 2 * std:  # 2 sigma threshold
                    anomalies.append({
                        'metric': metric,
                        'value': float(value),
                        'mean': float(mean),
                        'std': float(std),
                        'severity': float(abs(value - mean) / std)
                    })
                    
        return anomalies

class PerformanceMetrics:
    """Collects and analyzes performance metrics"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.timing_history = deque(maxlen=config.history_length)
        self.memory_history = deque(maxlen=config.history_length)
        self.error_history = deque(maxlen=config.history_length)
        
    def record_timing(self, duration: float, operation: str) -> None:
        """Record timing metric"""
        self.timing_history.append({
            'timestamp': time.time(),
            'duration': duration,
            'operation': operation
        })
        
        # Update Prometheus metrics
        PROCESSING_TIME.observe(duration)
        
    def record_memory(self, usage: int) -> None:
        """Record memory usage"""
        self.memory_history.append({
            'timestamp': time.time(),
            'usage': usage
        })
        
        # Update Prometheus metrics
        MEMORY_USAGE.set(usage)
        
    def record_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Record error occurrence"""
        self.error_history.append({
            'timestamp': time.time(),
            'error': str(error),
            'context': context
        })
        
        # Update Prometheus metrics
        ERROR_COUNT.inc()
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            'timing_metrics': self._compute_timing_metrics(),
            'memory_metrics': self._compute_memory_metrics(),
            'error_metrics': self._compute_error_metrics()
        }
        
    def _compute_timing_metrics(self) -> Dict[str, float]:
        """Compute timing metrics"""
        if not self.timing_history:
            return {}
            
        durations = [t['duration'] for t in self.timing_history]
        operations = pd.DataFrame(list(self.timing_history))
        
        return {
            'mean_duration': float(np.mean(durations)),
            'p95_duration': float(np.percentile(durations, 95)),
            'operation_stats': {
                op: {
                    'count': int(group['duration'].count()),
                    'mean_duration': float(group['duration'].mean())
                }
                for op, group in operations.groupby('operation')
            }
        }
        
    def _compute_memory_metrics(self) -> Dict[str, float]:
        """Compute memory metrics"""
        if not self.memory_history:
            return {}
            
        usages = [m['usage'] for m in self.memory_history]
        
        return {
            'current_usage': float(usages[-1]),
            'mean_usage': float(np.mean(usages)),
            'max_usage': float(np.max(usages)),
            'usage_trend': float(np.mean(np.diff(usages))) if len(usages) > 1 else 0.0
        }
        
    def _compute_error_metrics(self) -> Dict[str, Any]:
        """Compute error metrics"""
        if not self.error_history:
            return {}
            
        error_types = {}
        for error in self.error_history:
            error_type = error['error'].split(':')[0]
            if error_type not in error_types:
                error_types[error_type] = 0
            error_types[error_type] += 1
            
        return {
            'total_errors': len(self.error_history),
            'error_rate': float(len(self.error_history) / self.config.history_length),
            'error_types': error_types
        }

class FractiMetrics:
    """Main metrics system with fractal awareness"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.pattern_metrics = PatternMetrics(config)
        self.system_metrics = SystemMetrics(config)
        self.performance_metrics = PerformanceMetrics(config)
        
    def collect_metrics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        # Extract patterns
        patterns = self._extract_patterns(state)
        pattern_metrics = {
            pattern_id: self.pattern_metrics.record_pattern(pattern)
            for pattern_id, pattern in patterns.items()
        }
        
        # Collect system metrics
        system_metrics = {}
        for component, data in state.items():
            if isinstance(data, dict) and 'metrics' in data:
                metrics = self.system_metrics.record_metrics(
                    data['metrics'],
                    component
                )
                system_metrics[component] = metrics
                
        # Get performance metrics
        performance_metrics = self.performance_metrics.get_performance_metrics()
        
        return {
            'patterns': pattern_metrics,
            'system': system_metrics,
            'performance': performance_metrics,
            'analysis': self._analyze_metrics(
                pattern_metrics,
                system_metrics,
                performance_metrics
            )
        }
        
    def _extract_patterns(self, state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract patterns from system state"""
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
        
    def _analyze_metrics(self, pattern_metrics: Dict[str, Dict[str, float]],
                        system_metrics: Dict[str, Dict[str, Any]],
                        performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze collected metrics"""
        return {
            'system_health': self._compute_system_health(
                pattern_metrics,
                system_metrics,
                performance_metrics
            ),
            'recommendations': self._generate_recommendations(
                pattern_metrics,
                system_metrics,
                performance_metrics
            )
        }
        
    def _compute_system_health(self, 
                             pattern_metrics: Dict[str, Dict[str, float]],
                             system_metrics: Dict[str, Dict[str, Any]],
                             performance_metrics: Dict[str, Any]) -> float:
        """Compute overall system health score"""
        # Pattern health
        pattern_scores = [
            metrics['stability'] * (1 - metrics['complexity'])
            for metrics in pattern_metrics.values()
        ]
        pattern_health = np.mean(pattern_scores) if pattern_scores else 1.0
        
        # System health
        system_scores = [
            metrics['coherence'] * metrics['stability']
            for metrics in system_metrics.values()
        ]
        system_health = np.mean(system_scores) if system_scores else 1.0
        
        # Performance health
        timing_metrics = performance_metrics.get('timing_metrics', {})
        error_metrics = performance_metrics.get('error_metrics', {})
        
        performance_health = 1.0
        if timing_metrics and error_metrics:
            error_rate = error_metrics.get('error_rate', 0)
            performance_health = 1.0 - error_rate
            
        # Combine scores
        return float(np.mean([pattern_health, system_health, performance_health]))
        
    def _generate_recommendations(self,
                                pattern_metrics: Dict[str, Dict[str, float]],
                                system_metrics: Dict[str, Dict[str, Any]],
                                performance_metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate system recommendations"""
        recommendations = []
        
        # Pattern recommendations
        for pattern_id, metrics in pattern_metrics.items():
            if metrics['complexity'] > 0.8:
                recommendations.append({
                    'type': 'pattern',
                    'component': pattern_id,
                    'description': "High pattern complexity detected",
                    'suggestion': "Consider pattern simplification"
                })
                
        # System recommendations
        for component, metrics in system_metrics.items():
            if metrics['coherence'] < 0.7:
                recommendations.append({
                    'type': 'system',
                    'component': component,
                    'description': "Low system coherence detected",
                    'suggestion': "Review component interactions"
                })
                
        # Performance recommendations
        error_metrics = performance_metrics.get('error_metrics', {})
        if error_metrics.get('error_rate', 0) > 0.1:
            recommendations.append({
                'type': 'performance',
                'description': "High error rate detected",
                'suggestion': "Investigate error patterns"
            })
            
        return recommendations 