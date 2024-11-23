"""
monitoring.py

Monitoring system for FractiAI API to track performance, usage, and system health.
"""

from typing import Dict, List, Optional, Any
import time
import logging
from dataclasses import dataclass
from collections import deque
import psutil
import numpy as np
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('fractiai_requests_total', 'Total requests', ['endpoint'])
REQUEST_LATENCY = Histogram('fractiai_request_latency_seconds', 'Request latency')
ACTIVE_CONNECTIONS = Gauge('fractiai_active_connections', 'Active connections')
SYSTEM_MEMORY = Gauge('fractiai_memory_usage_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('fractiai_cpu_usage_percent', 'CPU usage percentage')

@dataclass
class MonitoringConfig:
    """Configuration for monitoring system"""
    history_size: int = 1000
    alert_threshold: float = 0.9
    sampling_interval: float = 1.0
    retention_days: int = 7

class MetricsCollector:
    """Collects and processes system metrics"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_history = deque(maxlen=config.history_size)
        self.alerts = deque(maxlen=config.history_size)
        self.start_time = time.time()
        
    def collect_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'network_connections': len(psutil.net_connections()),
            'timestamp': time.time()
        }
        
        self.metrics_history.append(metrics)
        self._check_alerts(metrics)
        
        return metrics
        
    def _check_alerts(self, metrics: Dict[str, float]) -> None:
        """Check metrics for alert conditions"""
        if metrics['cpu_percent'] > self.config.alert_threshold * 100:
            self._create_alert('High CPU usage', metrics['cpu_percent'])
            
        if metrics['memory_percent'] > self.config.alert_threshold * 100:
            self._create_alert('High memory usage', metrics['memory_percent'])
            
    def _create_alert(self, message: str, value: float) -> None:
        """Create new alert"""
        alert = {
            'message': message,
            'value': value,
            'timestamp': time.time()
        }
        self.alerts.append(alert)
        logger.warning(f"Alert: {message} ({value})")
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics"""
        if not self.metrics_history:
            return {}
            
        metrics_array = np.array([
            [m[key] for key in ['cpu_percent', 'memory_percent', 'disk_percent']]
            for m in self.metrics_history
        ])
        
        return {
            'current': self.metrics_history[-1],
            'averages': {
                'cpu': float(np.mean(metrics_array[:, 0])),
                'memory': float(np.mean(metrics_array[:, 1])),
                'disk': float(np.mean(metrics_array[:, 2]))
            },
            'peaks': {
                'cpu': float(np.max(metrics_array[:, 0])),
                'memory': float(np.max(metrics_array[:, 1])),
                'disk': float(np.max(metrics_array[:, 2]))
            }
        }

class RequestTracker:
    """Tracks API request metrics"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.requests: Dict[str, List[Dict]] = {}
        
    def track_request(self, endpoint: str, duration: float, 
                     status_code: int) -> None:
        """Track API request"""
        if endpoint not in self.requests:
            self.requests[endpoint] = deque(maxlen=self.config.history_size)
            
        request_data = {
            'duration': duration,
            'status_code': status_code,
            'timestamp': time.time()
        }
        
        self.requests[endpoint].append(request_data)
        
        # Update Prometheus metrics
        REQUEST_COUNT.labels(endpoint=endpoint).inc()
        REQUEST_LATENCY.observe(duration)
        
    def get_request_stats(self, endpoint: Optional[str] = None) -> Dict[str, Any]:
        """Get request statistics"""
        if endpoint:
            return self._compute_endpoint_stats(endpoint)
            
        return {
            endpoint: self._compute_endpoint_stats(endpoint)
            for endpoint in self.requests
        }
        
    def _compute_endpoint_stats(self, endpoint: str) -> Dict[str, Any]:
        """Compute statistics for endpoint"""
        if endpoint not in self.requests or not self.requests[endpoint]:
            return {}
            
        durations = [r['duration'] for r in self.requests[endpoint]]
        status_codes = [r['status_code'] for r in self.requests[endpoint]]
        
        return {
            'total_requests': len(self.requests[endpoint]),
            'average_duration': float(np.mean(durations)),
            'p95_duration': float(np.percentile(durations, 95)),
            'success_rate': float(sum(1 for c in status_codes if c < 400) / 
                                len(status_codes)),
            'status_codes': {
                str(code): status_codes.count(code)
                for code in set(status_codes)
            }
        }

class MonitoringSystem:
    """Main monitoring system for FractiAI API"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_collector = MetricsCollector(config)
        self.request_tracker = RequestTracker(config)
        
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect all system metrics"""
        system_metrics = self.metrics_collector.collect_metrics()
        
        # Update Prometheus metrics
        SYSTEM_MEMORY.set(psutil.virtual_memory().used)
        CPU_USAGE.set(system_metrics['cpu_percent'])
        ACTIVE_CONNECTIONS.set(system_metrics['network_connections'])
        
        return system_metrics
        
    def track_request(self, endpoint: str, duration: float, 
                     status_code: int) -> None:
        """Track API request"""
        self.request_tracker.track_request(endpoint, duration, status_code)
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        return {
            'system_metrics': self.metrics_collector.get_metrics_summary(),
            'request_stats': self.request_tracker.get_request_stats(),
            'alerts': list(self.metrics_collector.alerts),
            'uptime': time.time() - self.metrics_collector.start_time
        }

# Initialize monitoring system
monitoring = MonitoringSystem(MonitoringConfig()) 