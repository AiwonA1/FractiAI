"""
Monitoring module for FractiAI API

Implements comprehensive system monitoring, metrics collection,
and analysis capabilities.
"""

from typing import Dict, List, Optional, Any, Set
import asyncio
import time
import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import psutil
from collections import defaultdict

logger = logging.getLogger(__name__)

class MetricType(str, Enum):
    """Types of metrics"""
    SYSTEM = "system"
    DOMAIN = "domain"
    QUANTUM = "quantum"
    PERFORMANCE = "performance"
    RESOURCE = "resource"

@dataclass
class MetricValue:
    """Metric value with metadata"""
    value: float
    timestamp: datetime
    confidence: Optional[float] = None
    source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'source': self.source
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricValue':
        """Create from dictionary"""
        return cls(
            value=data['value'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            confidence=data.get('confidence'),
            source=data.get('source')
        )

class MetricsCollector:
    """Collects and manages system metrics"""
    
    def __init__(
        self,
        collection_interval: float = 1.0,
        history_window: timedelta = timedelta(hours=1)
    ):
        self._collection_interval = collection_interval
        self._history_window = history_window
        self._metrics: Dict[str, Dict[str, List[MetricValue]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._running = False
        self._collectors: Dict[str, callable] = {}
        self._subscribers: Dict[str, Set[callable]] = defaultdict(set)
        
        # Register default collectors
        self._register_default_collectors()
        
    def _register_default_collectors(self) -> None:
        """Register default metric collectors"""
        # System metrics
        self.register_collector(
            'cpu_usage',
            MetricType.SYSTEM,
            self._collect_cpu_usage
        )
        self.register_collector(
            'memory_usage',
            MetricType.SYSTEM,
            self._collect_memory_usage
        )
        self.register_collector(
            'disk_usage',
            MetricType.SYSTEM,
            self._collect_disk_usage
        )
        
        # Performance metrics
        self.register_collector(
            'request_latency',
            MetricType.PERFORMANCE,
            self._collect_request_latency
        )
        self.register_collector(
            'error_rate',
            MetricType.PERFORMANCE,
            self._collect_error_rate
        )
        
    async def start(self) -> None:
        """Start metrics collection"""
        self._running = True
        asyncio.create_task(self._collection_loop())
        
    async def stop(self) -> None:
        """Stop metrics collection"""
        self._running = False
        
    def register_collector(
        self,
        name: str,
        metric_type: MetricType,
        collector: callable
    ) -> None:
        """Register metric collector"""
        self._collectors[f"{metric_type.value}:{name}"] = collector
        
    def subscribe(
        self,
        metric_type: MetricType,
        name: str,
        callback: callable
    ) -> None:
        """Subscribe to metric updates"""
        self._subscribers[f"{metric_type.value}:{name}"].add(callback)
        
    def unsubscribe(
        self,
        metric_type: MetricType,
        name: str,
        callback: callable
    ) -> None:
        """Unsubscribe from metric updates"""
        self._subscribers[f"{metric_type.value}:{name}"].discard(callback)
        
    async def _collection_loop(self) -> None:
        """Main collection loop"""
        while self._running:
            try:
                # Collect metrics from all collectors
                for metric_id, collector in self._collectors.items():
                    try:
                        value = await collector()
                        await self._store_metric(metric_id, value)
                    except Exception as e:
                        logger.error(f"Error collecting metric {metric_id}: {str(e)}")
                        
                # Clean up old metrics
                self._cleanup_metrics()
                
                await asyncio.sleep(self._collection_interval)
            except Exception as e:
                logger.error(f"Error in collection loop: {str(e)}")
                await asyncio.sleep(1)
                
    async def _store_metric(
        self,
        metric_id: str,
        value: float,
        confidence: Optional[float] = None,
        source: Optional[str] = None
    ) -> None:
        """Store metric value"""
        metric = MetricValue(
            value=value,
            timestamp=datetime.utcnow(),
            confidence=confidence,
            source=source
        )
        
        # Store metric
        self._metrics[metric_id].append(metric)
        
        # Notify subscribers
        for callback in self._subscribers[metric_id]:
            try:
                await callback(metric)
            except Exception as e:
                logger.error(f"Error in metric callback: {str(e)}")
                
    def _cleanup_metrics(self) -> None:
        """Clean up old metrics"""
        cutoff = datetime.utcnow() - self._history_window
        
        for metric_id in self._metrics:
            self._metrics[metric_id] = [
                m for m in self._metrics[metric_id]
                if m.timestamp > cutoff
            ]
            
    async def get_metrics(
        self,
        metric_type: Optional[MetricType] = None,
        name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get metrics within time range"""
        result = {}
        
        # Filter metrics
        for metric_id, values in self._metrics.items():
            type_str, metric_name = metric_id.split(':', 1)
            
            # Apply filters
            if metric_type and type_str != metric_type.value:
                continue
            if name and metric_name != name:
                continue
                
            # Filter by time range
            filtered_values = [
                v.to_dict() for v in values
                if (not start_time or v.timestamp >= start_time) and
                (not end_time or v.timestamp <= end_time)
            ]
            
            if filtered_values:
                result[metric_id] = filtered_values
                
        return result
        
    async def get_statistics(
        self,
        metric_type: MetricType,
        name: str,
        window: timedelta = timedelta(minutes=5)
    ) -> Dict[str, float]:
        """Get statistics for metric"""
        metric_id = f"{metric_type.value}:{name}"
        if metric_id not in self._metrics:
            return {}
            
        # Get recent values
        cutoff = datetime.utcnow() - window
        values = [
            m.value for m in self._metrics[metric_id]
            if m.timestamp > cutoff
        ]
        
        if not values:
            return {}
            
        # Compute statistics
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'count': len(values)
        }
        
    async def check_anomalies(
        self,
        metric_type: MetricType,
        name: str,
        window: timedelta = timedelta(minutes=5),
        threshold: float = 2.0
    ) -> List[Dict[str, Any]]:
        """Check for anomalies in metric"""
        metric_id = f"{metric_type.value}:{name}"
        if metric_id not in self._metrics:
            return []
            
        # Get recent values
        cutoff = datetime.utcnow() - window
        recent_metrics = [
            m for m in self._metrics[metric_id]
            if m.timestamp > cutoff
        ]
        
        if not recent_metrics:
            return []
            
        # Compute statistics
        values = [m.value for m in recent_metrics]
        mean = np.mean(values)
        std = np.std(values)
        
        # Find anomalies
        anomalies = []
        for metric in recent_metrics:
            z_score = abs(metric.value - mean) / std if std > 0 else 0
            if z_score > threshold:
                anomalies.append({
                    'timestamp': metric.timestamp.isoformat(),
                    'value': metric.value,
                    'z_score': z_score
                })
                
        return anomalies
        
    # Default collectors
    async def _collect_cpu_usage(self) -> float:
        """Collect CPU usage"""
        return psutil.cpu_percent()
        
    async def _collect_memory_usage(self) -> float:
        """Collect memory usage"""
        return psutil.virtual_memory().percent
        
    async def _collect_disk_usage(self) -> float:
        """Collect disk usage"""
        return psutil.disk_usage('/').percent
        
    async def _collect_request_latency(self) -> float:
        """Collect request latency"""
        # TODO: Implement request latency collection
        return 0.0
        
    async def _collect_error_rate(self) -> float:
        """Collect error rate"""
        # TODO: Implement error rate collection
        return 0.0

# Initialize metrics collector
metrics_collector = MetricsCollector() 