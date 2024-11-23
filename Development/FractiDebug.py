"""
FractiDebug.py

Implements comprehensive debugging framework for FractiAI, enabling pattern-aware
debugging and analysis across dimensions and scales.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging
import time
from collections import deque
import json
import traceback

logger = logging.getLogger(__name__)

@dataclass
class DebugConfig:
    """Configuration for debugging framework"""
    trace_depth: int = 5
    pattern_threshold: float = 0.7
    memory_size: int = 1000
    log_level: str = "DEBUG"
    save_traces: bool = True
    visualization: bool = True

class PatternTrace:
    """Traces pattern evolution through system"""
    
    def __init__(self, trace_id: str, pattern: np.ndarray):
        self.trace_id = trace_id
        self.initial_pattern = pattern
        self.transformations = []
        self.errors = []
        self.start_time = time.time()
        
    def add_transformation(self, component: str, 
                         transformed: np.ndarray) -> None:
        """Add pattern transformation"""
        self.transformations.append({
            'component': component,
            'pattern': transformed,
            'timestamp': time.time(),
            'duration': time.time() - self.start_time
        })
        
    def add_error(self, error: Exception, 
                 context: Dict[str, Any]) -> None:
        """Add error information"""
        self.errors.append({
            'error': str(error),
            'traceback': traceback.format_exc(),
            'context': context,
            'timestamp': time.time()
        })
        
    def get_metrics(self) -> Dict[str, float]:
        """Get trace metrics"""
        return {
            'num_transformations': len(self.transformations),
            'num_errors': len(self.errors),
            'total_duration': time.time() - self.start_time,
            'pattern_stability': self._compute_stability()
        }
        
    def _compute_stability(self) -> float:
        """Compute pattern stability through transformations"""
        if len(self.transformations) < 2:
            return 1.0
            
        patterns = [t['pattern'] for t in self.transformations]
        diffs = np.diff(patterns, axis=0)
        return float(1.0 - np.mean(np.abs(diffs)))

class ComponentDebugger:
    """Debugger for individual system components"""
    
    def __init__(self, component_id: str):
        self.component_id = component_id
        self.call_history = deque(maxlen=1000)
        self.error_history = deque(maxlen=1000)
        self.performance_metrics = {}
        
    def log_call(self, method: str, args: Tuple, 
                 kwargs: Dict, result: Any) -> None:
        """Log method call"""
        call_info = {
            'method': method,
            'args': args,
            'kwargs': kwargs,
            'result': result,
            'timestamp': time.time()
        }
        self.call_history.append(call_info)
        
    def log_error(self, error: Exception, 
                 context: Dict[str, Any]) -> None:
        """Log error information"""
        error_info = {
            'error': str(error),
            'traceback': traceback.format_exc(),
            'context': context,
            'timestamp': time.time()
        }
        self.error_history.append(error_info)
        
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update performance metrics"""
        self.performance_metrics.update(metrics)
        
    def get_debug_info(self) -> Dict[str, Any]:
        """Get comprehensive debug information"""
        return {
            'component_id': self.component_id,
            'call_history': list(self.call_history),
            'error_history': list(self.error_history),
            'performance_metrics': self.performance_metrics,
            'analysis': self._analyze_behavior()
        }
        
    def _analyze_behavior(self) -> Dict[str, Any]:
        """Analyze component behavior"""
        if not self.call_history:
            return {}
            
        return {
            'call_patterns': self._analyze_call_patterns(),
            'error_patterns': self._analyze_error_patterns(),
            'performance_trends': self._analyze_performance()
        }
        
    def _analyze_call_patterns(self) -> Dict[str, Any]:
        """Analyze method call patterns"""
        method_calls = {}
        for call in self.call_history:
            method = call['method']
            if method not in method_calls:
                method_calls[method] = []
            method_calls[method].append(call)
            
        return {
            method: {
                'count': len(calls),
                'avg_duration': np.mean([
                    calls[-1]['timestamp'] - calls[0]['timestamp']
                    for calls in [calls[i:i+2] 
                                for i in range(len(calls)-1)]
                ]) if len(calls) > 1 else 0
            }
            for method, calls in method_calls.items()
        }
        
    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns"""
        error_types = {}
        for error in self.error_history:
            error_type = error['error'].split(':')[0]
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(error)
            
        return {
            error_type: {
                'count': len(errors),
                'contexts': list(set([
                    str(e['context']) for e in errors
                ]))
            }
            for error_type, errors in error_types.items()
        }
        
    def _analyze_performance(self) -> Dict[str, float]:
        """Analyze performance trends"""
        if not self.performance_metrics:
            return {}
            
        metrics_history = [m for m in self.performance_metrics.values() 
                         if isinstance(m, (int, float))]
        
        if not metrics_history:
            return {}
            
        return {
            'mean': float(np.mean(metrics_history)),
            'std': float(np.std(metrics_history)),
            'trend': float(np.mean(np.diff(metrics_history))) 
                    if len(metrics_history) > 1 else 0.0
        }

class FractiDebug:
    """Main debugging system with pattern awareness"""
    
    def __init__(self, config: DebugConfig):
        self.config = config
        self.pattern_traces: Dict[str, PatternTrace] = {}
        self.component_debuggers: Dict[str, ComponentDebugger] = {}
        self.system_metrics = {}
        self.debug_history = deque(maxlen=config.memory_size)
        
        # Configure logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        
    def trace_pattern(self, pattern: np.ndarray) -> str:
        """Start tracing a pattern"""
        trace_id = f"TRACE_{len(self.pattern_traces)}"
        self.pattern_traces[trace_id] = PatternTrace(trace_id, pattern)
        return trace_id
        
    def add_component(self, component_id: str) -> None:
        """Add component to debug system"""
        self.component_debuggers[component_id] = ComponentDebugger(component_id)
        
    def log_transformation(self, trace_id: str, 
                         component: str, 
                         transformed: np.ndarray) -> None:
        """Log pattern transformation"""
        if trace_id in self.pattern_traces:
            self.pattern_traces[trace_id].add_transformation(
                component, transformed
            )
            
    def log_error(self, component_id: str, 
                 error: Exception, 
                 context: Dict[str, Any]) -> None:
        """Log component error"""
        if component_id in self.component_debuggers:
            self.component_debuggers[component_id].log_error(error, context)
            
        # Store in debug history
        debug_info = {
            'timestamp': time.time(),
            'type': 'error',
            'component': component_id,
            'error': str(error),
            'context': context
        }
        self.debug_history.append(debug_info)
        
    def log_call(self, component_id: str, 
                 method: str, 
                 args: Tuple, 
                 kwargs: Dict, 
                 result: Any) -> None:
        """Log component method call"""
        if component_id in self.component_debuggers:
            self.component_debuggers[component_id].log_call(
                method, args, kwargs, result
            )
            
    def update_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """Update system metrics"""
        self.system_metrics.update(metrics)
        
        # Update component metrics
        for component_id, component_metrics in metrics.items():
            if component_id in self.component_debuggers:
                self.component_debuggers[component_id].update_metrics(
                    component_metrics
                )
                
    def get_debug_report(self) -> Dict[str, Any]:
        """Generate comprehensive debug report"""
        return {
            'pattern_traces': {
                trace_id: {
                    'metrics': trace.get_metrics(),
                    'transformations': trace.transformations,
                    'errors': trace.errors
                }
                for trace_id, trace in self.pattern_traces.items()
            },
            'component_analysis': {
                component_id: debugger.get_debug_info()
                for component_id, debugger in self.component_debuggers.items()
            },
            'system_analysis': self._analyze_system(),
            'recommendations': self._generate_recommendations()
        }
        
    def _analyze_system(self) -> Dict[str, Any]:
        """Analyze system-wide behavior"""
        return {
            'error_analysis': self._analyze_errors(),
            'performance_analysis': self._analyze_performance(),
            'pattern_analysis': self._analyze_patterns()
        }
        
    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze system-wide errors"""
        component_errors = {}
        for component_id, debugger in self.component_debuggers.items():
            error_analysis = debugger._analyze_error_patterns()
            if error_analysis:
                component_errors[component_id] = error_analysis
                
        return {
            'component_errors': component_errors,
            'error_correlations': self._analyze_error_correlations(),
            'error_trends': self._analyze_error_trends()
        }
        
    def _analyze_performance(self) -> Dict[str, float]:
        """Analyze system-wide performance"""
        component_metrics = []
        for debugger in self.component_debuggers.values():
            metrics = debugger._analyze_performance()
            if metrics:
                component_metrics.append(metrics)
                
        if not component_metrics:
            return {}
            
        return {
            'mean_performance': float(np.mean([m['mean'] for m in component_metrics])),
            'performance_stability': float(1.0 - np.mean([m['std'] for m in component_metrics])),
            'performance_trend': float(np.mean([m['trend'] for m in component_metrics]))
        }
        
    def _analyze_patterns(self) -> Dict[str, Any]:
        """Analyze pattern behavior"""
        pattern_metrics = []
        for trace in self.pattern_traces.values():
            metrics = trace.get_metrics()
            pattern_metrics.append(metrics)
            
        if not pattern_metrics:
            return {}
            
        return {
            'pattern_stability': float(np.mean([m['pattern_stability'] 
                                              for m in pattern_metrics])),
            'transformation_depth': float(np.mean([m['num_transformations'] 
                                                 for m in pattern_metrics])),
            'error_rate': float(np.mean([m['num_errors'] 
                                       for m in pattern_metrics]))
        }
        
    def _analyze_error_correlations(self) -> Dict[str, float]:
        """Analyze correlations between errors"""
        error_times = {}
        for component_id, debugger in self.component_debuggers.items():
            error_times[component_id] = [
                e['timestamp'] for e in debugger.error_history
            ]
            
        correlations = {}
        for comp1 in error_times:
            for comp2 in error_times:
                if comp1 < comp2:
                    times1 = np.array(error_times[comp1])
                    times2 = np.array(error_times[comp2])
                    if len(times1) > 0 and len(times2) > 0:
                        correlation = np.corrcoef(times1, times2)[0,1]
                        correlations[f"{comp1}-{comp2}"] = float(correlation)
                        
        return correlations
        
    def _analyze_error_trends(self) -> Dict[str, float]:
        """Analyze error occurrence trends"""
        error_counts = {}
        for debugger in self.component_debuggers.values():
            for error in debugger.error_history:
                timestamp = error['timestamp']
                error_counts[timestamp] = error_counts.get(timestamp, 0) + 1
                
        if not error_counts:
            return {}
            
        times = sorted(error_counts.keys())
        counts = [error_counts[t] for t in times]
        
        return {
            'error_rate': float(np.mean(counts)),
            'error_trend': float(np.mean(np.diff(counts))) if len(counts) > 1 else 0.0,
            'error_stability': float(1.0 - np.std(counts) / (np.mean(counts) + 1e-7))
        }
        
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate debugging recommendations"""
        recommendations = []
        
        # Analyze error patterns
        error_analysis = self._analyze_errors()
        if error_analysis['component_errors']:
            for component_id, errors in error_analysis['component_errors'].items():
                if len(errors) > self.config.pattern_threshold:
                    recommendations.append({
                        'type': 'error_pattern',
                        'component': component_id,
                        'description': f"Recurring error pattern detected in {component_id}",
                        'suggestion': "Investigate error correlation patterns"
                    })
                    
        # Analyze performance
        performance = self._analyze_performance()
        if performance.get('performance_trend', 0) < 0:
            recommendations.append({
                'type': 'performance',
                'description': "Declining performance trend detected",
                'suggestion': "Review component optimization opportunities"
            })
            
        # Analyze patterns
        pattern_analysis = self._analyze_patterns()
        if pattern_analysis.get('pattern_stability', 1.0) < self.config.pattern_threshold:
            recommendations.append({
                'type': 'pattern',
                'description': "Low pattern stability detected",
                'suggestion': "Review pattern transformation logic"
            })
            
        return recommendations
        
    def save_debug_info(self, filename: Optional[str] = None) -> None:
        """Save debug information to file"""
        if filename is None:
            filename = f"fracti_debug_{int(time.time())}.json"
            
        debug_info = {
            'config': self.config.__dict__,
            'system_analysis': self._analyze_system(),
            'recommendations': self._generate_recommendations()
        }
        
        with open(filename, 'w') as f:
            json.dump(debug_info, f, indent=2) 