"""
FractiProfile.py

Implements comprehensive profiling framework for FractiAI, enabling performance
analysis across dimensions, scales, and patterns through fractal-based profiling.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging
import time
import cProfile
import pstats
from memory_profiler import profile as memory_profile
import json
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class ProfileConfig:
    """Configuration for profiling framework"""
    memory_tracking: bool = True
    time_tracking: bool = True
    pattern_tracking: bool = True
    profile_depth: int = 3
    sample_interval: float = 0.1
    history_size: int = 1000

class ComponentProfile:
    """Profiles individual system component"""
    
    def __init__(self, component_id: str, config: ProfileConfig):
        self.component_id = component_id
        self.config = config
        self.time_samples = deque(maxlen=config.history_size)
        self.memory_samples = deque(maxlen=config.history_size)
        self.pattern_samples = deque(maxlen=config.history_size)
        self.call_stats = {}
        
    def start_operation(self, operation: str) -> float:
        """Start timing an operation"""
        timestamp = time.time()
        if operation not in self.call_stats:
            self.call_stats[operation] = {
                'calls': 0,
                'total_time': 0.0,
                'memory_usage': []
            }
        self.call_stats[operation]['calls'] += 1
        return timestamp
        
    def end_operation(self, operation: str, start_time: float) -> None:
        """End timing an operation"""
        duration = time.time() - start_time
        self.call_stats[operation]['total_time'] += duration
        
        if self.config.memory_tracking:
            memory_usage = self._measure_memory()
            self.call_stats[operation]['memory_usage'].append(memory_usage)
            
    def sample_performance(self) -> Dict[str, float]:
        """Take performance sample"""
        sample = {}
        
        if self.config.time_tracking:
            sample['time'] = time.time()
            
        if self.config.memory_tracking:
            sample['memory'] = self._measure_memory()
            
        if self.config.pattern_tracking:
            sample['pattern'] = self._measure_pattern_complexity()
            
        return sample
        
    def _measure_memory(self) -> float:
        """Measure current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            return float(process.memory_info().rss / 1024 / 1024)  # MB
        except:
            return 0.0
            
    def _measure_pattern_complexity(self) -> float:
        """Measure complexity of current patterns"""
        if not self.pattern_samples:
            return 0.0
            
        patterns = np.array(list(self.pattern_samples))
        return float(np.std(patterns))
        
    def get_profile_metrics(self) -> Dict[str, Any]:
        """Get comprehensive profile metrics"""
        return {
            'operation_stats': self._compute_operation_stats(),
            'performance_metrics': self._compute_performance_metrics(),
            'pattern_metrics': self._compute_pattern_metrics()
        }
        
    def _compute_operation_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute statistics for operations"""
        stats = {}
        for operation, data in self.call_stats.items():
            stats[operation] = {
                'call_count': data['calls'],
                'total_time': data['total_time'],
                'avg_time': data['total_time'] / data['calls'] if data['calls'] > 0 else 0,
                'avg_memory': np.mean(data['memory_usage']) if data['memory_usage'] else 0
            }
        return stats
        
    def _compute_performance_metrics(self) -> Dict[str, float]:
        """Compute overall performance metrics"""
        if not self.time_samples or not self.memory_samples:
            return {}
            
        time_series = np.array([s['time'] for s in self.time_samples])
        memory_series = np.array([s['memory'] for s in self.memory_samples])
        
        return {
            'time_efficiency': float(1.0 - np.std(np.diff(time_series)) / 
                                   (np.mean(np.diff(time_series)) + 1e-7)),
            'memory_stability': float(1.0 - np.std(memory_series) / 
                                    (np.mean(memory_series) + 1e-7)),
            'performance_trend': float(np.mean(np.diff(time_series)))
        }
        
    def _compute_pattern_metrics(self) -> Dict[str, float]:
        """Compute pattern-related metrics"""
        if not self.pattern_samples:
            return {}
            
        patterns = np.array([s['pattern'] for s in self.pattern_samples 
                           if 'pattern' in s])
        
        return {
            'pattern_complexity': float(np.mean(patterns)),
            'pattern_stability': float(1.0 - np.std(patterns) / 
                                    (np.mean(patterns) + 1e-7)),
            'pattern_trend': float(np.mean(np.diff(patterns))) if len(patterns) > 1 else 0.0
        }

class FractiProfile:
    """Main profiling system with fractal awareness"""
    
    def __init__(self, config: ProfileConfig):
        self.config = config
        self.component_profiles: Dict[str, ComponentProfile] = {}
        self.system_samples = deque(maxlen=config.history_size)
        self.profiler = cProfile.Profile()
        
    def add_component(self, component_id: str) -> None:
        """Add component to profiling system"""
        self.component_profiles[component_id] = ComponentProfile(
            component_id, 
            self.config
        )
        
    def start_profiling(self) -> None:
        """Start system profiling"""
        self.profiler.enable()
        
    def stop_profiling(self) -> None:
        """Stop system profiling"""
        self.profiler.disable()
        
    def profile_operation(self, component_id: str, 
                         operation: str) -> Callable:
        """Decorator for profiling operations"""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                if component_id not in self.component_profiles:
                    self.add_component(component_id)
                    
                profile = self.component_profiles[component_id]
                start_time = profile.start_operation(operation)
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    profile.end_operation(operation, start_time)
                    
            return wrapper
        return decorator
        
    def sample_system(self) -> None:
        """Take system-wide performance sample"""
        sample = {
            'timestamp': time.time(),
            'component_samples': {}
        }
        
        for component_id, profile in self.component_profiles.items():
            sample['component_samples'][component_id] = profile.sample_performance()
            
        self.system_samples.append(sample)
        
    def get_profile_report(self) -> Dict[str, Any]:
        """Generate comprehensive profile report"""
        # Get profiler statistics
        stats = pstats.Stats(self.profiler)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        
        return {
            'system_profile': self._analyze_system_profile(),
            'component_profiles': {
                component_id: profile.get_profile_metrics()
                for component_id, profile in self.component_profiles.items()
            },
            'profiler_stats': self._format_profiler_stats(stats),
            'recommendations': self._generate_recommendations()
        }
        
    def _analyze_system_profile(self) -> Dict[str, Any]:
        """Analyze system-wide profile"""
        if not self.system_samples:
            return {}
            
        return {
            'performance_analysis': self._analyze_performance(),
            'memory_analysis': self._analyze_memory(),
            'pattern_analysis': self._analyze_patterns()
        }
        
    def _analyze_performance(self) -> Dict[str, float]:
        """Analyze system performance"""
        if not self.system_samples:
            return {}
            
        timestamps = [s['timestamp'] for s in self.system_samples]
        intervals = np.diff(timestamps)
        
        return {
            'avg_interval': float(np.mean(intervals)),
            'interval_stability': float(1.0 - np.std(intervals) / 
                                     (np.mean(intervals) + 1e-7)),
            'performance_trend': float(np.mean(np.diff(intervals))) 
                               if len(intervals) > 1 else 0.0
        }
        
    def _analyze_memory(self) -> Dict[str, float]:
        """Analyze memory usage"""
        memory_samples = []
        for sample in self.system_samples:
            for component_samples in sample['component_samples'].values():
                if 'memory' in component_samples:
                    memory_samples.append(component_samples['memory'])
                    
        if not memory_samples:
            return {}
            
        return {
            'avg_memory': float(np.mean(memory_samples)),
            'memory_stability': float(1.0 - np.std(memory_samples) / 
                                    (np.mean(memory_samples) + 1e-7)),
            'memory_trend': float(np.mean(np.diff(memory_samples))) 
                          if len(memory_samples) > 1 else 0.0
        }
        
    def _analyze_patterns(self) -> Dict[str, float]:
        """Analyze pattern behavior"""
        pattern_samples = []
        for sample in self.system_samples:
            for component_samples in sample['component_samples'].values():
                if 'pattern' in component_samples:
                    pattern_samples.append(component_samples['pattern'])
                    
        if not pattern_samples:
            return {}
            
        return {
            'pattern_complexity': float(np.mean(pattern_samples)),
            'pattern_stability': float(1.0 - np.std(pattern_samples) / 
                                    (np.mean(pattern_samples) + 1e-7)),
            'pattern_trend': float(np.mean(np.diff(pattern_samples))) 
                           if len(pattern_samples) > 1 else 0.0
        }
        
    def _format_profiler_stats(self, stats: pstats.Stats) -> Dict[str, Any]:
        """Format profiler statistics"""
        # Capture stats output
        import io
        output = io.StringIO()
        stats.stream = output
        stats.print_stats()
        
        # Parse and format stats
        lines = output.getvalue().split('\n')
        formatted_stats = {
            'total_calls': int(lines[0].split()[0]),
            'primitive_calls': int(lines[0].split()[1]),
            'total_time': float(lines[0].split()[2]),
            'function_stats': []
        }
        
        # Parse function statistics
        for line in lines[5:]:  # Skip header lines
            if line.strip():
                parts = line.split()
                if len(parts) >= 6:
                    formatted_stats['function_stats'].append({
                        'ncalls': parts[0],
                        'tottime': float(parts[1]),
                        'percall': float(parts[2]),
                        'cumtime': float(parts[3]),
                        'percall_cum': float(parts[4]),
                        'filename_function': ' '.join(parts[5:])
                    })
                    
        return formatted_stats
        
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Analyze performance
        perf_analysis = self._analyze_performance()
        if perf_analysis.get('interval_stability', 1.0) < 0.8:
            recommendations.append({
                'type': 'performance',
                'component': 'system',
                'description': "Low performance stability detected",
                'suggestion': "Review component synchronization and load balancing"
            })
            
        # Analyze memory
        memory_analysis = self._analyze_memory()
        if memory_analysis.get('memory_trend', 0) > 0.1:
            recommendations.append({
                'type': 'memory',
                'component': 'system',
                'description': "Increasing memory trend detected",
                'suggestion': "Review memory management and potential leaks"
            })
            
        # Analyze patterns
        pattern_analysis = self._analyze_patterns()
        if pattern_analysis.get('pattern_stability', 1.0) < 0.7:
            recommendations.append({
                'type': 'pattern',
                'component': 'system',
                'description': "Unstable pattern behavior detected",
                'suggestion': "Review pattern processing and transformation logic"
            })
            
        # Component-specific recommendations
        for component_id, profile in self.component_profiles.items():
            metrics = profile.get_profile_metrics()
            
            # Check operation performance
            for operation, stats in metrics['operation_stats'].items():
                if stats['avg_time'] > 1.0:  # High average time
                    recommendations.append({
                        'type': 'operation',
                        'component': component_id,
                        'description': f"High execution time for operation {operation}",
                        'suggestion': "Consider optimization or parallelization"
                    })
                    
        return recommendations
        
    def save_profile(self, filename: Optional[str] = None) -> None:
        """Save profile data to file"""
        if filename is None:
            filename = f"fracti_profile_{int(time.time())}.json"
            
        profile_data = {
            'config': self.config.__dict__,
            'profile_report': self.get_profile_report()
        }
        
        with open(filename, 'w') as f:
            json.dump(profile_data, f, indent=2) 