"""
FractiAnalytics.py

Implements comprehensive analytics framework for FractiAI, enabling pattern analysis,
performance insights, and system optimization through fractal-based analytics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import time
from collections import deque
import pandas as pd
from scipy.stats import entropy
from scipy.signal import find_peaks
import networkx as nx

logger = logging.getLogger(__name__)

@dataclass
class AnalyticsConfig:
    """Configuration for analytics framework"""
    pattern_threshold: float = 0.7
    history_length: int = 1000
    analysis_interval: int = 100
    fractal_depth: int = 3
    significance_level: float = 0.05

class PatternAnalyzer:
    """Analyzes fractal patterns in system behavior"""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.pattern_history = deque(maxlen=config.history_length)
        self.discovered_patterns = {}
        
    def analyze_pattern(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze pattern in data"""
        # Store pattern
        self.pattern_history.append(data)
        
        # Analyze at different scales
        scales = self._analyze_scales(data)
        
        # Find recurring patterns
        patterns = self._find_patterns(data)
        
        # Compute pattern metrics
        metrics = self._compute_pattern_metrics(data)
        
        return {
            'scales': scales,
            'patterns': patterns,
            'metrics': metrics
        }
        
    def _analyze_scales(self, data: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Analyze data at different scales"""
        scales = {}
        
        for i in range(self.config.fractal_depth):
            scale = 2 ** i
            scaled_data = self._rescale_data(data, scale)
            
            scales[scale] = {
                'complexity': self._compute_complexity(scaled_data),
                'stability': self._compute_stability(scaled_data),
                'correlation': self._compute_correlation(data, scaled_data)
            }
            
        return scales
        
    def _find_patterns(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Find recurring patterns in data"""
        patterns = []
        
        # Use sliding windows of different sizes
        for size in [4, 8, 16]:
            if len(data) >= size:
                windows = self._get_windows(data, size)
                similar = self._find_similar_windows(windows)
                patterns.extend(similar)
                
        return patterns
        
    def _compute_pattern_metrics(self, data: np.ndarray) -> Dict[str, float]:
        """Compute pattern-related metrics"""
        return {
            'entropy': float(entropy(data / np.sum(data))),
            'peak_ratio': self._compute_peak_ratio(data),
            'fractal_dimension': self._compute_fractal_dimension(data)
        }
        
    def _rescale_data(self, data: np.ndarray, scale: int) -> np.ndarray:
        """Rescale data to given scale"""
        if len(data) < scale:
            return data
            
        return np.array([
            np.mean(chunk) 
            for chunk in np.array_split(data, scale)
        ])
        
    def _compute_complexity(self, data: np.ndarray) -> float:
        """Compute complexity measure"""
        return float(entropy(data / np.sum(data)))
        
    def _compute_stability(self, data: np.ndarray) -> float:
        """Compute stability measure"""
        if len(data) < 2:
            return 1.0
        return float(1.0 - np.std(data) / (np.mean(np.abs(data)) + 1e-7))
        
    def _compute_correlation(self, original: np.ndarray, 
                           scaled: np.ndarray) -> float:
        """Compute correlation between original and scaled data"""
        if len(scaled) < 2:
            return 1.0
            
        # Interpolate scaled data to match original size
        scaled_interp = np.interp(
            np.linspace(0, 1, len(original)),
            np.linspace(0, 1, len(scaled)),
            scaled
        )
        
        return float(np.corrcoef(original, scaled_interp)[0,1])
        
    def _get_windows(self, data: np.ndarray, size: int) -> np.ndarray:
        """Get sliding windows from data"""
        return np.array([
            data[i:i+size] 
            for i in range(len(data)-size+1)
        ])
        
    def _find_similar_windows(self, windows: np.ndarray) -> List[Dict[str, Any]]:
        """Find similar windows using correlation"""
        similar = []
        
        for i in range(len(windows)):
            for j in range(i+1, len(windows)):
                corr = np.corrcoef(windows[i], windows[j])[0,1]
                if abs(corr) > self.config.pattern_threshold:
                    similar.append({
                        'pattern': windows[i].tolist(),
                        'similarity': float(corr),
                        'position': (i, j)
                    })
                    
        return similar
        
    def _compute_peak_ratio(self, data: np.ndarray) -> float:
        """Compute ratio of peaks to data length"""
        peaks, _ = find_peaks(data)
        return float(len(peaks) / len(data))
        
    def _compute_fractal_dimension(self, data: np.ndarray) -> float:
        """Compute fractal dimension using box-counting"""
        scales = np.logspace(0, 2, num=10, base=2, dtype=int)
        counts = []
        
        for scale in scales:
            scaled = self._rescale_data(data, scale)
            counts.append(len(np.unique(scaled)))
            
        if len(counts) > 1:
            coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
            return float(-coeffs[0])
        return 1.0

class PerformanceAnalyzer:
    """Analyzes system performance metrics"""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.metrics_history = deque(maxlen=config.history_length)
        
    def analyze_performance(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze performance metrics"""
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Analyze trends
        trends = self._analyze_trends(metrics)
        
        # Compute statistics
        stats = self._compute_statistics()
        
        # Generate insights
        insights = self._generate_insights(trends, stats)
        
        return {
            'trends': trends,
            'statistics': stats,
            'insights': insights
        }
        
    def _analyze_trends(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Analyze metric trends"""
        if len(self.metrics_history) < 2:
            return {}
            
        trends = {}
        for key in metrics:
            history = [m[key] for m in self.metrics_history if key in m]
            if len(history) > 1:
                trends[key] = {
                    'slope': float(np.mean(np.diff(history))),
                    'stability': float(1.0 - np.std(history) / (np.mean(np.abs(history)) + 1e-7)),
                    'acceleration': float(np.mean(np.diff(np.diff(history)))) 
                              if len(history) > 2 else 0.0
                }
                
        return trends
        
    def _compute_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute statistical measures"""
        if not self.metrics_history:
            return {}
            
        df = pd.DataFrame(self.metrics_history)
        
        return {
            column: {
                'mean': float(df[column].mean()),
                'std': float(df[column].std()),
                'min': float(df[column].min()),
                'max': float(df[column].max()),
                'q25': float(df[column].quantile(0.25)),
                'q75': float(df[column].quantile(0.75))
            }
            for column in df.columns
        }
        
    def _generate_insights(self, trends: Dict[str, Dict[str, float]],
                         stats: Dict[str, Dict[str, float]]) -> List[Dict[str, str]]:
        """Generate insights from analysis"""
        insights = []
        
        # Analyze trends
        for metric, trend in trends.items():
            if abs(trend['slope']) > 0.1:
                direction = 'increasing' if trend['slope'] > 0 else 'decreasing'
                insights.append({
                    'type': 'trend',
                    'metric': metric,
                    'description': f"Significant {direction} trend detected",
                    'importance': 'high' if abs(trend['slope']) > 0.2 else 'medium'
                })
                
        # Analyze stability
        for metric, stat in stats.items():
            if 'std' in stat and stat['std'] > 0.2 * stat['mean']:
                insights.append({
                    'type': 'stability',
                    'metric': metric,
                    'description': "High variability detected",
                    'importance': 'medium'
                })
                
        return insights

class SystemAnalyzer:
    """Analyzes overall system behavior"""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.pattern_analyzer = PatternAnalyzer(config)
        self.performance_analyzer = PerformanceAnalyzer(config)
        self.system_history = deque(maxlen=config.history_length)
        
    def analyze_system(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system state"""
        # Store state
        self.system_history.append(state)
        
        # Analyze patterns
        patterns = self.pattern_analyzer.analyze_pattern(
            self._extract_pattern_data(state)
        )
        
        # Analyze performance
        performance = self.performance_analyzer.analyze_performance(
            self._extract_metrics(state)
        )
        
        # Analyze system structure
        structure = self._analyze_structure(state)
        
        return {
            'patterns': patterns,
            'performance': performance,
            'structure': structure,
            'recommendations': self._generate_recommendations(
                patterns, performance, structure
            )
        }
        
    def _extract_pattern_data(self, state: Dict[str, Any]) -> np.ndarray:
        """Extract pattern-relevant data from state"""
        # Implementation depends on state structure
        return np.array([])
        
    def _extract_metrics(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Extract performance metrics from state"""
        # Implementation depends on state structure
        return {}
        
    def _analyze_structure(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system structure"""
        # Implementation depends on state structure
        return {}
        
    def _generate_recommendations(self, 
                                patterns: Dict[str, Any],
                                performance: Dict[str, Any],
                                structure: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate system recommendations"""
        recommendations = []
        
        # Pattern-based recommendations
        if patterns['metrics']['entropy'] > 0.8:
            recommendations.append({
                'type': 'pattern',
                'description': "High pattern complexity detected",
                'suggestion': "Consider pattern simplification"
            })
            
        # Performance-based recommendations
        for insight in performance['insights']:
            if insight['importance'] == 'high':
                recommendations.append({
                    'type': 'performance',
                    'description': insight['description'],
                    'suggestion': "Investigate metric behavior"
                })
                
        return recommendations

class FractiAnalytics:
    """Main analytics system with fractal awareness"""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.system_analyzer = SystemAnalyzer(config)
        self.analysis_history = deque(maxlen=config.history_length)
        
    def analyze(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive system analysis"""
        # Analyze system
        analysis = self.system_analyzer.analyze_system(state)
        
        # Store analysis
        self.analysis_history.append({
            'timestamp': time.time(),
            'analysis': analysis
        })
        
        # Generate report
        return {
            'current_analysis': analysis,
            'historical_trends': self._analyze_history(),
            'recommendations': analysis['recommendations']
        }
        
    def _analyze_history(self) -> Dict[str, Any]:
        """Analyze historical trends"""
        if len(self.analysis_history) < 2:
            return {}
            
        return {
            'pattern_evolution': self._analyze_pattern_evolution(),
            'performance_trends': self._analyze_performance_trends(),
            'system_stability': self._analyze_system_stability()
        }
        
    def _analyze_pattern_evolution(self) -> Dict[str, float]:
        """Analyze evolution of patterns"""
        if not self.analysis_history:
            return {}
            
        pattern_metrics = [
            a['analysis']['patterns']['metrics']
            for a in self.analysis_history
        ]
        
        return {
            'complexity_trend': float(np.mean([m['entropy'] for m in pattern_metrics])),
            'stability_trend': float(1.0 - np.std([m['peak_ratio'] 
                                                  for m in pattern_metrics]))
        }
        
    def _analyze_performance_trends(self) -> Dict[str, float]:
        """Analyze performance trends"""
        if not self.analysis_history:
            return {}
            
        performance_metrics = [
            a['analysis']['performance']['statistics']
            for a in self.analysis_history
        ]
        
        trends = {}
        for metric in performance_metrics[0].keys():
            values = [m[metric]['mean'] for m in performance_metrics]
            trends[metric] = {
                'trend': float(np.mean(np.diff(values))) if len(values) > 1 else 0.0,
                'stability': float(1.0 - np.std(values) / (np.mean(np.abs(values)) + 1e-7))
            }
            
        return trends
        
    def _analyze_system_stability(self) -> float:
        """Analyze overall system stability"""
        if not self.analysis_history:
            return 1.0
            
        # Combine multiple stability indicators
        pattern_stability = self._analyze_pattern_evolution()['stability_trend']
        performance_stabilities = [
            t['stability'] 
            for t in self._analyze_performance_trends().values()
        ]
        
        return float(np.mean([pattern_stability] + performance_stabilities)) 