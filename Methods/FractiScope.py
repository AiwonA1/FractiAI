"""
FractiScope.py

Implements integrated analysis capabilities for FractiAI, providing real-time
monitoring, pattern detection, and system-wide validation across dimensions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy import stats
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class ScopeConfig:
    """Configuration for FractiScope analysis"""
    history_size: int = 1000
    pattern_threshold: float = 0.75
    dimension_weights: Dict[str, float] = None
    analysis_interval: int = 50
    confidence_threshold: float = 0.95

class PatternBank:
    """Stores and analyzes fractal patterns"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.patterns = deque(maxlen=capacity)
        self.metadata = {}
        
    def add_pattern(self, pattern: np.ndarray, metadata: Dict) -> None:
        """Store new pattern with metadata"""
        self.patterns.append(pattern)
        pattern_id = len(self.patterns) - 1
        self.metadata[pattern_id] = {
            'timestamp': np.datetime64('now'),
            'source': metadata.get('source', 'unknown'),
            'dimension': metadata.get('dimension', 'unknown'),
            'confidence': metadata.get('confidence', 1.0)
        }
    
    def analyze_evolution(self) -> Dict[str, Any]:
        """Analyze pattern evolution over time"""
        if len(self.patterns) < 2:
            return {}
            
        patterns = np.array(self.patterns)
        return {
            'trend': self._compute_trend(patterns),
            'stability': self._compute_stability(patterns),
            'complexity': self._estimate_complexity(patterns)
        }
    
    def _compute_trend(self, patterns: np.ndarray) -> Dict[str, float]:
        """Compute trend statistics for pattern evolution"""
        trend = np.mean(np.diff(patterns, axis=0), axis=0)
        return {
            'mean_change': float(np.mean(np.abs(trend))),
            'direction': float(np.mean(np.sign(trend)))
        }
    
    def _compute_stability(self, patterns: np.ndarray) -> float:
        """Measure pattern stability over time"""
        variations = np.std(patterns, axis=0)
        return float(1.0 - np.mean(variations))
    
    def _estimate_complexity(self, patterns: np.ndarray) -> float:
        """Estimate pattern complexity using approximate entropy"""
        # Simplified complexity estimation
        return float(np.mean([stats.entropy(p) for p in patterns]))

class DimensionalAnalyzer:
    """Analyzes patterns across organic, inorganic, and abstract dimensions"""
    
    def __init__(self, config: ScopeConfig):
        self.config = config
        self.dimension_banks = {
            'organic': PatternBank(config.history_size),
            'inorganic': PatternBank(config.history_size),
            'abstract': PatternBank(config.history_size)
        }
        
    def analyze_pattern(self, pattern: np.ndarray, dimension: str) -> Dict:
        """Analyze pattern in specific dimension"""
        if dimension not in self.dimension_banks:
            raise ValueError(f"Unknown dimension: {dimension}")
            
        bank = self.dimension_banks[dimension]
        
        # Compute pattern metrics
        metrics = {
            'complexity': self._compute_pattern_complexity(pattern),
            'coherence': self._measure_pattern_coherence(pattern),
            'dimension_alignment': self._check_dimension_alignment(pattern, dimension)
        }
        
        # Store pattern if significant
        if metrics['coherence'] > self.config.pattern_threshold:
            bank.add_pattern(pattern, {
                'dimension': dimension,
                'confidence': metrics['coherence']
            })
            
        return metrics
    
    def _compute_pattern_complexity(self, pattern: np.ndarray) -> float:
        """Compute complexity of pattern"""
        return float(stats.entropy(pattern.flatten()))
    
    def _measure_pattern_coherence(self, pattern: np.ndarray) -> float:
        """Measure internal coherence of pattern"""
        if pattern.size < 2:
            return 1.0
        
        # Compute auto-correlation
        correlation = np.corrcoef(pattern.flatten(), pattern.flatten())[0,1]
        return float(abs(correlation))
    
    def _check_dimension_alignment(self, pattern: np.ndarray, dimension: str) -> float:
        """Check how well pattern aligns with dimension characteristics"""
        # Simplified dimension-specific checks
        if dimension == 'organic':
            # Look for smooth transitions
            return float(1.0 - np.mean(np.abs(np.diff(pattern))))
        elif dimension == 'inorganic':
            # Look for structured patterns
            return float(np.mean(np.abs(np.fft.fft(pattern.flatten()))))
        else:  # abstract
            # Look for hierarchical patterns
            return float(np.mean([stats.entropy(p) for p in np.split(pattern, 2)]))

class FractiScope:
    """Main analysis and monitoring system for FractiAI"""
    
    def __init__(self, config: ScopeConfig):
        self.config = config
        self.analyzer = DimensionalAnalyzer(config)
        self.global_metrics = {}
        self.analysis_count = 0
        
    def analyze_system(self, state_dict: Dict) -> Dict[str, Any]:
        """Perform comprehensive system analysis"""
        self.analysis_count += 1
        
        analysis_results = {
            'dimensional_analysis': self._analyze_dimensions(state_dict),
            'system_coherence': self._evaluate_coherence(state_dict),
            'performance_metrics': self._compute_performance(state_dict)
        }
        
        # Update global metrics periodically
        if self.analysis_count % self.config.analysis_interval == 0:
            self._update_global_metrics(analysis_results)
            
        return analysis_results
    
    def _analyze_dimensions(self, state_dict: Dict) -> Dict:
        """Analyze patterns across all dimensions"""
        dimension_results = {}
        
        for dimension in ['organic', 'inorganic', 'abstract']:
            if dimension in state_dict:
                pattern = state_dict[dimension]
                dimension_results[dimension] = self.analyzer.analyze_pattern(
                    pattern, dimension
                )
                
        return dimension_results
    
    def _evaluate_coherence(self, state_dict: Dict) -> Dict:
        """Evaluate system-wide coherence"""
        coherence_metrics = {
            'cross_dimensional': self._compute_cross_dimensional_coherence(state_dict),
            'temporal': self._compute_temporal_coherence(state_dict),
            'structural': self._compute_structural_coherence(state_dict)
        }
        
        return coherence_metrics
    
    def _compute_cross_dimensional_coherence(self, state_dict: Dict) -> float:
        """Compute coherence across dimensions"""
        if len(state_dict) < 2:
            return 1.0
            
        coherence_scores = []
        dimensions = list(state_dict.keys())
        
        for i in range(len(dimensions)):
            for j in range(i + 1, len(dimensions)):
                pattern1 = state_dict[dimensions[i]]
                pattern2 = state_dict[dimensions[j]]
                if pattern1.size == pattern2.size:
                    correlation = np.corrcoef(pattern1.flatten(), 
                                           pattern2.flatten())[0,1]
                    coherence_scores.append(abs(correlation))
                    
        return float(np.mean(coherence_scores)) if coherence_scores else 0.0
    
    def _compute_temporal_coherence(self, state_dict: Dict) -> float:
        """Compute temporal coherence of system state"""
        # Analyze pattern evolution in each dimension
        temporal_scores = []
        
        for dimension, bank in self.analyzer.dimension_banks.items():
            evolution = bank.analyze_evolution()
            if 'stability' in evolution:
                temporal_scores.append(evolution['stability'])
                
        return float(np.mean(temporal_scores)) if temporal_scores else 1.0
    
    def _compute_structural_coherence(self, state_dict: Dict) -> float:
        """Compute structural coherence of system"""
        # Analyze pattern structure across dimensions
        structure_scores = []
        
        for dimension, pattern in state_dict.items():
            if isinstance(pattern, np.ndarray):
                # Compute pattern self-similarity
                if pattern.size > 1:
                    splits = np.array_split(pattern, 2)
                    correlation = np.corrcoef(splits[0].flatten(), 
                                           splits[1].flatten())[0,1]
                    structure_scores.append(abs(correlation))
                    
        return float(np.mean(structure_scores)) if structure_scores else 1.0
    
    def _compute_performance(self, state_dict: Dict) -> Dict:
        """Compute system performance metrics"""
        return {
            'efficiency': self._compute_efficiency(state_dict),
            'adaptability': self._compute_adaptability(state_dict),
            'stability': self._compute_stability(state_dict)
        }
    
    def _compute_efficiency(self, state_dict: Dict) -> float:
        """Compute system efficiency"""
        # Measure processing efficiency across dimensions
        pattern_sizes = [p.size for p in state_dict.values() 
                        if isinstance(p, np.ndarray)]
        if not pattern_sizes:
            return 1.0
        return float(1.0 - np.std(pattern_sizes) / np.mean(pattern_sizes))
    
    def _compute_adaptability(self, state_dict: Dict) -> float:
        """Compute system adaptability"""
        # Analyze pattern evolution rates
        adaptation_scores = []
        
        for dimension, bank in self.analyzer.dimension_banks.items():
            evolution = bank.analyze_evolution()
            if 'trend' in evolution:
                adaptation_scores.append(evolution['trend']['mean_change'])
                
        return float(np.mean(adaptation_scores)) if adaptation_scores else 0.0
    
    def _compute_stability(self, state_dict: Dict) -> float:
        """Compute system stability"""
        stability_scores = []
        
        for dimension, pattern in state_dict.items():
            if isinstance(pattern, np.ndarray):
                stability_scores.append(1.0 - np.std(pattern) / np.mean(np.abs(pattern)))
                
        return float(np.mean(stability_scores)) if stability_scores else 1.0
    
    def _update_global_metrics(self, analysis_results: Dict) -> None:
        """Update global system metrics"""
        self.global_metrics = {
            'timestamp': np.datetime64('now'),
            'overall_coherence': np.mean([
                analysis_results['system_coherence']['cross_dimensional'],
                analysis_results['system_coherence']['temporal'],
                analysis_results['system_coherence']['structural']
            ]),
            'system_health': self._compute_system_health(analysis_results),
            'analysis_confidence': self._compute_analysis_confidence(analysis_results)
        }
    
    def _compute_system_health(self, analysis_results: Dict) -> float:
        """Compute overall system health score"""
        metrics = analysis_results['performance_metrics']
        return float(np.mean([
            metrics['efficiency'],
            metrics['adaptability'],
            metrics['stability']
        ]))
    
    def _compute_analysis_confidence(self, analysis_results: Dict) -> float:
        """Compute confidence level in analysis results"""
        # Consider multiple factors for confidence calculation
        factors = [
            len(analysis_results['dimensional_analysis']) / 3,  # Dimension coverage
            np.mean([m for m in analysis_results['system_coherence'].values()]),
            self._compute_system_health(analysis_results)
        ]
        
        confidence = np.mean(factors)
        return float(min(confidence, self.config.confidence_threshold)) 