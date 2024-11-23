"""
FractiAnalytics.py

Implements advanced pattern analysis and visualization capabilities for FractiAI,
providing deep insights into system behavior and fractal patterns across dimensions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy import stats, signal
from sklearn.decomposition import PCA
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class AnalyticsConfig:
    """Configuration for FractiAI analytics"""
    history_size: int = 1000
    pattern_threshold: float = 0.75
    analysis_interval: int = 50
    dimension_weights: Dict[str, float] = None
    pca_components: int = 3

class PatternAnalyzer:
    """Advanced pattern analysis system"""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.pattern_history = deque(maxlen=config.history_size)
        self.pca = PCA(n_components=config.pca_components)
        self.dimension_correlations = {}
        
    def analyze_pattern(self, pattern: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive pattern analysis"""
        self.pattern_history.append(pattern)
        
        analysis = {
            'fractal_metrics': self._compute_fractal_metrics(pattern),
            'temporal_evolution': self._analyze_temporal_evolution(),
            'dimensional_relationships': self._analyze_dimensions(pattern),
            'emergent_properties': self._detect_emergent_properties()
        }
        
        return analysis
    
    def _compute_fractal_metrics(self, pattern: np.ndarray) -> Dict[str, float]:
        """Compute various fractal characteristics"""
        return {
            'hurst_exponent': self._compute_hurst_exponent(pattern),
            'fractal_dimension': self._estimate_fractal_dimension(pattern),
            'self_similarity': self._measure_self_similarity(pattern),
            'complexity': self._compute_complexity(pattern)
        }
    
    def _compute_hurst_exponent(self, pattern: np.ndarray) -> float:
        """Compute Hurst exponent for pattern"""
        if pattern.size < 2:
            return 0.5
            
        # Simplified R/S analysis
        lags = range(2, min(len(pattern) // 2, 20))
        rs_values = []
        
        for lag in lags:
            rs = self._compute_rs_value(pattern, lag)
            rs_values.append(rs)
            
        if not rs_values:
            return 0.5
            
        # Compute Hurst exponent from slope
        log_rs = np.log(rs_values)
        log_lags = np.log(lags)
        slope, _ = np.polyfit(log_lags, log_rs, 1)
        
        return float(slope)
    
    def _compute_rs_value(self, pattern: np.ndarray, lag: int) -> float:
        """Compute R/S value for given lag"""
        pattern_mean = np.mean(pattern)
        pattern_std = np.std(pattern)
        
        if pattern_std == 0:
            return 1.0
            
        cumsum = np.cumsum(pattern - pattern_mean)
        r = np.max(cumsum) - np.min(cumsum)
        s = pattern_std
        
        return r/s if s > 0 else 1.0
    
    def _estimate_fractal_dimension(self, pattern: np.ndarray) -> float:
        """Estimate fractal dimension using box-counting method"""
        if pattern.size < 2:
            return 1.0
            
        # Simplified box-counting
        scales = [2, 4, 8]
        counts = []
        
        for scale in scales:
            reshaped = pattern.reshape(-1, scale)
            count = np.sum(np.any(reshaped, axis=1))
            counts.append(count)
            
        if len(counts) > 1:
            # Compute dimension from slope
            log_counts = np.log(counts)
            log_scales = np.log(scales)
            slope, _ = np.polyfit(log_scales, log_counts, 1)
            return float(abs(slope))
            
        return 1.0
    
    def _measure_self_similarity(self, pattern: np.ndarray) -> float:
        """Measure pattern self-similarity across scales"""
        if pattern.size < 4:
            return 1.0
            
        # Compare pattern at different scales
        half_size = pattern.size // 2
        similarity_scores = []
        
        original = pattern[:half_size]
        scaled = signal.resample(pattern[half_size:], len(original))
        
        correlation = np.corrcoef(original, scaled)[0,1]
        return float(abs(correlation))
    
    def _compute_complexity(self, pattern: np.ndarray) -> float:
        """Compute pattern complexity using approximate entropy"""
        return float(stats.entropy(pattern.flatten()))
    
    def _analyze_temporal_evolution(self) -> Dict[str, Any]:
        """Analyze pattern evolution over time"""
        if len(self.pattern_history) < 2:
            return {}
            
        patterns = np.array(self.pattern_history)
        
        return {
            'trend': self._compute_trend(patterns),
            'stability': self._compute_stability(patterns),
            'periodicity': self._detect_periodicity(patterns),
            'divergence': self._measure_divergence(patterns)
        }
    
    def _compute_trend(self, patterns: np.ndarray) -> Dict[str, float]:
        """Compute trend statistics"""
        trend = np.mean(np.diff(patterns, axis=0), axis=0)
        return {
            'magnitude': float(np.mean(np.abs(trend))),
            'direction': float(np.mean(np.sign(trend))),
            'acceleration': float(np.mean(np.diff(np.abs(trend))))
        }
    
    def _compute_stability(self, patterns: np.ndarray) -> float:
        """Compute pattern stability over time"""
        return float(1.0 - np.std(patterns) / (np.mean(np.abs(patterns)) + 1e-7))
    
    def _detect_periodicity(self, patterns: np.ndarray) -> Dict[str, float]:
        """Detect periodic behavior in patterns"""
        if patterns.shape[0] < 4:
            return {'periodicity': 0.0, 'frequency': 0.0}
            
        # Compute autocorrelation
        mean_pattern = np.mean(patterns, axis=1)
        autocorr = signal.correlate(mean_pattern, mean_pattern, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks
        peaks, _ = signal.find_peaks(autocorr)
        
        if len(peaks) > 1:
            return {
                'periodicity': float(np.mean(np.diff(peaks))),
                'frequency': float(1.0 / (np.mean(np.diff(peaks)) + 1e-7))
            }
        return {'periodicity': 0.0, 'frequency': 0.0}
    
    def _measure_divergence(self, patterns: np.ndarray) -> float:
        """Measure pattern divergence over time"""
        if patterns.shape[0] < 2:
            return 0.0
            
        distances = np.linalg.norm(np.diff(patterns, axis=0), axis=1)
        return float(np.mean(distances))

class FractiAnalytics:
    """Main analytics system for FractiAI"""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.analyzer = PatternAnalyzer(config)
        self.global_patterns = {}
        self.cross_dimensional_metrics = {}
        
    def analyze_patterns(self, patterns: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Perform comprehensive pattern analysis across dimensions"""
        analysis_results = {}
        
        # Analyze individual patterns
        for dimension, pattern in patterns.items():
            analysis_results[dimension] = self.analyzer.analyze_pattern(pattern)
            
        # Compute cross-dimensional metrics
        self.cross_dimensional_metrics = self._compute_cross_dimensional_metrics(patterns)
        
        # Update global patterns
        self._update_global_patterns(patterns)
        
        return {
            'dimensional_analysis': analysis_results,
            'cross_dimensional': self.cross_dimensional_metrics,
            'global_patterns': self.global_patterns
        }
    
    def _compute_cross_dimensional_metrics(self, 
                                        patterns: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute metrics across dimensions"""
        if len(patterns) < 2:
            return {}
            
        # Compute correlations between dimensions
        correlations = {}
        dimensions = list(patterns.keys())
        
        for i in range(len(dimensions)):
            for j in range(i + 1, len(dimensions)):
                dim1, dim2 = dimensions[i], dimensions[j]
                if patterns[dim1].size == patterns[dim2].size:
                    corr = np.corrcoef(patterns[dim1].flatten(), 
                                     patterns[dim2].flatten())[0,1]
                    correlations[f"{dim1}-{dim2}"] = float(abs(corr))
                    
        return {
            'dimensional_correlations': correlations,
            'harmony_score': float(np.mean(list(correlations.values()))) if correlations else 0.0
        }
    
    def _update_global_patterns(self, patterns: Dict[str, np.ndarray]) -> None:
        """Update global pattern analysis"""
        if not patterns:
            return
            
        # Combine patterns across dimensions
        combined = np.concatenate([p.flatten() for p in patterns.values()])
        
        # Update global patterns
        self.global_patterns = {
            'complexity': float(stats.entropy(combined)),
            'stability': float(1.0 - np.std(combined) / (np.mean(np.abs(combined)) + 1e-7)),
            'dimensionality': self._estimate_effective_dimensions(combined)
        }
    
    def _estimate_effective_dimensions(self, pattern: np.ndarray) -> int:
        """Estimate effective dimensionality of pattern"""
        if pattern.size < self.config.pca_components:
            return 1
            
        # Reshape for PCA if needed
        reshaped = pattern.reshape(-1, 1) if pattern.ndim == 1 else pattern
        
        # Fit PCA
        self.pca.fit(reshaped)
        
        # Count significant components
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Return number of components needed to explain 95% of variance
        return int(np.sum(cumulative_variance < 0.95) + 1) 