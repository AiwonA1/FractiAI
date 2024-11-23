"""
TemporalMetaInference.py

Implements temporal meta-level active inference for FractiAI, enabling meta-learning
and adaptive inference across different time scales through fractal patterns.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy.stats import entropy
from scipy.special import softmax
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class TemporalConfig:
    """Configuration for temporal meta-inference"""
    time_scales: List[int] = (1, 5, 10, 50, 100)  # Different time scales
    memory_size: int = 1000
    meta_horizon: int = 10
    adaptation_rate: float = 0.01
    fractal_depth: int = 3
    prediction_horizon: int = 5

class TimeScale:
    """Represents processing at a specific time scale"""
    
    def __init__(self, scale: int, memory_size: int):
        self.scale = scale
        self.memory = deque(maxlen=memory_size)
        self.predictions = deque(maxlen=memory_size)
        self.meta_params = {
            'learning_rate': 0.01,
            'prediction_weight': 0.5,
            'pattern_threshold': 0.7
        }
        self.patterns = []
        
    def process(self, data: np.ndarray) -> Dict[str, Any]:
        """Process data at this time scale"""
        # Store data
        self.memory.append(data)
        
        # Generate prediction
        prediction = self._generate_prediction()
        self.predictions.append(prediction)
        
        # Update patterns
        if len(self.memory) >= self.scale:
            pattern = self._extract_pattern()
            if pattern is not None:
                self.patterns.append(pattern)
                
        return {
            'prediction': prediction,
            'pattern': self.patterns[-1] if self.patterns else None,
            'metrics': self._compute_metrics()
        }
        
    def _generate_prediction(self) -> np.ndarray:
        """Generate prediction based on patterns"""
        if not self.patterns:
            return np.zeros_like(self.memory[0]) if self.memory else np.array([])
            
        # Use recent patterns for prediction
        recent_patterns = self.patterns[-3:]
        prediction = np.mean([p['values'] for p in recent_patterns], axis=0)
        
        return prediction
        
    def _extract_pattern(self) -> Optional[Dict[str, Any]]:
        """Extract temporal pattern at current scale"""
        if len(self.memory) < self.scale:
            return None
            
        # Get data window
        window = list(self.memory)[-self.scale:]
        values = np.array(window)
        
        # Compute pattern metrics
        pattern = {
            'values': values.mean(axis=0),
            'variance': values.var(axis=0),
            'trend': np.mean(np.diff(values, axis=0), axis=0),
            'scale': self.scale
        }
        
        return pattern
        
    def _compute_metrics(self) -> Dict[str, float]:
        """Compute time scale metrics"""
        if not self.memory or not self.predictions:
            return {}
            
        # Compute prediction error
        if len(self.memory) > 1:
            prediction_error = np.mean((self.predictions[-1] - self.memory[-1]) ** 2)
        else:
            prediction_error = 0.0
            
        return {
            'prediction_error': float(prediction_error),
            'pattern_count': len(self.patterns),
            'memory_usage': len(self.memory) / self.memory.maxlen
        }
        
    def adapt_meta_params(self, feedback: float) -> None:
        """Adapt meta-parameters based on feedback"""
        # Update learning rate
        self.meta_params['learning_rate'] *= (1 + feedback * 0.1)
        
        # Update prediction weight
        self.meta_params['prediction_weight'] *= (1 + feedback * 0.05)
        
        # Update pattern threshold
        self.meta_params['pattern_threshold'] *= (1 + feedback * 0.01)

class TemporalMetaInference:
    """Temporal meta-learning system with fractal patterns"""
    
    def __init__(self, config: TemporalConfig):
        self.config = config
        self.time_scales: Dict[int, TimeScale] = {}
        self.meta_memory = deque(maxlen=config.memory_size)
        self.temporal_patterns = []
        self.initialize_scales()
        
    def initialize_scales(self) -> None:
        """Initialize time scales"""
        for scale in self.config.time_scales:
            self.time_scales[scale] = TimeScale(scale, self.config.memory_size)
            
    def process(self, data: np.ndarray) -> Dict[str, Any]:
        """Process data through temporal scales"""
        scale_results = {}
        
        # Process through each time scale
        for scale, processor in self.time_scales.items():
            results = processor.process(data)
            scale_results[scale] = results
            
        # Extract cross-scale patterns
        cross_scale_patterns = self._extract_cross_scale_patterns(scale_results)
        
        # Generate predictions
        predictions = self._generate_predictions(scale_results)
        
        # Update meta-memory
        self._update_meta_memory(data, scale_results)
        
        return {
            'scale_results': scale_results,
            'cross_scale_patterns': cross_scale_patterns,
            'predictions': predictions,
            'metrics': self._compute_metrics(scale_results)
        }
        
    def _extract_cross_scale_patterns(self, 
                                    scale_results: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract patterns across time scales"""
        patterns = []
        
        # Look for similar patterns across scales
        for scale1 in self.config.time_scales:
            for scale2 in self.config.time_scales:
                if scale1 < scale2:
                    pattern1 = scale_results[scale1].get('pattern')
                    pattern2 = scale_results[scale2].get('pattern')
                    
                    if pattern1 is not None and pattern2 is not None:
                        similarity = self._compute_pattern_similarity(
                            pattern1['values'],
                            pattern2['values']
                        )
                        
                        if similarity > 0.7:  # Similarity threshold
                            patterns.append({
                                'scales': (scale1, scale2),
                                'similarity': similarity,
                                'pattern': (pattern1['values'] + pattern2['values']) / 2
                            })
                            
        return patterns
        
    def _generate_predictions(self, 
                            scale_results: Dict[int, Dict[str, Any]]) -> Dict[int, np.ndarray]:
        """Generate predictions for each time scale"""
        predictions = {}
        
        for scale, processor in self.time_scales.items():
            # Get base prediction from scale
            base_prediction = scale_results[scale]['prediction']
            
            # Adjust prediction using cross-scale information
            adjusted_prediction = self._adjust_prediction(
                base_prediction,
                scale,
                scale_results
            )
            
            predictions[scale] = adjusted_prediction
            
        return predictions
        
    def _adjust_prediction(self, base_prediction: np.ndarray,
                         scale: int,
                         scale_results: Dict[int, Dict[str, Any]]) -> np.ndarray:
        """Adjust prediction using cross-scale information"""
        # Get predictions from other scales
        other_predictions = []
        for other_scale, results in scale_results.items():
            if other_scale != scale:
                other_predictions.append(results['prediction'])
                
        if not other_predictions:
            return base_prediction
            
        # Compute weighted average
        weights = softmax([1/abs(s - scale) for s in self.config.time_scales 
                         if s != scale])
        adjusted = np.average(other_predictions, axis=0, weights=weights)
        
        # Combine with base prediction
        return 0.7 * base_prediction + 0.3 * adjusted
        
    def _update_meta_memory(self, data: np.ndarray,
                          scale_results: Dict[int, Dict[str, Any]]) -> None:
        """Update meta-level memory"""
        meta_experience = {
            'data': data,
            'scale_results': scale_results,
            'timestamp': len(self.meta_memory)
        }
        self.meta_memory.append(meta_experience)
        
        # Update temporal patterns
        if len(self.meta_memory) >= max(self.config.time_scales):
            pattern = self._extract_temporal_pattern()
            if pattern is not None:
                self.temporal_patterns.append(pattern)
                
    def _extract_temporal_pattern(self) -> Optional[Dict[str, Any]]:
        """Extract temporal pattern from meta-memory"""
        if not self.meta_memory:
            return None
            
        # Analyze recent memory
        recent_memory = list(self.meta_memory)[-max(self.config.time_scales):]
        
        # Extract pattern features
        pattern = {
            'duration': len(recent_memory),
            'scales': self._analyze_scale_importance(recent_memory),
            'transitions': self._analyze_transitions(recent_memory)
        }
        
        return pattern
        
    def _analyze_scale_importance(self, 
                                memory: List[Dict[str, Any]]) -> Dict[int, float]:
        """Analyze importance of different time scales"""
        scale_errors = {scale: [] for scale in self.config.time_scales}
        
        for experience in memory:
            for scale, results in experience['scale_results'].items():
                if 'metrics' in results:
                    error = results['metrics'].get('prediction_error', 1.0)
                    scale_errors[scale].append(error)
                    
        return {
            scale: 1.0 / (np.mean(errors) + 1e-7)
            for scale, errors in scale_errors.items()
        }
        
    def _analyze_transitions(self, 
                           memory: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze temporal transitions"""
        if len(memory) < 2:
            return {}
            
        transitions = []
        for i in range(len(memory)-1):
            transition = np.mean([
                np.mean((memory[i+1]['data'] - memory[i]['data']) ** 2)
                for scale in self.config.time_scales
            ])
            transitions.append(transition)
            
        return {
            'mean_transition': float(np.mean(transitions)),
            'transition_stability': float(1.0 - np.std(transitions))
        }
        
    def _compute_metrics(self, 
                        scale_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Compute comprehensive metrics"""
        return {
            'scale_metrics': {
                scale: results['metrics']
                for scale, results in scale_results.items()
            },
            'temporal_metrics': self._compute_temporal_metrics(),
            'meta_metrics': self._compute_meta_metrics()
        }
        
    def _compute_temporal_metrics(self) -> Dict[str, float]:
        """Compute temporal learning metrics"""
        if not self.temporal_patterns:
            return {}
            
        return {
            'pattern_count': len(self.temporal_patterns),
            'scale_diversity': self._compute_scale_diversity(),
            'temporal_stability': self._compute_temporal_stability()
        }
        
    def _compute_scale_diversity(self) -> float:
        """Compute diversity of scale usage"""
        if not self.temporal_patterns:
            return 0.0
            
        scale_usage = np.zeros(len(self.config.time_scales))
        for pattern in self.temporal_patterns:
            for scale, importance in pattern['scales'].items():
                scale_idx = self.config.time_scales.index(scale)
                scale_usage[scale_idx] += importance
                
        scale_usage = scale_usage / (np.sum(scale_usage) + 1e-7)
        return float(-np.sum(scale_usage * np.log(scale_usage + 1e-7)))
        
    def _compute_temporal_stability(self) -> float:
        """Compute stability of temporal patterns"""
        if len(self.temporal_patterns) < 2:
            return 1.0
            
        transitions = [p['transitions']['mean_transition'] 
                      for p in self.temporal_patterns]
        return float(1.0 - np.std(transitions) / (np.mean(transitions) + 1e-7))
        
    def _compute_meta_metrics(self) -> Dict[str, float]:
        """Compute meta-learning metrics"""
        return {
            'memory_usage': len(self.meta_memory) / self.meta_memory.maxlen,
            'pattern_complexity': self._compute_pattern_complexity(),
            'learning_progress': self._compute_learning_progress()
        }
        
    def _compute_pattern_complexity(self) -> float:
        """Compute complexity of temporal patterns"""
        if not self.temporal_patterns:
            return 0.0
            
        complexities = [len(p['scales']) * p['duration'] 
                       for p in self.temporal_patterns]
        return float(np.mean(complexities))
        
    def _compute_learning_progress(self) -> float:
        """Compute meta-learning progress"""
        if len(self.meta_memory) < 2:
            return 0.0
            
        recent_errors = []
        for experience in list(self.meta_memory)[-10:]:
            scale_errors = [
                results['metrics'].get('prediction_error', 1.0)
                for results in experience['scale_results'].values()
            ]
            recent_errors.append(np.mean(scale_errors))
            
        if len(recent_errors) < 2:
            return 0.0
            
        return float(-np.mean(np.diff(recent_errors)))
        
    def _compute_pattern_similarity(self, pattern1: np.ndarray, 
                                  pattern2: np.ndarray) -> float:
        """Compute similarity between patterns"""
        return float(np.abs(np.corrcoef(pattern1, pattern2)[0,1]))
</rewritten_file> 