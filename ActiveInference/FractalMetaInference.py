"""
FractalMetaInference.py

Implements fractal-based meta-level active inference for FractiAI, enabling 
self-similar learning and pattern discovery across scales through fractal patterns.
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
class FractalConfig:
    """Configuration for fractal meta-inference"""
    fractal_depth: int = 5
    scale_levels: int = 4
    pattern_dim: int = 32
    memory_size: int = 1000
    learning_rate: float = 0.01
    similarity_threshold: float = 0.7

class FractalPattern:
    """Represents a fractal pattern with self-similar structure"""
    
    def __init__(self, pattern_id: str, base_pattern: np.ndarray):
        self.pattern_id = pattern_id
        self.base_pattern = base_pattern
        self.sub_patterns = []
        self.super_patterns = []
        self.scale_transforms = {}
        self.similarity_scores = {}
        
    def add_sub_pattern(self, pattern: np.ndarray, scale: float) -> None:
        """Add sub-pattern at given scale"""
        self.sub_patterns.append({
            'pattern': pattern,
            'scale': scale,
            'similarity': self._compute_similarity(pattern)
        })
        
    def add_super_pattern(self, pattern: np.ndarray, scale: float) -> None:
        """Add super-pattern at given scale"""
        self.super_patterns.append({
            'pattern': pattern,
            'scale': scale,
            'similarity': self._compute_similarity(pattern)
        })
        
    def _compute_similarity(self, pattern: np.ndarray) -> float:
        """Compute similarity with base pattern"""
        # Resize pattern if needed
        if pattern.shape != self.base_pattern.shape:
            pattern = self._rescale_pattern(pattern, len(self.base_pattern))
            
        return float(np.abs(np.corrcoef(pattern, self.base_pattern)[0,1]))
        
    def _rescale_pattern(self, pattern: np.ndarray, target_size: int) -> np.ndarray:
        """Rescale pattern to target size"""
        if len(pattern) == target_size:
            return pattern
            
        # Simple averaging for downscaling
        if len(pattern) > target_size:
            ratio = len(pattern) // target_size
            return np.mean(pattern.reshape(-1, ratio), axis=1)
            
        # Simple interpolation for upscaling
        ratio = target_size // len(pattern)
        return np.repeat(pattern, ratio)

class FractalMemory:
    """Memory system with fractal organization"""
    
    def __init__(self, config: FractalConfig):
        self.config = config
        self.patterns: Dict[str, FractalPattern] = {}
        self.scale_indices: Dict[float, List[str]] = {}
        self.pattern_history = deque(maxlen=config.memory_size)
        
    def store_pattern(self, pattern: np.ndarray, scale: float = 1.0) -> str:
        """Store new pattern with fractal analysis"""
        # Generate pattern ID
        pattern_id = f"PATTERN_{len(self.patterns)}"
        
        # Create fractal pattern
        fractal_pattern = FractalPattern(pattern_id, pattern)
        
        # Analyze at different scales
        self._analyze_scales(fractal_pattern, pattern, scale)
        
        # Store pattern
        self.patterns[pattern_id] = fractal_pattern
        
        # Update scale indices
        if scale not in self.scale_indices:
            self.scale_indices[scale] = []
        self.scale_indices[scale].append(pattern_id)
        
        # Store in history
        self.pattern_history.append({
            'pattern_id': pattern_id,
            'pattern': pattern,
            'scale': scale
        })
        
        return pattern_id
        
    def _analyze_scales(self, fractal_pattern: FractalPattern, 
                       pattern: np.ndarray, base_scale: float) -> None:
        """Analyze pattern at different scales"""
        # Analyze sub-patterns
        for i in range(1, self.config.scale_levels):
            scale = base_scale / (2 ** i)
            sub_pattern = self._extract_sub_pattern(pattern, scale)
            fractal_pattern.add_sub_pattern(sub_pattern, scale)
            
        # Analyze super-patterns
        for i in range(1, self.config.scale_levels):
            scale = base_scale * (2 ** i)
            super_pattern = self._extract_super_pattern(pattern, scale)
            fractal_pattern.add_super_pattern(super_pattern, scale)
            
    def _extract_sub_pattern(self, pattern: np.ndarray, scale: float) -> np.ndarray:
        """Extract sub-pattern at given scale"""
        target_size = max(1, int(len(pattern) * scale))
        return pattern[:target_size]
        
    def _extract_super_pattern(self, pattern: np.ndarray, scale: float) -> np.ndarray:
        """Extract super-pattern at given scale"""
        target_size = int(len(pattern) * scale)
        return np.repeat(pattern, int(scale))[:target_size]
        
    def find_similar_patterns(self, pattern: np.ndarray, 
                            scale: float = 1.0) -> List[Dict[str, Any]]:
        """Find similar patterns at given scale"""
        similar_patterns = []
        
        # Check patterns at same scale
        if scale in self.scale_indices:
            for pattern_id in self.scale_indices[scale]:
                stored_pattern = self.patterns[pattern_id]
                similarity = stored_pattern._compute_similarity(pattern)
                
                if similarity > self.config.similarity_threshold:
                    similar_patterns.append({
                        'pattern_id': pattern_id,
                        'similarity': similarity,
                        'pattern': stored_pattern
                    })
                    
        return similar_patterns

class FractalMetaLearner:
    """Meta-learning system with fractal organization"""
    
    def __init__(self, config: FractalConfig):
        self.config = config
        self.memory = FractalMemory(config)
        self.meta_patterns = {}
        self.learning_history = []
        
    def learn_pattern(self, pattern: np.ndarray, 
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Learn new pattern with meta-level analysis"""
        # Store pattern
        pattern_id = self.memory.store_pattern(pattern)
        
        # Find similar patterns
        similar_patterns = self.memory.find_similar_patterns(pattern)
        
        # Extract meta-pattern if similar patterns found
        if similar_patterns:
            meta_pattern = self._extract_meta_pattern(similar_patterns)
            self.meta_patterns[pattern_id] = meta_pattern
            
        # Store learning experience
        experience = {
            'pattern_id': pattern_id,
            'similar_patterns': similar_patterns,
            'context': context,
            'meta_pattern': self.meta_patterns.get(pattern_id)
        }
        self.learning_history.append(experience)
        
        return experience
        
    def _extract_meta_pattern(self, 
                            similar_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract meta-pattern from similar patterns"""
        patterns = [p['pattern'].base_pattern for p in similar_patterns]
        similarities = [p['similarity'] for p in similar_patterns]
        
        # Weight patterns by similarity
        weights = softmax(similarities)
        meta_pattern = np.average(patterns, axis=0, weights=weights)
        
        return {
            'pattern': meta_pattern,
            'source_patterns': [p['pattern_id'] for p in similar_patterns],
            'confidence': float(np.mean(similarities))
        }

class FractalMetaInference:
    """Fractal-based meta-inference system"""
    
    def __init__(self, config: FractalConfig):
        self.config = config
        self.meta_learner = FractalMetaLearner(config)
        self.inference_history = []
        
    def process(self, input_data: np.ndarray, 
               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process input through fractal meta-inference"""
        # Learn pattern
        learning_result = self.meta_learner.learn_pattern(input_data, context)
        
        # Generate prediction using meta-patterns
        prediction = self._generate_prediction(input_data, learning_result)
        
        # Store inference
        inference = {
            'input': input_data,
            'learning_result': learning_result,
            'prediction': prediction,
            'context': context
        }
        self.inference_history.append(inference)
        
        return {
            'output': prediction,
            'meta_pattern': learning_result.get('meta_pattern'),
            'metrics': self._compute_metrics(inference)
        }
        
    def _generate_prediction(self, input_data: np.ndarray,
                           learning_result: Dict[str, Any]) -> np.ndarray:
        """Generate prediction using meta-patterns"""
        if 'meta_pattern' not in learning_result:
            return input_data
            
        meta_pattern = learning_result['meta_pattern']
        prediction = meta_pattern['pattern']
        
        # Scale prediction to input size
        if len(prediction) != len(input_data):
            prediction = np.interp(
                np.linspace(0, 1, len(input_data)),
                np.linspace(0, 1, len(prediction)),
                prediction
            )
            
        return prediction
        
    def _compute_metrics(self, inference: Dict[str, Any]) -> Dict[str, float]:
        """Compute fractal meta-inference metrics"""
        return {
            'pattern_metrics': self._compute_pattern_metrics(inference),
            'learning_metrics': self._compute_learning_metrics(),
            'fractal_metrics': self._compute_fractal_metrics()
        }
        
    def _compute_pattern_metrics(self, 
                               inference: Dict[str, Any]) -> Dict[str, float]:
        """Compute pattern-related metrics"""
        if 'meta_pattern' not in inference['learning_result']:
            return {}
            
        meta_pattern = inference['learning_result']['meta_pattern']
        return {
            'pattern_confidence': meta_pattern['confidence'],
            'pattern_complexity': float(entropy(softmax(meta_pattern['pattern']))),
            'pattern_count': len(meta_pattern['source_patterns'])
        }
        
    def _compute_learning_metrics(self) -> Dict[str, float]:
        """Compute learning-related metrics"""
        if not self.inference_history:
            return {}
            
        # Analyze recent learning
        recent = self.inference_history[-min(10, len(self.inference_history)):]
        confidences = [
            inf['learning_result'].get('meta_pattern', {}).get('confidence', 0)
            for inf in recent
        ]
        
        return {
            'learning_progress': float(np.mean(np.diff(confidences))) if len(confidences) > 1 else 0.0,
            'learning_stability': float(1.0 - np.std(confidences)) if confidences else 0.0,
            'pattern_discovery_rate': float(len(self.meta_learner.meta_patterns) / 
                                         len(self.inference_history))
        }
        
    def _compute_fractal_metrics(self) -> Dict[str, float]:
        """Compute fractal-related metrics"""
        if not self.meta_learner.memory.patterns:
            return {}
            
        # Analyze fractal properties
        scale_counts = {
            scale: len(patterns)
            for scale, patterns in self.meta_learner.memory.scale_indices.items()
        }
        
        return {
            'scale_diversity': float(entropy(list(scale_counts.values()))),
            'fractal_depth': float(np.mean([
                len(p.sub_patterns) + len(p.super_patterns)
                for p in self.meta_learner.memory.patterns.values()
            ])),
            'pattern_reuse': float(np.mean([
                len(p.get('source_patterns', []))
                for p in self.meta_learner.meta_patterns.values()
            ]))
        } 