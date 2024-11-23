"""
FractalTemplate.py

Implements Master Fractal Templates that guide FractiAI's behavior and ensure
coherence across all operational dimensions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

@dataclass
class TemplateConfig:
    """Configuration for Master Fractal Templates"""
    name: str
    dimensions: int = 3
    coherence_threshold: float = 0.8
    adaptation_rate: float = 0.01
    pattern_memory_size: int = 100
    objective_weights: Dict[str, float] = None

class TemplatePattern:
    """Represents a stored fractal pattern with metadata"""
    
    def __init__(self, pattern: np.ndarray, metadata: Dict):
        self.pattern = pattern
        self.metadata = metadata
        self.timestamp = np.datetime64('now')
        self.usage_count = 0
        
    def update_usage(self) -> None:
        """Track pattern usage"""
        self.usage_count += 1

class FractalTemplate(ABC):
    """Abstract base class for Master Fractal Templates"""
    
    @abstractmethod
    def validate_coherence(self, actions: List[dict]) -> float:
        """Validate actions against template objectives"""
        pass
    
    @abstractmethod
    def adapt(self, feedback: Dict) -> None:
        """Adapt template based on feedback"""
        pass

class MasterFractalTemplate(FractalTemplate):
    """Advanced implementation of Master Fractal Templates"""
    
    def __init__(self, config: TemplateConfig):
        self.config = config
        self.objectives = config.objective_weights or {
            "harmony": 0.9,
            "efficiency": 0.8,
            "sustainability": 0.7
        }
        self.patterns = []
        self.feedback_history = []
        self.coherence_metrics = {}
        
    def validate_coherence(self, actions: List[dict]) -> float:
        """Validate actions against template objectives with pattern awareness"""
        coherence_score = 0.0
        pattern_influence = self._compute_pattern_influence()
        
        for action in actions:
            # Base coherence from objectives
            if action['objective'] in self.objectives:
                base_score = self.objectives[action['objective']]
                
                # Adjust score based on pattern history
                adjusted_score = base_score * (1 + pattern_influence)
                coherence_score += adjusted_score
                
        normalized_score = coherence_score / len(actions) if actions else 0.0
        
        # Store coherence metrics
        self.coherence_metrics[np.datetime64('now')] = {
            'score': normalized_score,
            'pattern_influence': pattern_influence,
            'action_count': len(actions)
        }
        
        return normalized_score
    
    def adapt(self, feedback: Dict) -> None:
        """Adapt template based on feedback with pattern learning"""
        self.feedback_history.append(feedback)
        
        # Update objectives based on feedback
        for key, value in feedback.items():
            if key in self.objectives:
                current = self.objectives[key]
                self.objectives[key] = current + self.config.adaptation_rate * (value - current)
        
        # Extract and store patterns
        if 'pattern' in feedback:
            self._store_pattern(feedback['pattern'], feedback)
            
        # Prune old patterns if needed
        self._prune_patterns()
    
    def apply_transformation(self, data: np.ndarray) -> np.ndarray:
        """Apply template patterns to transform input data"""
        if not self.patterns:
            return data
            
        # Find most relevant pattern
        pattern = self._select_pattern(data)
        if pattern:
            pattern.update_usage()
            return self._apply_pattern(data, pattern.pattern)
            
        return data
    
    def measure_alignment(self, data: np.ndarray) -> float:
        """Measure how well data aligns with template patterns"""
        if not self.patterns:
            return 1.0
            
        alignments = []
        for pattern in self.patterns:
            alignment = np.corrcoef(data.flatten(), pattern.pattern.flatten())[0,1]
            alignments.append(abs(alignment))
            
        return float(np.mean(alignments))
    
    def _store_pattern(self, pattern: np.ndarray, metadata: Dict) -> None:
        """Store new pattern with metadata"""
        self.patterns.append(TemplatePattern(pattern, metadata))
    
    def _prune_patterns(self) -> None:
        """Remove old or unused patterns"""
        if len(self.patterns) > self.config.pattern_memory_size:
            # Sort by usage count and recency
            self.patterns.sort(key=lambda p: (p.usage_count, p.timestamp))
            # Keep only the most relevant patterns
            self.patterns = self.patterns[-self.config.pattern_memory_size:]
    
    def _select_pattern(self, data: np.ndarray) -> Optional[TemplatePattern]:
        """Select most relevant pattern for given input"""
        if not self.patterns:
            return None
            
        similarities = []
        for pattern in self.patterns:
            similarity = np.corrcoef(data.flatten(), pattern.pattern.flatten())[0,1]
            similarities.append((abs(similarity), pattern))
            
        if similarities:
            return max(similarities, key=lambda x: x[0])[1]
        return None
    
    def _apply_pattern(self, data: np.ndarray, pattern: np.ndarray) -> np.ndarray:
        """Apply pattern transformation to input data"""
        # Ensure compatible shapes
        if data.shape != pattern.shape:
            pattern = np.resize(pattern, data.shape)
        
        # Combine input with pattern
        return np.tanh(data + pattern * self.config.adaptation_rate)
    
    def _compute_pattern_influence(self) -> float:
        """Compute current influence of stored patterns"""
        if not self.patterns:
            return 0.0
            
        recent_patterns = sorted(self.patterns, key=lambda p: p.timestamp)[-10:]
        influences = [p.usage_count / (len(self.patterns) + 1) for p in recent_patterns]
        return float(np.mean(influences)) 