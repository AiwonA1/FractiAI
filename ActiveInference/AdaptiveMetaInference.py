"""
AdaptiveMetaInference.py

Implements adaptive meta-level active inference for FractiAI, enabling dynamic 
adaptation and optimization through meta-learning and fractal patterns.
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
class AdaptiveConfig:
    """Configuration for adaptive meta-inference"""
    adaptation_levels: int = 3
    memory_size: int = 1000
    learning_rate: float = 0.01
    adaptation_rate: float = 0.1
    fractal_depth: int = 3
    stability_threshold: float = 0.7

class AdaptiveStrategy:
    """Represents an adaptive learning strategy"""
    
    def __init__(self, strategy_id: str, dim: int):
        self.strategy_id = strategy_id
        self.parameters = np.random.randn(dim)
        self.adaptation_history = []
        self.performance_history = []
        self.meta_parameters = {
            'learning_rate': 0.01,
            'adaptation_strength': 0.5,
            'stability_weight': 0.7
        }
        
    def adapt(self, feedback: float, context: Optional[np.ndarray] = None) -> None:
        """Adapt strategy based on feedback"""
        # Update parameters
        if context is not None:
            gradient = self._compute_gradient(context)
            self.parameters += self.meta_parameters['learning_rate'] * feedback * gradient
            
        # Store adaptation
        self.adaptation_history.append({
            'feedback': feedback,
            'context': context,
            'parameters': self.parameters.copy()
        })
        
        # Update meta-parameters
        self._update_meta_parameters(feedback)
        
    def _compute_gradient(self, context: np.ndarray) -> np.ndarray:
        """Compute adaptation gradient"""
        # Simplified gradient computation
        return context * self.meta_parameters['adaptation_strength']
        
    def _update_meta_parameters(self, feedback: float) -> None:
        """Update meta-parameters based on feedback"""
        # Update learning rate
        self.meta_parameters['learning_rate'] *= (1 + feedback * 0.1)
        
        # Update adaptation strength
        self.meta_parameters['adaptation_strength'] *= (1 + feedback * 0.05)
        
        # Update stability weight
        stability = self._compute_stability()
        self.meta_parameters['stability_weight'] *= (1 + stability * 0.01)
        
    def _compute_stability(self) -> float:
        """Compute strategy stability"""
        if len(self.adaptation_history) < 2:
            return 1.0
            
        parameters = [a['parameters'] for a in self.adaptation_history]
        diffs = np.diff(parameters, axis=0)
        return float(1.0 - np.mean(np.abs(diffs)))
        
    def get_metrics(self) -> Dict[str, float]:
        """Get strategy metrics"""
        return {
            'stability': self._compute_stability(),
            'adaptation_rate': float(np.mean([a['feedback'] 
                                            for a in self.adaptation_history])),
            'meta_parameters': self.meta_parameters
        }

class AdaptationLevel:
    """Represents a level in the adaptation hierarchy"""
    
    def __init__(self, level_id: int, config: AdaptiveConfig):
        self.level_id = level_id
        self.config = config
        self.strategies: Dict[str, AdaptiveStrategy] = {}
        self.adaptation_history = []
        
    def adapt(self, feedback: Dict[str, float], 
             context: Optional[np.ndarray] = None) -> None:
        """Adapt level strategies"""
        for strategy_id, strategy_feedback in feedback.items():
            if strategy_id in self.strategies:
                self.strategies[strategy_id].adapt(strategy_feedback, context)
                
        # Store adaptation
        self.adaptation_history.append({
            'feedback': feedback,
            'context': context
        })
        
    def add_strategy(self, strategy_id: str, dim: int) -> None:
        """Add new strategy to level"""
        self.strategies[strategy_id] = AdaptiveStrategy(strategy_id, dim)
        
    def get_best_strategy(self, context: Optional[np.ndarray] = None) -> Tuple[str, AdaptiveStrategy]:
        """Get best performing strategy"""
        if not self.strategies:
            return None, None
            
        # Compute strategy scores
        scores = {}
        for strategy_id, strategy in self.strategies.items():
            performance = np.mean([a['feedback'] for a in strategy.adaptation_history]) \
                         if strategy.adaptation_history else 0
            stability = strategy._compute_stability()
            scores[strategy_id] = performance * stability
            
        # Select best strategy
        best_id = max(scores.items(), key=lambda x: x[1])[0]
        return best_id, self.strategies[best_id]
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get level metrics"""
        return {
            'num_strategies': len(self.strategies),
            'strategy_metrics': {
                strategy_id: strategy.get_metrics()
                for strategy_id, strategy in self.strategies.items()
            },
            'adaptation_stability': self._compute_adaptation_stability()
        }
        
    def _compute_adaptation_stability(self) -> float:
        """Compute stability of adaptations"""
        if len(self.adaptation_history) < 2:
            return 1.0
            
        feedbacks = [
            np.mean(list(a['feedback'].values()))
            for a in self.adaptation_history
        ]
        return float(1.0 - np.std(feedbacks) / (np.mean(np.abs(feedbacks)) + 1e-7))

class AdaptiveMetaInference:
    """Adaptive meta-inference system"""
    
    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.levels: List[AdaptationLevel] = []
        self.meta_memory = deque(maxlen=config.memory_size)
        self.initialize_levels()
        
    def initialize_levels(self) -> None:
        """Initialize adaptation levels"""
        for i in range(self.config.adaptation_levels):
            level = AdaptationLevel(i, self.config)
            # Add initial strategies
            for j in range(3):  # Start with 3 strategies per level
                level.add_strategy(f"STRATEGY_{i}_{j}", dim=32)
            self.levels.append(level)
            
    def adapt(self, feedback: Dict[str, float], 
             context: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Adapt system based on feedback"""
        level_results = []
        
        # Adapt each level
        for level in self.levels:
            level.adapt(feedback, context)
            
            # Get best strategy
            strategy_id, strategy = level.get_best_strategy(context)
            
            if strategy is not None:
                level_results.append({
                    'level_id': level.level_id,
                    'strategy_id': strategy_id,
                    'metrics': strategy.get_metrics()
                })
                
        # Store adaptation
        self._store_adaptation(feedback, context, level_results)
        
        return {
            'level_results': level_results,
            'metrics': self._compute_metrics()
        }
        
    def _store_adaptation(self, feedback: Dict[str, float],
                         context: Optional[np.ndarray],
                         results: List[Dict[str, Any]]) -> None:
        """Store adaptation experience"""
        experience = {
            'feedback': feedback,
            'context': context,
            'results': results,
            'timestamp': len(self.meta_memory)
        }
        self.meta_memory.append(experience)
        
    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute comprehensive metrics"""
        return {
            'level_metrics': self._compute_level_metrics(),
            'adaptation_metrics': self._compute_adaptation_metrics(),
            'meta_metrics': self._compute_meta_metrics()
        }
        
    def _compute_level_metrics(self) -> List[Dict[str, Any]]:
        """Compute metrics for each level"""
        return [level.get_metrics() for level in self.levels]
        
    def _compute_adaptation_metrics(self) -> Dict[str, float]:
        """Compute adaptation-related metrics"""
        if not self.meta_memory:
            return {}
            
        recent = list(self.meta_memory)[-10:]
        feedbacks = [
            np.mean(list(exp['feedback'].values()))
            for exp in recent
        ]
        
        return {
            'adaptation_rate': float(np.mean(np.diff(feedbacks))) if len(feedbacks) > 1 else 0.0,
            'adaptation_stability': float(1.0 - np.std(feedbacks)) if feedbacks else 0.0,
            'adaptation_progress': self._compute_adaptation_progress()
        }
        
    def _compute_meta_metrics(self) -> Dict[str, float]:
        """Compute meta-learning metrics"""
        return {
            'meta_stability': self._compute_meta_stability(),
            'strategy_diversity': self._compute_strategy_diversity(),
            'learning_efficiency': self._compute_learning_efficiency()
        }
        
    def _compute_adaptation_progress(self) -> float:
        """Compute overall adaptation progress"""
        if len(self.meta_memory) < 2:
            return 0.0
            
        # Compare recent performance with initial performance
        recent_perf = np.mean([
            np.mean(list(exp['feedback'].values()))
            for exp in list(self.meta_memory)[-10:]
        ])
        
        initial_perf = np.mean([
            np.mean(list(exp['feedback'].values()))
            for exp in list(self.meta_memory)[:10]
        ])
        
        return float(recent_perf - initial_perf)
        
    def _compute_meta_stability(self) -> float:
        """Compute stability of meta-learning"""
        stabilities = []
        for level in self.levels:
            level_stability = level._compute_adaptation_stability()
            stabilities.append(level_stability)
            
        return float(np.mean(stabilities))
        
    def _compute_strategy_diversity(self) -> float:
        """Compute diversity of strategies"""
        strategy_params = []
        for level in self.levels:
            for strategy in level.strategies.values():
                strategy_params.append(strategy.parameters)
                
        if not strategy_params:
            return 0.0
            
        # Use parameter variance as diversity measure
        return float(np.mean([np.var(params) for params in strategy_params]))
        
    def _compute_learning_efficiency(self) -> float:
        """Compute efficiency of learning process"""
        if not self.meta_memory:
            return 0.0
            
        # Compare learning progress with adaptation cost
        progress = self._compute_adaptation_progress()
        adaptations = len(self.meta_memory)
        
        return float(progress / (adaptations + 1e-7)) 