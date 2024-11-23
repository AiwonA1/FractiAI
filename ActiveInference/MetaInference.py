"""
MetaInference.py

Implements meta-level active inference for FractiAI, enabling learning-to-learn
and adaptive inference strategies through fractal patterns.
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
class MetaConfig:
    """Configuration for meta-inference system"""
    num_strategies: int = 10
    strategy_dim: int = 32
    adaptation_rate: float = 0.01
    memory_size: int = 1000
    meta_horizon: int = 5
    fractal_depth: int = 3

class InferenceStrategy:
    """Represents a learned inference strategy"""
    
    def __init__(self, strategy_id: str, dim: int):
        self.strategy_id = strategy_id
        self.parameters = np.random.randn(dim)
        self.performance_history = []
        self.adaptation_history = []
        self.meta_parameters = {}
        
    def apply(self, state: np.ndarray) -> np.ndarray:
        """Apply inference strategy to state"""
        # Transform state using strategy parameters
        transformed = np.tanh(np.dot(state, self.parameters))
        
        # Apply meta-parameters if available
        if self.meta_parameters:
            for param_name, value in self.meta_parameters.items():
                if param_name == 'scale':
                    transformed *= value
                elif param_name == 'bias':
                    transformed += value
                    
        return transformed
        
    def adapt(self, feedback: float, learning_rate: float) -> None:
        """Adapt strategy based on feedback"""
        # Update parameters using feedback
        gradient = np.random.randn(*self.parameters.shape)  # Simplified gradient
        self.parameters += learning_rate * feedback * gradient
        
        # Store adaptation
        self.adaptation_history.append({
            'feedback': feedback,
            'change': np.mean(np.abs(learning_rate * feedback * gradient))
        })
        
    def update_meta(self, meta_params: Dict[str, float]) -> None:
        """Update meta-parameters"""
        self.meta_parameters.update(meta_params)
        
    def get_metrics(self) -> Dict[str, float]:
        """Get strategy metrics"""
        if not self.adaptation_history:
            return {}
            
        adaptations = [a['change'] for a in self.adaptation_history]
        feedbacks = [a['feedback'] for a in self.adaptation_history]
        
        return {
            'mean_feedback': float(np.mean(feedbacks)),
            'adaptation_rate': float(np.mean(adaptations)),
            'stability': float(1.0 - np.std(feedbacks) / (np.mean(np.abs(feedbacks)) + 1e-7))
        }

class MetaLearner:
    """Meta-learning system for inference strategies"""
    
    def __init__(self, config: MetaConfig):
        self.config = config
        self.strategies: Dict[str, InferenceStrategy] = {}
        self.strategy_history = deque(maxlen=config.memory_size)
        self.meta_knowledge = {}
        self.initialize_strategies()
        
    def initialize_strategies(self) -> None:
        """Initialize inference strategies"""
        for i in range(self.config.num_strategies):
            strategy_id = f"STRATEGY_{i}"
            self.strategies[strategy_id] = InferenceStrategy(
                strategy_id,
                self.config.strategy_dim
            )
            
    def select_strategy(self, state: np.ndarray) -> Tuple[str, np.ndarray]:
        """Select best strategy for current state"""
        # Compute strategy scores
        scores = {}
        for strategy_id, strategy in self.strategies.items():
            # Get strategy performance
            performance = np.mean(strategy.performance_history) if strategy.performance_history else 0
            
            # Get strategy adaptation
            adaptation = strategy.get_metrics().get('adaptation_rate', 0)
            
            # Combine scores
            scores[strategy_id] = performance + adaptation * self.config.adaptation_rate
            
        # Select strategy
        selected_id = max(scores.items(), key=lambda x: x[1])[0]
        selected_strategy = self.strategies[selected_id]
        
        # Apply strategy
        output = selected_strategy.apply(state)
        
        # Store selection
        self.strategy_history.append({
            'strategy_id': selected_id,
            'state': state,
            'output': output,
            'score': scores[selected_id]
        })
        
        return selected_id, output
        
    def update_strategy(self, strategy_id: str, 
                       feedback: float) -> None:
        """Update strategy based on feedback"""
        if strategy_id in self.strategies:
            strategy = self.strategies[strategy_id]
            
            # Store performance
            strategy.performance_history.append(feedback)
            
            # Adapt strategy
            strategy.adapt(feedback, self.config.adaptation_rate)
            
            # Update meta-knowledge
            self._update_meta_knowledge(strategy_id, feedback)
            
    def _update_meta_knowledge(self, strategy_id: str, 
                             feedback: float) -> None:
        """Update meta-level knowledge"""
        if strategy_id not in self.meta_knowledge:
            self.meta_knowledge[strategy_id] = {
                'feedback_history': [],
                'context_history': []
            }
            
        # Store feedback
        self.meta_knowledge[strategy_id]['feedback_history'].append(feedback)
        
        # Store context if available
        if self.strategy_history:
            last_context = self.strategy_history[-1]['state']
            self.meta_knowledge[strategy_id]['context_history'].append(last_context)
            
        # Update meta-parameters
        self._adapt_meta_parameters(strategy_id)
        
    def _adapt_meta_parameters(self, strategy_id: str) -> None:
        """Adapt meta-parameters based on accumulated knowledge"""
        if strategy_id not in self.meta_knowledge:
            return
            
        knowledge = self.meta_knowledge[strategy_id]
        
        if len(knowledge['feedback_history']) < 2:
            return
            
        # Compute meta-parameters
        feedbacks = np.array(knowledge['feedback_history'])
        
        meta_params = {
            'scale': float(1.0 + np.mean(feedbacks)),
            'bias': float(np.mean(feedbacks) * self.config.adaptation_rate)
        }
        
        # Update strategy meta-parameters
        self.strategies[strategy_id].update_meta(meta_params)

class MetaInference:
    """Meta-level active inference system with fractal patterns"""
    
    def __init__(self, config: MetaConfig):
        self.config = config
        self.meta_learner = MetaLearner(config)
        self.state_history = []
        self.meta_metrics = {}
        
    def process(self, state: np.ndarray) -> Dict[str, Any]:
        """Process state through meta-inference"""
        # Store state
        self.state_history.append(state)
        
        # Select and apply strategy
        strategy_id, output = self.meta_learner.select_strategy(state)
        
        # Compute feedback
        feedback = self._compute_feedback(state, output)
        
        # Update strategy
        self.meta_learner.update_strategy(strategy_id, feedback)
        
        # Update metrics
        self._update_metrics()
        
        return {
            'output': output,
            'strategy': strategy_id,
            'feedback': feedback,
            'metrics': self.meta_metrics
        }
        
    def _compute_feedback(self, state: np.ndarray, 
                         output: np.ndarray) -> float:
        """Compute feedback for strategy"""
        # Use prediction error as feedback
        error = np.mean((state - output) ** 2)
        return float(1.0 - error)
        
    def _update_metrics(self) -> None:
        """Update meta-inference metrics"""
        self.meta_metrics = {
            'strategy_metrics': {
                strategy_id: strategy.get_metrics()
                for strategy_id, strategy in self.meta_learner.strategies.items()
            },
            'learning_progress': self._compute_learning_progress(),
            'adaptation_efficiency': self._compute_adaptation_efficiency()
        }
        
    def _compute_learning_progress(self) -> float:
        """Compute overall learning progress"""
        if not self.meta_learner.strategy_history:
            return 0.0
            
        scores = [s['score'] for s in self.meta_learner.strategy_history]
        if len(scores) < 2:
            return 0.0
            
        return float(np.mean(np.diff(scores)))
        
    def _compute_adaptation_efficiency(self) -> float:
        """Compute efficiency of adaptation"""
        efficiencies = []
        
        for strategy in self.meta_learner.strategies.values():
            if strategy.adaptation_history:
                changes = [a['change'] for a in strategy.adaptation_history]
                feedbacks = [a['feedback'] for a in strategy.adaptation_history]
                
                if changes and feedbacks:
                    efficiency = np.mean(feedbacks) / (np.mean(changes) + 1e-7)
                    efficiencies.append(efficiency)
                    
        return float(np.mean(efficiencies)) if efficiencies else 0.0 