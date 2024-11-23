"""
HierarchicalMetaInference.py

Implements hierarchical meta-level active inference for FractiAI, enabling multi-level
meta-learning and adaptive inference strategies through fractal patterns.
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
class HierarchicalConfig:
    """Configuration for hierarchical meta-inference"""
    num_levels: int = 5
    level_dims: List[int] = (256, 128, 64, 32, 16)
    meta_horizon: int = 10
    adaptation_rate: float = 0.01
    fractal_depth: int = 3
    memory_size: int = 1000

class MetaLevel:
    """Single level in hierarchical meta-learning system"""
    
    def __init__(self, level_id: int, input_dim: int, output_dim: int):
        self.level_id = level_id
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize meta-parameters
        self.meta_params = {
            'learning_rate': 0.01,
            'adaptation_rate': 0.1,
            'complexity_penalty': 0.01
        }
        
        # Initialize strategies and memory
        self.strategies = []
        self.memory = deque(maxlen=1000)
        self.performance_history = []
        
    def adapt_meta_params(self, feedback: float) -> None:
        """Adapt meta-parameters based on feedback"""
        # Update learning rate
        self.meta_params['learning_rate'] *= (1 + feedback * 0.1)
        
        # Update adaptation rate
        self.meta_params['adaptation_rate'] *= (1 + feedback * 0.05)
        
        # Update complexity penalty
        complexity = self._compute_complexity()
        self.meta_params['complexity_penalty'] *= (1 + complexity * 0.01)
        
    def store_experience(self, experience: Dict[str, Any]) -> None:
        """Store learning experience"""
        self.memory.append(experience)
        
        # Update performance history
        if 'performance' in experience:
            self.performance_history.append(experience['performance'])
            
    def get_meta_strategy(self, context: np.ndarray) -> Dict[str, Any]:
        """Get meta-strategy for given context"""
        if not self.memory:
            return self._get_default_strategy()
            
        # Find similar contexts
        similar_experiences = self._find_similar_experiences(context)
        
        # Generate meta-strategy
        return self._generate_meta_strategy(similar_experiences)
        
    def _find_similar_experiences(self, 
                                context: np.ndarray) -> List[Dict[str, Any]]:
        """Find experiences with similar contexts"""
        similarities = []
        for exp in self.memory:
            if 'context' in exp:
                similarity = self._compute_similarity(context, exp['context'])
                similarities.append((similarity, exp))
                
        # Get top 5 most similar
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in similarities[:5]]
        
    def _generate_meta_strategy(self, 
                              experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate meta-strategy from similar experiences"""
        if not experiences:
            return self._get_default_strategy()
            
        # Aggregate successful strategies
        strategies = [exp['strategy'] for exp in experiences 
                     if exp.get('performance', 0) > 0.5]
        
        if not strategies:
            return self._get_default_strategy()
            
        # Combine strategies
        return {
            'learning_rate': np.mean([s.get('learning_rate', 0.01) 
                                    for s in strategies]),
            'adaptation_rate': np.mean([s.get('adaptation_rate', 0.1) 
                                      for s in strategies]),
            'complexity_penalty': np.mean([s.get('complexity_penalty', 0.01) 
                                         for s in strategies])
        }
        
    def _get_default_strategy(self) -> Dict[str, Any]:
        """Get default meta-strategy"""
        return self.meta_params.copy()
        
    def _compute_similarity(self, context1: np.ndarray, 
                          context2: np.ndarray) -> float:
        """Compute context similarity"""
        return float(np.abs(np.corrcoef(context1, context2)[0,1]))
        
    def _compute_complexity(self) -> float:
        """Compute current strategy complexity"""
        if not self.memory:
            return 0.0
            
        strategies = [exp['strategy'] for exp in self.memory 
                     if 'strategy' in exp]
        if not strategies:
            return 0.0
            
        # Use strategy parameter variance as complexity measure
        complexities = []
        for param in ['learning_rate', 'adaptation_rate', 'complexity_penalty']:
            values = [s.get(param, 0) for s in strategies]
            complexities.append(np.var(values))
            
        return float(np.mean(complexities))

class HierarchicalMetaInference:
    """Hierarchical meta-learning system with fractal patterns"""
    
    def __init__(self, config: HierarchicalConfig):
        self.config = config
        self.levels: List[MetaLevel] = []
        self.meta_memory = deque(maxlen=config.memory_size)
        self.initialize_hierarchy()
        
    def initialize_hierarchy(self) -> None:
        """Initialize hierarchical structure"""
        for i in range(self.config.num_levels):
            input_dim = self.config.level_dims[i]
            output_dim = self.config.level_dims[i+1] if i < len(self.config.level_dims)-1 else self.config.level_dims[-1]
            
            level = MetaLevel(i, input_dim, output_dim)
            self.levels.append(level)
            
    def process(self, input_data: np.ndarray, 
               context: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Process input through hierarchy with meta-learning"""
        if context is None:
            context = input_data
            
        # Get meta-strategies for each level
        meta_strategies = []
        for level in self.levels:
            strategy = level.get_meta_strategy(context)
            meta_strategies.append(strategy)
            
        # Process through levels
        current_input = input_data
        level_outputs = []
        
        for level, strategy in zip(self.levels, meta_strategies):
            # Apply meta-strategy
            output = self._process_level(level, current_input, strategy)
            level_outputs.append(output)
            current_input = output['output']
            
        # Store meta-experience
        self._store_meta_experience(context, meta_strategies, level_outputs)
        
        return {
            'final_output': current_input,
            'level_outputs': level_outputs,
            'meta_strategies': meta_strategies,
            'metrics': self._compute_metrics()
        }
        
    def _process_level(self, level: MetaLevel, 
                      input_data: np.ndarray,
                      strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through single level"""
        # Apply strategy parameters
        learning_rate = strategy['learning_rate']
        adaptation_rate = strategy['adaptation_rate']
        
        # Transform input
        output = np.tanh(input_data * adaptation_rate)
        
        # Compute performance
        performance = self._compute_performance(input_data, output)
        
        # Store experience
        experience = {
            'context': input_data,
            'strategy': strategy,
            'output': output,
            'performance': performance
        }
        level.store_experience(experience)
        
        # Adapt meta-parameters
        level.adapt_meta_params(performance)
        
        return experience
        
    def _compute_performance(self, input_data: np.ndarray, 
                           output: np.ndarray) -> float:
        """Compute performance metric"""
        # Use prediction error as performance measure
        error = np.mean((input_data - output) ** 2)
        return float(1.0 - error)
        
    def _store_meta_experience(self, context: np.ndarray,
                             strategies: List[Dict[str, Any]],
                             outputs: List[Dict[str, Any]]) -> None:
        """Store meta-learning experience"""
        experience = {
            'context': context,
            'strategies': strategies,
            'outputs': outputs,
            'performance': np.mean([o['performance'] for o in outputs])
        }
        self.meta_memory.append(experience)
        
    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute hierarchical meta-learning metrics"""
        return {
            'level_metrics': self._compute_level_metrics(),
            'meta_metrics': self._compute_meta_metrics(),
            'hierarchy_metrics': self._compute_hierarchy_metrics()
        }
        
    def _compute_level_metrics(self) -> List[Dict[str, float]]:
        """Compute metrics for each level"""
        metrics = []
        for level in self.levels:
            if level.performance_history:
                metrics.append({
                    'mean_performance': float(np.mean(level.performance_history)),
                    'learning_progress': float(np.mean(np.diff(level.performance_history))),
                    'strategy_complexity': level._compute_complexity()
                })
        return metrics
        
    def _compute_meta_metrics(self) -> Dict[str, float]:
        """Compute meta-learning metrics"""
        if not self.meta_memory:
            return {}
            
        performances = [exp['performance'] for exp in self.meta_memory]
        return {
            'meta_performance': float(np.mean(performances)),
            'meta_stability': float(1.0 - np.std(performances)),
            'meta_progress': float(np.mean(np.diff(performances)))
        }
        
    def _compute_hierarchy_metrics(self) -> Dict[str, float]:
        """Compute hierarchical structure metrics"""
        level_performances = [
            level.performance_history for level in self.levels
            if level.performance_history
        ]
        
        if not level_performances:
            return {}
            
        # Compute hierarchical gradients
        gradients = []
        for i in range(len(level_performances)-1):
            gradient = np.mean(level_performances[i]) - np.mean(level_performances[i+1])
            gradients.append(gradient)
            
        return {
            'hierarchy_depth': len(self.levels),
            'level_gradient': float(np.mean(gradients)) if gradients else 0.0,
            'hierarchy_coherence': self._compute_coherence()
        }
        
    def _compute_coherence(self) -> float:
        """Compute coherence across hierarchy"""
        if not self.meta_memory:
            return 0.0
            
        # Compare strategies across levels
        coherence_scores = []
        for exp in self.meta_memory:
            strategies = exp['strategies']
            for i in range(len(strategies)-1):
                similarity = self._compute_strategy_similarity(
                    strategies[i],
                    strategies[i+1]
                )
                coherence_scores.append(similarity)
                
        return float(np.mean(coherence_scores)) if coherence_scores else 0.0
        
    def _compute_strategy_similarity(self, strategy1: Dict[str, Any],
                                   strategy2: Dict[str, Any]) -> float:
        """Compute similarity between strategies"""
        params1 = np.array([strategy1.get(p, 0) for p in 
                          ['learning_rate', 'adaptation_rate', 'complexity_penalty']])
        params2 = np.array([strategy2.get(p, 0) for p in 
                          ['learning_rate', 'adaptation_rate', 'complexity_penalty']])
        
        return float(np.abs(np.corrcoef(params1, params2)[0,1]))
</rewritten_file> 