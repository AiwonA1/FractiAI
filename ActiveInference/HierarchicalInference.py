"""
HierarchicalInference.py

Implements hierarchical active inference for FractiAI, enabling multi-level
predictive processing and belief propagation through fractal patterns.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy.stats import entropy
from scipy.special import softmax

logger = logging.getLogger(__name__)

@dataclass
class HierarchyConfig:
    """Configuration for hierarchical inference"""
    num_levels: int = 5
    level_dims: List[int] = (256, 128, 64, 32, 16)
    temporal_horizon: int = 10
    learning_rate: float = 0.01
    precision_init: float = 1.0
    fractal_depth: int = 3

class InferenceLevel:
    """Single level in hierarchical inference system"""
    
    def __init__(self, level_id: int, input_dim: int, output_dim: int, config: HierarchyConfig):
        self.level_id = level_id
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        
        # Initialize generative model
        self.generative_weights = np.random.randn(output_dim, input_dim) * 0.1
        self.recognition_weights = np.random.randn(input_dim, output_dim) * 0.1
        
        # Initialize state and predictions
        self.state = np.zeros(output_dim)
        self.prediction = np.zeros(input_dim)
        self.prediction_error = np.zeros(input_dim)
        
        # Initialize precision matrices
        self.state_precision = np.eye(output_dim) * config.precision_init
        self.prediction_precision = np.eye(input_dim) * config.precision_init
        
        # Store history
        self.state_history = []
        self.error_history = []
        
    def predict_down(self) -> np.ndarray:
        """Generate prediction for level below"""
        self.prediction = np.dot(self.generative_weights.T, self.state)
        return self.prediction
        
    def predict_up(self, input_data: np.ndarray) -> np.ndarray:
        """Generate prediction for level above"""
        return np.dot(self.recognition_weights.T, input_data)
        
    def update_state(self, bottom_up: np.ndarray, 
                    top_down: Optional[np.ndarray] = None) -> None:
        """Update level state based on bottom-up and top-down signals"""
        # Compute prediction errors
        bottom_up_error = bottom_up - self.prediction
        self.prediction_error = bottom_up_error
        
        # Update state
        state_update = np.dot(self.recognition_weights, bottom_up_error)
        if top_down is not None:
            top_down_error = self.state - top_down
            state_update -= np.dot(self.state_precision, top_down_error)
            
        self.state += self.config.learning_rate * state_update
        
        # Store history
        self.state_history.append(self.state.copy())
        self.error_history.append(np.mean(np.abs(self.prediction_error)))
        
    def update_weights(self, learning_rate: Optional[float] = None) -> None:
        """Update weights based on prediction errors"""
        if learning_rate is None:
            learning_rate = self.config.learning_rate
            
        # Update generative weights
        self.generative_weights += learning_rate * np.outer(
            self.state,
            self.prediction_error
        )
        
        # Update recognition weights
        self.recognition_weights += learning_rate * np.outer(
            self.prediction_error,
            self.state
        )
        
    def update_precision(self, learning_rate: Optional[float] = None) -> None:
        """Update precision matrices based on errors"""
        if learning_rate is None:
            learning_rate = self.config.learning_rate
            
        # Update state precision
        state_error = np.outer(self.state, self.state)
        self.state_precision += learning_rate * (
            state_error - self.state_precision
        )
        
        # Update prediction precision
        pred_error = np.outer(self.prediction_error, self.prediction_error)
        self.prediction_precision += learning_rate * (
            pred_error - self.prediction_precision
        )

class HierarchicalInference:
    """Hierarchical active inference system with fractal patterns"""
    
    def __init__(self, config: HierarchyConfig):
        self.config = config
        self.levels: List[InferenceLevel] = []
        self.initialize_hierarchy()
        
    def initialize_hierarchy(self) -> None:
        """Initialize hierarchical structure"""
        for i in range(self.config.num_levels):
            input_dim = self.config.level_dims[i]
            output_dim = self.config.level_dims[i+1] if i < len(self.config.level_dims)-1 else self.config.level_dims[-1]
            
            level = InferenceLevel(i, input_dim, output_dim, self.config)
            self.levels.append(level)
            
    def process_input(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Process input through hierarchy"""
        # Bottom-up pass
        current_input = input_data
        for level in self.levels:
            prediction = level.predict_up(current_input)
            level.update_state(current_input)
            current_input = prediction
            
        # Top-down pass
        for level in reversed(self.levels[:-1]):
            top_down = level.predict_down()
            level.update_state(current_input, top_down)
            
        # Update weights and precision
        self._update_parameters()
        
        return self._get_processing_results()
        
    def _update_parameters(self) -> None:
        """Update weights and precision across hierarchy"""
        for level in self.levels:
            level.update_weights()
            level.update_precision()
            
    def _get_processing_results(self) -> Dict[str, Any]:
        """Get processing results and metrics"""
        return {
            'states': [level.state.copy() for level in self.levels],
            'predictions': [level.prediction.copy() for level in self.levels],
            'errors': [level.prediction_error.copy() for level in self.levels],
            'metrics': self._compute_metrics()
        }
        
    def _compute_metrics(self) -> Dict[str, float]:
        """Compute hierarchical inference metrics"""
        metrics = {
            'mean_error': np.mean([np.mean(np.abs(level.prediction_error)) 
                                 for level in self.levels]),
            'error_gradient': self._compute_error_gradient(),
            'state_complexity': self._compute_state_complexity(),
            'prediction_accuracy': self._compute_prediction_accuracy()
        }
        
        return metrics
        
    def _compute_error_gradient(self) -> float:
        """Compute error gradient across hierarchy"""
        errors = [np.mean(np.abs(level.prediction_error)) 
                 for level in self.levels]
        return float(np.mean(np.diff(errors)))
        
    def _compute_state_complexity(self) -> float:
        """Compute complexity of hierarchical state representation"""
        complexities = []
        for level in self.levels:
            state_dist = softmax(level.state)
            complexities.append(entropy(state_dist))
        return float(np.mean(complexities))
        
    def _compute_prediction_accuracy(self) -> float:
        """Compute overall prediction accuracy"""
        accuracies = []
        for level in self.levels:
            if len(level.error_history) > 1:
                recent_errors = level.error_history[-10:]
                accuracy = 1.0 - np.mean(recent_errors)
                accuracies.append(accuracy)
        return float(np.mean(accuracies))
        
    def get_level_metrics(self, level_id: int) -> Dict[str, Any]:
        """Get detailed metrics for specific level"""
        if level_id >= len(self.levels):
            return {}
            
        level = self.levels[level_id]
        return {
            'mean_error': float(np.mean(np.abs(level.prediction_error))),
            'state_entropy': float(entropy(softmax(level.state))),
            'weight_sparsity': float(np.mean(np.abs(level.generative_weights) < 0.01)),
            'precision_strength': float(np.mean(np.diag(level.state_precision)))
        }
        
    def reset_states(self) -> None:
        """Reset hierarchical states"""
        for level in self.levels:
            level.state = np.zeros_like(level.state)
            level.prediction = np.zeros_like(level.prediction)
            level.prediction_error = np.zeros_like(level.prediction_error) 