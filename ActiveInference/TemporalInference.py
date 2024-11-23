"""
TemporalInference.py

Implements temporal active inference for FractiAI, enabling time-series prediction
and causal inference through fractal patterns and temporal dependencies.
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
    """Configuration for temporal inference system"""
    sequence_length: int = 100
    prediction_horizon: int = 10
    hidden_dim: int = 64
    num_layers: int = 3
    learning_rate: float = 0.01
    fractal_depth: int = 3

class TemporalCell:
    """Recurrent cell with fractal temporal dependencies"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_ih = np.random.randn(hidden_dim, input_dim) * scale
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b_h = np.zeros(hidden_dim)
        
        # Initialize temporal dependencies
        self.temporal_weights = np.eye(hidden_dim)  # Identity initially
        self.temporal_bias = np.zeros(hidden_dim)
        
    def forward(self, x: np.ndarray, h: np.ndarray, 
               temporal_context: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass with temporal context"""
        # Regular hidden state update
        h_new = np.tanh(
            np.dot(self.W_ih, x) + 
            np.dot(self.W_hh, h) + 
            self.b_h
        )
        
        # Apply temporal dependencies if context provided
        if temporal_context is not None:
            temporal_influence = np.dot(self.temporal_weights, temporal_context)
            h_new += np.tanh(temporal_influence + self.temporal_bias)
            
        return h_new
        
    def update_temporal_weights(self, error: np.ndarray, 
                              learning_rate: float) -> None:
        """Update temporal dependency weights"""
        # Simple gradient update
        self.temporal_weights -= learning_rate * np.outer(error, error)
        self.temporal_bias -= learning_rate * error

class TemporalLayer:
    """Layer of temporal cells with fractal connectivity"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_cells: int):
        self.cells = [TemporalCell(input_dim, hidden_dim) 
                     for _ in range(num_cells)]
        self.hidden_states = [np.zeros(hidden_dim) 
                            for _ in range(num_cells)]
        self.temporal_memory = deque(maxlen=100)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through temporal layer"""
        outputs = []
        
        # Get temporal context from memory
        temporal_context = self._get_temporal_context()
        
        # Process through cells
        for i, cell in enumerate(self.cells):
            h_new = cell.forward(x, self.hidden_states[i], temporal_context)
            self.hidden_states[i] = h_new
            outputs.append(h_new)
            
        # Store in temporal memory
        self.temporal_memory.append(np.mean(outputs, axis=0))
        
        return np.mean(outputs, axis=0)
        
    def _get_temporal_context(self) -> Optional[np.ndarray]:
        """Get temporal context from memory"""
        if len(self.temporal_memory) < 2:
            return None
            
        recent_states = list(self.temporal_memory)[-10:]
        return np.mean(recent_states, axis=0)

class TemporalInference:
    """Temporal active inference system with fractal patterns"""
    
    def __init__(self, config: TemporalConfig):
        self.config = config
        self.layers: List[TemporalLayer] = []
        self.predictions = deque(maxlen=config.sequence_length)
        self.errors = deque(maxlen=config.sequence_length)
        self.initialize_network()
        
    def initialize_network(self) -> None:
        """Initialize temporal network"""
        input_dim = self.config.hidden_dim
        
        for _ in range(self.config.num_layers):
            layer = TemporalLayer(
                input_dim,
                self.config.hidden_dim,
                num_cells=3  # Use 3 cells per layer
            )
            self.layers.append(layer)
            
    def process_sequence(self, sequence: np.ndarray) -> Dict[str, Any]:
        """Process temporal sequence"""
        outputs = []
        predictions = []
        
        # Process sequence
        for t in range(len(sequence)):
            # Current input
            x_t = sequence[t]
            
            # Forward pass through layers
            h_t = x_t
            for layer in self.layers:
                h_t = layer.forward(h_t)
            outputs.append(h_t)
            
            # Generate prediction for next timestep
            if t < len(sequence) - 1:
                pred_t = self._predict_next(h_t)
                predictions.append(pred_t)
                
                # Compute prediction error
                error = sequence[t+1] - pred_t
                self.errors.append(error)
                
                # Update temporal weights
                self._update_temporal_weights(error)
                
        # Store predictions
        self.predictions.extend(predictions)
        
        return {
            'outputs': np.array(outputs),
            'predictions': np.array(predictions),
            'metrics': self._compute_metrics()
        }
        
    def _predict_next(self, current_state: np.ndarray) -> np.ndarray:
        """Predict next timestep"""
        # Use temporal context for prediction
        temporal_context = self._get_temporal_context()
        
        if temporal_context is not None:
            prediction = np.tanh(
                current_state + 
                0.5 * temporal_context  # Weight temporal influence
            )
        else:
            prediction = current_state
            
        return prediction
        
    def _get_temporal_context(self) -> Optional[np.ndarray]:
        """Get temporal context from predictions"""
        if len(self.predictions) < 2:
            return None
            
        recent_predictions = list(self.predictions)[-10:]
        return np.mean(recent_predictions, axis=0)
        
    def _update_temporal_weights(self, error: np.ndarray) -> None:
        """Update temporal weights in all layers"""
        for layer in self.layers:
            for cell in layer.cells:
                cell.update_temporal_weights(
                    error,
                    self.config.learning_rate
                )
                
    def _compute_metrics(self) -> Dict[str, float]:
        """Compute temporal inference metrics"""
        if not self.predictions or not self.errors:
            return {}
            
        predictions = np.array(self.predictions)
        errors = np.array(self.errors)
        
        return {
            'prediction_error': float(np.mean(np.abs(errors))),
            'temporal_stability': self._compute_temporal_stability(predictions),
            'prediction_complexity': self._compute_prediction_complexity(predictions),
            'causal_strength': self._compute_causal_strength()
        }
        
    def _compute_temporal_stability(self, predictions: np.ndarray) -> float:
        """Compute stability of temporal predictions"""
        if len(predictions) < 2:
            return 1.0
            
        diffs = np.diff(predictions, axis=0)
        return float(1.0 - np.mean(np.abs(diffs)))
        
    def _compute_prediction_complexity(self, predictions: np.ndarray) -> float:
        """Compute complexity of predictions"""
        # Use entropy of prediction distributions
        pred_dist = softmax(predictions, axis=-1)
        return float(np.mean([entropy(p) for p in pred_dist]))
        
    def _compute_causal_strength(self) -> float:
        """Compute strength of causal relationships"""
        if len(self.errors) < 2:
            return 0.0
            
        # Use autocorrelation of errors as causal strength measure
        errors = np.array(self.errors)
        autocorr = np.correlate(errors, errors, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        return float(1.0 - np.mean(np.abs(autocorr)))
        
    def get_prediction(self, horizon: Optional[int] = None) -> np.ndarray:
        """Get future predictions"""
        if horizon is None:
            horizon = self.config.prediction_horizon
            
        predictions = []
        current_state = self.predictions[-1] if self.predictions else np.zeros(self.config.hidden_dim)
        
        for _ in range(horizon):
            next_pred = self._predict_next(current_state)
            predictions.append(next_pred)
            current_state = next_pred
            
        return np.array(predictions) 