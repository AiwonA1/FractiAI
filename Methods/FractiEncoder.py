"""
FractiEncoder.py

Implements the FractiEncoder component of FractiAI, responsible for transforming data 
into fractal representations across organic, inorganic, and abstract dimensions.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class EncoderConfig:
    input_dim: int
    hidden_dims: List[int] = (64, 128, 256)
    activation: str = 'tanh'
    compression_ratio: float = 0.85
    harmony_weight: float = 0.7

class FractalLayer:
    """Implements a single fractal transformation layer"""
    
    def __init__(self, in_dim: int, out_dim: int):
        self.weights = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        self.bias = np.zeros(out_dim)
        self.pattern_memory = []
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply fractal transformation"""
        output = np.dot(x, self.weights) + self.bias
        self.pattern_memory.append(np.mean(output, axis=0))
        return output

class FractiEncoder:
    """Advanced encoder for transforming data into fractal representations"""
    
    def __init__(self, config: EncoderConfig):
        self.config = config
        self.layers = self._initialize_layers()
        self.pattern_bank = []
        
    def _initialize_layers(self) -> List[FractalLayer]:
        """Initialize fractal transformation layers"""
        layers = []
        in_dim = self.config.input_dim
        
        for hidden_dim in self.config.hidden_dims:
            layers.append(FractalLayer(in_dim, hidden_dim))
            in_dim = hidden_dim
            
        return layers
    
    def encode(self, data: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Transform input data into fractal representation"""
        x = data
        layer_outputs = []
        
        # Forward pass through fractal layers
        for layer in self.layers:
            x = layer.forward(x)
            x = self._apply_activation(x)
            layer_outputs.append(x)
            
        # Compute fractal metrics
        metrics = self._compute_fractal_metrics(layer_outputs)
        
        return x, metrics
    
    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        """Apply configured activation function"""
        if self.config.activation == 'tanh':
            return np.tanh(x)
        elif self.config.activation == 'relu':
            return np.maximum(0, x)
        return x
    
    def _compute_fractal_metrics(self, outputs: List[np.ndarray]) -> dict:
        """Compute metrics about the fractal encoding"""
        metrics = {
            'pattern_strength': self._measure_pattern_strength(outputs),
            'harmony_score': self._calculate_harmony(outputs),
            'compression_efficiency': self._evaluate_compression(outputs)
        }
        return metrics
    
    def _measure_pattern_strength(self, outputs: List[np.ndarray]) -> float:
        """Measure the strength of fractal patterns in the encoding"""
        pattern_scores = []
        for i in range(len(outputs) - 1):
            correlation = np.corrcoef(
                outputs[i].flatten(), 
                outputs[i+1].flatten()
            )[0,1]
            pattern_scores.append(abs(correlation))
        return np.mean(pattern_scores) if pattern_scores else 0.0
    
    def _calculate_harmony(self, outputs: List[np.ndarray]) -> float:
        """Calculate harmony score between layers"""
        harmony_scores = []
        for output in outputs:
            score = np.std(output) / (np.mean(abs(output)) + 1e-7)
            harmony_scores.append(self.config.harmony_weight * (1 - min(score, 1)))
        return np.mean(harmony_scores)
    
    def _evaluate_compression(self, outputs: List[np.ndarray]) -> float:
        """Evaluate the efficiency of fractal compression"""
        initial_size = outputs[0].size
        final_size = outputs[-1].size
        compression_ratio = final_size / initial_size
        return max(0, 1 - compression_ratio/self.config.compression_ratio) 