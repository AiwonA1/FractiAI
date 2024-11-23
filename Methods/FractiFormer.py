"""
FractiFormer.py

Implements the FractiFormer component of FractiAI, providing recursive attention mechanisms
and fractal pattern processing capabilities.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class FormerConfig:
    dim: int
    heads: int = 8
    depth: int = 6
    ff_dim: int = 256
    dropout: float = 0.1
    max_sequence_length: int = 1024

class FractalAttention:
    """Implements fractal-aware multi-head attention"""
    
    def __init__(self, config: FormerConfig):
        self.config = config
        self.head_dim = config.dim // config.heads
        self.scale = self.head_dim ** -0.5
        
        # Initialize transformation matrices
        self.W_q = np.random.randn(config.dim, config.dim)
        self.W_k = np.random.randn(config.dim, config.dim)
        self.W_v = np.random.randn(config.dim, config.dim)
        self.W_o = np.random.randn(config.dim, config.dim)
        
        self.pattern_memory = []
        
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """Process input through fractal attention mechanism"""
        batch_size, seq_len, _ = x.shape
        
        # Project inputs to Q, K, V
        Q = self._reshape_heads(np.dot(x, self.W_q))
        K = self._reshape_heads(np.dot(x, self.W_k))
        V = self._reshape_heads(np.dot(x, self.W_v))
        
        # Compute scaled dot-product attention with fractal awareness
        attention_scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
        attention_probs = self._fractal_softmax(attention_scores)
        
        # Apply attention to values
        context = np.matmul(attention_probs, V)
        context = self._reshape_batch(context)
        output = np.dot(context, self.W_o)
        
        # Compute attention metrics
        metrics = self._compute_attention_metrics(attention_probs)
        
        return output, metrics
    
    def _reshape_heads(self, x: np.ndarray) -> np.ndarray:
        """Reshape input for multi-head attention"""
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.config.heads, self.head_dim)
    
    def _reshape_batch(self, x: np.ndarray) -> np.ndarray:
        """Reshape attention output back to batch format"""
        batch_size, seq_len, _, _ = x.shape
        return x.reshape(batch_size, seq_len, self.config.dim)
    
    def _fractal_softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax with fractal pattern awareness"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _compute_attention_metrics(self, attention_probs: np.ndarray) -> dict:
        """Compute metrics about the attention patterns"""
        metrics = {
            'attention_entropy': self._compute_attention_entropy(attention_probs),
            'pattern_diversity': self._compute_pattern_diversity(attention_probs),
            'fractal_dimension': self._estimate_fractal_dimension(attention_probs)
        }
        return metrics
    
    def _compute_attention_entropy(self, probs: np.ndarray) -> float:
        """Compute entropy of attention distributions"""
        entropy = -np.sum(probs * np.log(probs + 1e-9), axis=-1)
        return float(np.mean(entropy))
    
    def _compute_pattern_diversity(self, probs: np.ndarray) -> float:
        """Measure diversity of attention patterns"""
        pattern_vectors = probs.reshape(-1, probs.shape[-1])
        similarities = np.matmul(pattern_vectors, pattern_vectors.T)
        return float(1 - np.mean(similarities))
    
    def _estimate_fractal_dimension(self, probs: np.ndarray) -> float:
        """Estimate fractal dimension of attention patterns"""
        # Simplified box-counting dimension estimation
        pattern = probs.mean(axis=(0,1))
        boxes = []
        for scale in [2, 4, 8]:
            boxes.append(np.sum(pattern.reshape(-1, scale).any(axis=1)))
        if len(boxes) > 1:
            dimension = np.polyfit(np.log(boxes), np.log([2,4,8]), 1)[0]
            return abs(float(dimension))
        return 1.0

class FractiFormer:
    """Advanced transformer architecture adapted for fractal processing"""
    
    def __init__(self, config: FormerConfig):
        self.config = config
        self.layers = [FractalAttention(config) for _ in range(config.depth)]
        self.pattern_bank = []
        
    def process(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[dict]]:
        """Process input through fractal transformer layers"""
        metrics_list = []
        
        for layer in self.layers:
            x, metrics = layer.forward(x, mask)
            metrics_list.append(metrics)
            
        # Store emergent patterns
        self.pattern_bank.append(self._extract_patterns(x))
        
        return x, metrics_list
    
    def _extract_patterns(self, x: np.ndarray) -> np.ndarray:
        """Extract and store emergent fractal patterns"""
        return np.mean(x, axis=(0,1)) 