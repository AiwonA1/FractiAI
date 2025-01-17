"""
FractalTransformer.py

Implements advanced fractal-based transformer architecture for pattern processing
and transformation across multiple scales and dimensions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

@dataclass
class FractalTransformerConfig:
    """Configuration for fractal transformer"""
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    activation: str = "relu"
    layer_norm_eps: float = 1e-5
    batch_first: bool = True
    norm_first: bool = False
    fractal_depth: int = 3
    scale_factor: float = 2.0

class FractalAttention(nn.Module):
    """Multi-scale fractal attention mechanism"""
    
    def __init__(self, config: FractalTransformerConfig):
        super().__init__()
        self.config = config
        self.scales = nn.ModuleList([
            nn.MultiheadAttention(
                config.d_model,
                config.nhead,
                dropout=config.dropout,
                batch_first=config.batch_first
            )
            for _ in range(config.fractal_depth)
        ])
        self.scale_weights = nn.Parameter(torch.ones(config.fractal_depth))
        
    def forward(self, x: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with multi-scale attention"""
        # Normalize scale weights
        scale_weights = F.softmax(self.scale_weights, dim=0)
        
        # Process each scale
        outputs = []
        current_x = x
        for i, attention in enumerate(self.scales):
            # Apply attention at current scale
            scale_out, _ = attention(current_x, current_x, current_x, 
                                   attn_mask=mask)
            outputs.append(scale_out * scale_weights[i])
            
            # Downsample for next scale
            if i < len(self.scales) - 1:
                current_x = F.avg_pool1d(
                    current_x.transpose(1, 2),
                    kernel_size=2,
                    stride=2
                ).transpose(1, 2)
                
        # Combine scales
        return sum(outputs)

class FractalTransformerLayer(nn.Module):
    """Transformer layer with fractal attention"""
    
    def __init__(self, config: FractalTransformerConfig):
        super().__init__()
        self.config = config
        
        # Fractal attention
        self.fractal_attention = FractalAttention(config)
        
        # Feed forward
        self.linear1 = nn.Linear(config.d_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Activation
        self.activation = getattr(F, config.activation)
        
    def forward(self, x: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        if self.config.norm_first:
            x = x + self._fractal_attention_block(self.norm1(x), mask)
            x = x + self._feedforward_block(self.norm2(x))
        else:
            x = self.norm1(x + self._fractal_attention_block(x, mask))
            x = self.norm2(x + self._feedforward_block(x))
        return x
        
    def _fractal_attention_block(self, x: torch.Tensor,
                               mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply fractal attention"""
        return self.fractal_attention(x, mask)
        
    def _feedforward_block(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed forward layer"""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)

class FractalTransformer(nn.Module):
    """Main fractal transformer model"""
    
    def __init__(self, config: FractalTransformerConfig):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.embedding = nn.Linear(config.d_model, config.d_model)
        
        # Fractal transformer layers
        self.layers = nn.ModuleList([
            FractalTransformerLayer(config)
            for _ in range(config.num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.d_model)
        
        # Position encoding
        self.pos_encoder = self._build_position_encoding()
        
    def _build_position_encoding(self) -> nn.Module:
        """Build sinusoidal position encoding"""
        pe = torch.zeros(1000, self.config.d_model)
        position = torch.arange(0, 1000).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.config.d_model, 2) * 
            -(np.log(10000.0) / self.config.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return nn.Parameter(pe, requires_grad=False)
        
    def forward(self, x: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        # Add position encoding
        x = x + self.pos_encoder[:, :x.size(1)]
        
        # Initial embedding
        x = self.embedding(x)
        
        # Process through layers
        for layer in self.layers:
            x = layer(x, mask)
            
        # Output projection
        x = self.output_projection(x)
        
        return x
    
    def encode_pattern(self, pattern: torch.Tensor) -> torch.Tensor:
        """Encode pattern through transformer"""
        return self.forward(pattern)
    
    def decode_pattern(self, encoded: torch.Tensor) -> torch.Tensor:
        """Decode pattern from transformer encoding"""
        # Use same forward pass but with different head
        return self.forward(encoded)
    
    def transform_pattern(self, pattern: torch.Tensor,
                        target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Transform pattern while preserving fractal properties"""
        # Encode pattern
        encoded = self.encode_pattern(pattern)
        
        if target is not None:
            # Align with target pattern
            target_encoded = self.encode_pattern(target)
            encoded = encoded + (target_encoded - encoded) * 0.5
            
        # Decode transformed pattern
        return self.decode_pattern(encoded) 