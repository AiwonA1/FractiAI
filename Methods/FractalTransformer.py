"""
FractalTransformer.py

Implements advanced fractal-based transformer architecture for pattern recognition,
transformation and resonance across multiple scales and domains.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.stats import entropy
import networkx as nx

@dataclass 
class FractalTransformerConfig:
    """Configuration for fractal transformer"""
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1
    max_seq_length: int = 1024
    pattern_dim: int = 64
    num_scales: int = 4
    resonance_factor: float = 0.1
    coherence_threshold: float = 0.7
    adaptation_rate: float = 0.01
    
class FractalAttention(nn.Module):
    """Multi-scale fractal attention mechanism"""
    
    def __init__(self, config: FractalTransformerConfig):
        super().__init__()
        self.config = config
        
        # Multi-scale projections
        self.scale_projections = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.hidden_dim)
            for _ in range(config.num_scales)
        ])
        
        # Fractal attention heads
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with fractal attention"""
        B, L, D = x.shape
        H = self.config.num_heads
        
        # Generate queries, keys, values
        q = self.q_proj(x).view(B, L, H, -1)
        k = self.k_proj(x).view(B, L, H, -1)
        v = self.v_proj(x).view(B, L, H, -1)
        
        # Multi-scale processing
        scale_outputs = []
        for scale_proj in self.scale_projections:
            # Project to scale space
            scale_q = scale_proj(q.view(B, L, -1)).view(B, L, H, -1)
            scale_k = scale_proj(k.view(B, L, -1)).view(B, L, H, -1)
            scale_v = scale_proj(v.view(B, L, -1)).view(B, L, H, -1)
            
            # Compute attention scores
            scores = torch.matmul(scale_q, scale_k.transpose(-2, -1)) / np.sqrt(D // H)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
                
            # Apply attention
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            
            # Get scale output
            scale_out = torch.matmul(attn, scale_v)
            scale_outputs.append(scale_out)
            
        # Combine scales with learned weights
        scale_weights = F.softmax(torch.randn(len(scale_outputs)), dim=0)
        combined = sum(w * o for w, o in zip(scale_weights, scale_outputs))
        
        # Project to output space
        output = self.o_proj(combined.view(B, L, -1))
        
        return output

class FractalPatternLayer(nn.Module):
    """Pattern recognition and transformation layer"""
    
    def __init__(self, config: FractalTransformerConfig):
        super().__init__()
        self.config = config
        
        # Pattern detection
        self.pattern_proj = nn.Linear(config.hidden_dim, config.pattern_dim)
        self.pattern_attn = FractalAttention(config)
        
        # Pattern transformation
        self.transform_net = nn.Sequential(
            nn.Linear(config.pattern_dim, config.ff_dim),
            nn.GELU(),
            nn.Linear(config.ff_dim, config.pattern_dim)
        )
        
        # Resonance components
        self.resonance_gate = nn.Linear(config.pattern_dim * 2, 1)
        self.adaptation_layer = nn.Linear(config.pattern_dim, config.pattern_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        
    def forward(self, x: torch.Tensor,
               patterns: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with pattern recognition"""
        # Project to pattern space
        pattern_features = self.pattern_proj(x)
        
        # Apply pattern attention
        pattern_context = self.pattern_attn(pattern_features)
        
        # Detect and transform patterns
        transformed_patterns = {}
        if patterns is not None:
            for name, pattern in patterns.items():
                # Pattern matching
                similarity = F.cosine_similarity(
                    pattern_features.unsqueeze(1),
                    pattern.unsqueeze(0),
                    dim=-1
                )
                
                # Transform matched patterns
                if similarity.max() > self.config.coherence_threshold:
                    transformed = self.transform_net(pattern)
                    
                    # Apply resonance
                    resonance = torch.sigmoid(
                        self.resonance_gate(
                            torch.cat([pattern, transformed], dim=-1)
                        )
                    )
                    transformed = resonance * transformed + (1 - resonance) * pattern
                    
                    # Adapt pattern
                    adaptation = torch.tanh(
                        self.adaptation_layer(transformed)
                    ) * self.config.adaptation_rate
                    transformed = transformed + adaptation
                    
                    transformed_patterns[name] = transformed
                    
        # Combine with input
        output = x + self.dropout(pattern_context)
        output = self.layer_norm(output)
        
        return output, transformed_patterns

class FractalTransformer(nn.Module):
    """Main fractal transformer architecture"""
    
    def __init__(self, config: FractalTransformerConfig):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.input_embed = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Fractal pattern layers
        self.pattern_layers = nn.ModuleList([
            FractalPatternLayer(config)
            for _ in range(config.num_layers)
        ])
        
        # Pattern memory
        self.pattern_memory: Dict[str, torch.Tensor] = {}
        self.pattern_graph = nx.Graph()
        
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        
    def forward(self, x: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through fractal transformer"""
        # Input embedding
        x = self.input_embed(x)
        x = self.dropout(x)
        
        # Process through pattern layers
        all_patterns = {}
        for layer in self.pattern_layers:
            x, patterns = layer(x, self.pattern_memory)
            all_patterns.update(patterns)
            
        # Update pattern memory
        self._update_pattern_memory(all_patterns)
        
        # Final normalization
        output = self.layer_norm(x)
        
        return output
        
    def _update_pattern_memory(self, new_patterns: Dict[str, torch.Tensor]) -> None:
        """Update pattern memory and graph"""
        # Update patterns
        self.pattern_memory.update(new_patterns)
        
        # Update pattern graph
        for name1, pattern1 in self.pattern_memory.items():
            self.pattern_graph.add_node(name1)
            for name2, pattern2 in self.pattern_memory.items():
                if name1 != name2:
                    similarity = F.cosine_similarity(
                        pattern1.mean(0),
                        pattern2.mean(0),
                        dim=0
                    )
                    if similarity > self.config.coherence_threshold:
                        self.pattern_graph.add_edge(
                            name1, name2, weight=float(similarity)
                        )
                        
        # Prune old patterns
        if len(self.pattern_memory) > self.config.max_seq_length:
            # Remove least connected patterns
            to_remove = []
            for node in self.pattern_graph.nodes():
                if len(list(self.pattern_graph.neighbors(node))) == 0:
                    to_remove.append(node)
            
            for node in to_remove:
                self.pattern_graph.remove_node(node)
                self.pattern_memory.pop(node)
                
    def get_pattern_clusters(self) -> List[List[str]]:
        """Get clusters of related patterns"""
        return list(nx.community.greedy_modularity_communities(self.pattern_graph))
        
    def get_pattern_centrality(self) -> Dict[str, float]:
        """Get pattern importance scores"""
        return nx.eigenvector_centrality(self.pattern_graph, weight='weight')
        
    def transform_pattern(self, pattern: torch.Tensor) -> torch.Tensor:
        """Transform a single pattern"""
        # Ensure correct shape
        if len(pattern.shape) == 2:
            pattern = pattern.unsqueeze(0)
            
        # Pass through transformer
        transformed = self.forward(pattern)
        
        return transformed.squeeze(0) 