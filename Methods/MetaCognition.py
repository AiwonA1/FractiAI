"""
MetaCognition.py

Implements advanced metacognitive capabilities for FractiAI, enabling self-awareness,
pattern recognition, and adaptive learning across multiple cognitive dimensions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import networkx as nx
from scipy.stats import entropy
from Methods.FractalTransformer import FractalTransformer, FractalTransformerConfig

@dataclass
class MetaCogConfig:
    """Configuration for metacognitive system"""
    state_dim: int = 256
    memory_size: int = 1000
    num_patterns: int = 100
    learning_rate: float = 0.01
    pattern_threshold: float = 0.7
    adaptation_rate: float = 0.1
    coherence_threshold: float = 0.8
    exploration_factor: float = 1.0
    num_cognitive_layers: int = 3
    
class CognitiveState:
    """Represents current cognitive state with pattern awareness"""
    
    def __init__(self, dim: int):
        self.dim = dim
        self.state_pattern = np.zeros(dim)
        self.uncertainty = 1.0
        self.active_patterns: Dict[str, np.ndarray] = {}
        self.pattern_graph = nx.Graph()
        self.state_history: List[np.ndarray] = []
        self.meta_state: Dict[str, Any] = {}
        
    def update(self, new_state: np.ndarray,
              meta_info: Optional[Dict[str, Any]] = None) -> None:
        """Update cognitive state"""
        # Update state pattern
        self.state_pattern = new_state
        self.state_history.append(new_state)
        
        # Update meta information
        if meta_info:
            self.meta_state.update(meta_info)
            
        # Prune history if too long
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-1000:]
            
    def get_quantum_state(self) -> np.ndarray:
        """Get quantum representation of cognitive state"""
        # Create quantum amplitude
        amplitude = np.fft.fft(self.state_pattern)
        
        # Add phase from pattern coherence
        phase = np.angle(amplitude)
        if self.active_patterns:
            # Modify phase based on active patterns
            pattern_phase = np.mean([
                np.angle(np.fft.fft(p))
                for p in self.active_patterns.values()
            ], axis=0)
            phase = phase + pattern_phase
            
        return amplitude * np.exp(1j * phase)
        
class PatternMemory:
    """Memory system for storing and retrieving patterns"""
    
    def __init__(self, config: MetaCogConfig):
        self.config = config
        self.patterns: Dict[str, np.ndarray] = {}
        self.pattern_graph = nx.Graph()
        self.pattern_stats = defaultdict(list)
        
    def add_pattern(self, pattern_id: str, pattern: np.ndarray) -> None:
        """Add new pattern to memory"""
        self.patterns[pattern_id] = pattern
        
        # Update pattern graph
        self.pattern_graph.add_node(pattern_id)
        for other_id, other_pattern in self.patterns.items():
            if other_id != pattern_id:
                similarity = float(F.cosine_similarity(
                    torch.from_numpy(pattern).unsqueeze(0),
                    torch.from_numpy(other_pattern).unsqueeze(0)
                ))
                if similarity > self.config.pattern_threshold:
                    self.pattern_graph.add_edge(
                        pattern_id, other_id, weight=similarity)
                    
    def get_similar_patterns(self, pattern: np.ndarray,
                           threshold: float = None) -> List[Tuple[str, float]]:
        """Find patterns similar to input"""
        if threshold is None:
            threshold = self.config.pattern_threshold
            
        similarities = []
        for pattern_id, stored_pattern in self.patterns.items():
            similarity = float(F.cosine_similarity(
                torch.from_numpy(pattern).unsqueeze(0),
                torch.from_numpy(stored_pattern).unsqueeze(0)
            ))
            if similarity > threshold:
                similarities.append((pattern_id, similarity))
                
        return sorted(similarities, key=lambda x: x[1], reverse=True)
        
    def get_pattern_clusters(self) -> List[List[str]]:
        """Get clusters of related patterns"""
        return list(nx.community.greedy_modularity_communities(self.pattern_graph))
        
class MetaCognition:
    """Main metacognitive system"""
    
    def __init__(self, config: MetaCogConfig):
        self.config = config
        
        # Initialize components
        self.state = CognitiveState(config.state_dim)
        self.memory = PatternMemory(config)
        
        # Initialize pattern transformer
        transformer_config = FractalTransformerConfig(
            hidden_dim=config.state_dim,
            pattern_dim=config.state_dim // 2,
            coherence_threshold=config.coherence_threshold
        )
        self.pattern_transformer = FractalTransformer(transformer_config)
        
        # Initialize cognitive layers
        self.cognitive_layers = nn.ModuleList([
            self._build_cognitive_layer()
            for _ in range(config.num_cognitive_layers)
        ])
        
        # Tracking metrics
        self.uncertainty_history = []
        self.adaptation_history = []
        self.exploration_history = []
        
    def _build_cognitive_layer(self) -> nn.Module:
        """Build single cognitive processing layer"""
        return nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.state_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.state_dim * 2, self.config.state_dim)
        )
        
    def process_state(self, state: np.ndarray,
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process cognitive state update"""
        # Convert to tensor
        state_tensor = torch.from_numpy(state).float()
        
        # Process through cognitive layers
        cognitive_states = []
        current_state = state_tensor
        for layer in self.cognitive_layers:
            current_state = layer(current_state)
            cognitive_states.append(current_state)
            
        # Transform pattern
        transformed = self.pattern_transformer.transform_pattern(
            state_tensor.unsqueeze(0))
            
        # Find similar patterns
        similar_patterns = self.memory.get_similar_patterns(
            transformed.squeeze(0).detach().numpy())
            
        # Update active patterns
        self.state.active_patterns.clear()
        for pattern_id, similarity in similar_patterns:
            self.state.active_patterns[pattern_id] = self.memory.patterns[pattern_id]
            
        # Calculate uncertainty
        if similar_patterns:
            uncertainty = 1.0 - max(s for _, s in similar_patterns)
        else:
            uncertainty = 1.0
            
        # Update state
        final_state = cognitive_states[-1].detach().numpy()
        self.state.update(final_state, {
            'uncertainty': uncertainty,
            'num_patterns': len(similar_patterns),
            'cognitive_states': [s.detach().numpy() for s in cognitive_states]
        })
        
        # Track metrics
        self.uncertainty_history.append(uncertainty)
        
        # Generate meta-cognitive insights
        insights = self._generate_insights(cognitive_states, similar_patterns)
        
        return {
            'state': final_state,
            'uncertainty': uncertainty,
            'active_patterns': list(self.state.active_patterns.keys()),
            'insights': insights
        }
        
    def _generate_insights(self, cognitive_states: List[torch.Tensor],
                         similar_patterns: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Generate metacognitive insights"""
        insights = {}
        
        # Analyze cognitive trajectory
        state_distances = [
            float(F.pairwise_distance(s1, s2).mean())
            for s1, s2 in zip(cognitive_states[:-1], cognitive_states[1:])
        ]
        
        # Calculate cognitive coherence
        coherence = float(torch.mean(torch.stack([
            F.cosine_similarity(s1, s2, dim=0)
            for s1, s2 in zip(cognitive_states[:-1], cognitive_states[1:])
        ])))
        
        # Analyze pattern stability
        pattern_stability = 0.0
        if similar_patterns:
            pattern_similarities = [s for _, s in similar_patterns]
            pattern_stability = float(np.std(pattern_similarities))
            
        insights.update({
            'cognitive_distance': np.mean(state_distances),
            'cognitive_coherence': coherence,
            'pattern_stability': pattern_stability,
            'exploration_factor': self.config.exploration_factor * 
                                (1.0 - coherence) * pattern_stability
        })
        
        return insights
        
    def adapt(self, reward: float) -> None:
        """Adapt based on reward signal"""
        # Update exploration factor
        if len(self.uncertainty_history) > 1:
            uncertainty_change = (self.uncertainty_history[-1] - 
                                self.uncertainty_history[-2])
            self.config.exploration_factor *= (1.0 + 
                uncertainty_change * self.config.adaptation_rate)
            
        # Track adaptation
        self.adaptation_history.append({
            'reward': reward,
            'exploration_factor': self.config.exploration_factor
        })
        
    def get_state(self) -> Dict[str, Any]:
        """Get current cognitive state"""
        return {
            'state_pattern': self.state.state_pattern,
            'uncertainty': self.state.uncertainty,
            'active_patterns': self.state.active_patterns,
            'meta_state': self.state.meta_state
        }
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get metacognitive metrics"""
        return {
            'mean_uncertainty': float(np.mean(self.uncertainty_history)),
            'exploration_factor': self.config.exploration_factor,
            'num_patterns': len(self.memory.patterns),
            'pattern_clusters': len(self.memory.get_pattern_clusters())
        } 