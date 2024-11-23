"""
UnipixelAgent.py

Implements autonomous Unipixel agents that form the operational core of FractiAI.
These agents embody SAUUHUPP principles through self-awareness, adaptability,
and fractal coherence.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class UnipixelConfig:
    """Configuration for Unipixel agents"""
    dimensions: int = 3  # Organic, Inorganic, Abstract
    recursive_depth: int = 5
    harmony_threshold: float = 0.85
    adaptation_rate: float = 0.01
    memory_capacity: int = 1000
    awareness_levels: List[str] = ('local', 'network', 'global')
    connection_limit: int = 50

class UnipixelMemory:
    """Memory system for Unipixel agents"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.patterns = []
        self.interactions = []
        self.feedback_history = []
        
    def store_pattern(self, pattern: np.ndarray) -> None:
        """Store observed pattern with FIFO behavior"""
        self.patterns.append(pattern)
        if len(self.patterns) > self.capacity:
            self.patterns.pop(0)
            
    def store_interaction(self, interaction: Dict) -> None:
        """Store interaction data"""
        self.interactions.append(interaction)
        if len(self.interactions) > self.capacity:
            self.interactions.pop(0)
            
    def analyze_patterns(self) -> Dict:
        """Analyze stored patterns for emergent behaviors"""
        if not self.patterns:
            return {}
        
        patterns = np.array(self.patterns)
        return {
            'mean_pattern': np.mean(patterns, axis=0),
            'pattern_variance': np.var(patterns, axis=0),
            'pattern_evolution': self._compute_pattern_evolution(patterns)
        }
    
    def _compute_pattern_evolution(self, patterns: np.ndarray) -> float:
        """Measure how patterns evolve over time"""
        if len(patterns) < 2:
            return 0.0
        differences = np.diff(patterns, axis=0)
        return float(np.mean(np.abs(differences)))

class UnipixelAgent:
    """Advanced autonomous agent implementing SAUUHUPP principles"""
    
    def __init__(self, config: UnipixelConfig):
        self.config = config
        self.state = np.zeros(config.dimensions)
        self.connections = []
        self.memory = UnipixelMemory(config.memory_capacity)
        self.awareness_state = {level: 0.0 for level in config.awareness_levels}
        self.active_template = None
        
    def process(self, input_data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process input through fractal patterns with self-awareness"""
        # Initialize processing metrics
        metrics = {'harmony_score': 0.0, 'adaptation_level': 0.0}
        
        # Apply recursive processing with awareness
        for depth in range(self.config.recursive_depth):
            # Transform data
            input_data = self._recursive_transform(input_data, depth)
            
            # Update self-awareness
            self._update_awareness(input_data, depth)
            
            # Store patterns
            self.memory.store_pattern(input_data)
            
            # Compute metrics
            metrics['harmony_score'] += self._compute_harmony(input_data)
            metrics['adaptation_level'] += self._measure_adaptation()
            
        # Normalize metrics
        metrics = {k: v/self.config.recursive_depth for k, v in metrics.items()}
        
        return input_data, metrics
    
    def connect(self, other_agent: 'UnipixelAgent') -> bool:
        """Establish connection with another Unipixel agent"""
        if len(self.connections) >= self.config.connection_limit:
            return False
        
        self.connections.append(other_agent)
        self.memory.store_interaction({
            'agent_id': id(other_agent),
            'timestamp': np.datetime64('now'),
            'connection_type': 'direct'
        })
        return True
    
    def _recursive_transform(self, data: np.ndarray, depth: int) -> np.ndarray:
        """Apply recursive transformation with depth awareness"""
        # Combine current state with input
        transformed = np.tanh(data + self.state * (1.0 / (depth + 1)))
        
        # Apply template if available
        if self.active_template:
            transformed = self.active_template.apply_transformation(transformed)
            
        return transformed
    
    def _update_awareness(self, data: np.ndarray, depth: int) -> None:
        """Update self-awareness states based on processing"""
        # Local awareness
        self.awareness_state['local'] = np.mean(np.abs(data - self.state))
        
        # Network awareness (if connected)
        if self.connections:
            network_state = np.mean([c.state for c in self.connections], axis=0)
            self.awareness_state['network'] = np.mean(np.abs(data - network_state))
        
        # Global awareness (through template alignment)
        if self.active_template:
            self.awareness_state['global'] = self.active_template.measure_alignment(data)
    
    def _compute_harmony(self, data: np.ndarray) -> float:
        """Compute harmony score of current processing state"""
        local_harmony = np.mean(np.abs(np.corrcoef(data, self.state)[0,1]))
        network_harmony = self._compute_network_harmony()
        return (local_harmony + network_harmony) / 2
    
    def _compute_network_harmony(self) -> float:
        """Compute harmony with connected agents"""
        if not self.connections:
            return 1.0
        
        harmonies = []
        for connection in self.connections:
            harmony = np.mean(np.abs(np.corrcoef(self.state, connection.state)[0,1]))
            harmonies.append(harmony)
        
        return np.mean(harmonies)
    
    def _measure_adaptation(self) -> float:
        """Measure agent's adaptation level"""
        memory_analysis = self.memory.analyze_patterns()
        if not memory_analysis:
            return 0.0
        
        evolution_rate = memory_analysis.get('pattern_evolution', 0.0)
        return np.clip(evolution_rate / self.config.adaptation_rate, 0, 1) 