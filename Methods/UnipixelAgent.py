"""
UnipixelAgent.py

Implements autonomous Unipixel agents that form the operational core of FractiAI.
These agents embody SAUUHUPP principles through self-awareness, adaptability,
and fractal coherence, enhanced with Free Energy Principle, Active Inference,
and Category Theory foundations.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from Methods.FreeEnergyPrinciple import FEPConfig, FreeEnergyMinimizer
from Methods.ActiveInference import ActiveInferenceConfig, ActiveInferenceAgent
from Methods.CategoryTheory import FractalCategory, NeuralObject, NeuralMorphism

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
    """Advanced autonomous agent implementing SAUUHUPP principles with FEP and Active Inference"""
    
    def __init__(self, config: UnipixelConfig):
        self.config = config
        self.state = np.zeros(config.dimensions)
        self.connections = []
        self.memory = UnipixelMemory(config.memory_capacity)
        self.awareness_state = {level: 0.0 for level in config.awareness_levels}
        
        # Initialize FEP components
        fep_config = FEPConfig(
            state_dim=config.dimensions,
            obs_dim=config.dimensions
        )
        self.fep = FreeEnergyMinimizer(fep_config)
        
        # Initialize Active Inference
        ai_config = ActiveInferenceConfig(
            n_policies=10,
            policy_horizon=5
        )
        self.active_inference = ActiveInferenceAgent(ai_config, fep_config)
        
        # Initialize Category Theory structure
        self.category = FractalCategory()
        self.neural_object = NeuralObject(
            dimension=config.dimensions,
            activation='tanh',
            state=self.state
        )
        self.category.neural_category.add_object(self.neural_object)
        
        # Add fractal scales
        self._initialize_fractal_structure()
        
    def _initialize_fractal_structure(self) -> None:
        """Initialize fractal categorical structure"""
        scales = [0.5, 0.25, 0.125]  # Multiple scales of observation
        for scale in scales:
            self.category.add_scale(scale)
            
        # Connect adjacent scales
        for i in range(len(scales) - 1):
            self.category.connect_scales(i, i + 1)
        
    def process(self, input_data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process input through fractal patterns with FEP and Active Inference"""
        metrics = {'harmony_score': 0.0, 'adaptation_level': 0.0}
        
        # Infer current state using FEP
        inferred_state, uncertainty = self.fep.infer_state(input_data)
        
        # Update internal state
        self.state = inferred_state
        self.neural_object.state = inferred_state
        
        # Process through fractal structure
        processed_data = input_data
        for depth in range(self.config.recursive_depth):
            # Transform data using categorical structure
            processed_data = self._categorical_transform(processed_data, depth)
            
            # Update self-awareness
            self._update_awareness(processed_data, depth)
            
            # Store patterns
            self.memory.store_pattern(processed_data)
            
            # Compute metrics
            metrics['harmony_score'] += self._compute_harmony(processed_data)
            metrics['adaptation_level'] += self._measure_adaptation()
        
        # Normalize metrics
        metrics = {k: v/self.config.recursive_depth for k, v in metrics.items()}
        
        # Add FEP and Active Inference metrics
        metrics['free_energy'] = self.fep.compute_free_energy(input_data)
        
        return processed_data, metrics
    
    def _categorical_transform(self, data: np.ndarray, depth: int) -> np.ndarray:
        """Apply transformation using categorical structure"""
        # Create morphism for current transformation
        morphism = NeuralMorphism(self.config.dimensions, self.config.dimensions)
        self.category.neural_category.add_morphism(self.neural_object, self.neural_object, morphism)
        
        # Apply transformation with depth-dependent scaling
        scale_factor = 1.0 / (depth + 1)
        transformed = morphism(data) * scale_factor
        
        # Verify categorical coherence
        if not self.category.verify_coherence():
            logger.warning("Categorical coherence violation detected")
            
        return transformed
    
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
        
        # Create categorical connection
        morphism = NeuralMorphism(self.config.dimensions, other_agent.config.dimensions)
        self.category.neural_category.add_morphism(
            self.neural_object,
            other_agent.neural_object,
            morphism
        )
        
        return True
    
    def _update_awareness(self, data: np.ndarray, depth: int) -> None:
        """Update self-awareness states based on processing"""
        # Local awareness through FEP
        self.awareness_state['local'] = 1.0 - self.fep.compute_free_energy(data)
        
        # Network awareness through connected agents
        if self.connections:
            network_state = np.mean([c.state for c in self.connections], axis=0)
            self.awareness_state['network'] = np.mean(np.abs(data - network_state))
        
        # Global awareness through categorical structure
        self.awareness_state['global'] = float(self.category.verify_coherence())
    
    def _compute_harmony(self, data: np.ndarray) -> float:
        """Compute harmony score of current processing state"""
        local_harmony = np.mean(np.abs(np.corrcoef(data, self.state)[0,1]))
        network_harmony = self._compute_network_harmony()
        categorical_harmony = float(self.category.verify_coherence())
        return (local_harmony + network_harmony + categorical_harmony) / 3
    
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
    
    def select_action(self, goal_state: np.ndarray) -> np.ndarray:
        """Select action using Active Inference"""
        return self.active_inference.select_action(self.state, goal_state)
    
    def update_models(self, observations: List[np.ndarray], 
                     actions: List[np.ndarray]) -> None:
        """Update internal models based on experience"""
        states = [self.state]  # Historical states
        self.active_inference.update_model(observations, actions, states) 