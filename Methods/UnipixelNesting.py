"""
UnipixelNesting.py

Implements elegant hierarchical nesting of Unipixel agents through categorical
composition and free energy optimization, enabling emergent multi-scale intelligence.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import logging
from Methods.UnipixelAgent import UnipixelAgent, UnipixelConfig
from Methods.CategoryTheory import FractalCategory, NeuralObject, NeuralMorphism
from Methods.FreeEnergyPrinciple import FEPConfig
from Methods.UnipixelConnector import UnipixelConnector, ConnectorConfig

logger = logging.getLogger(__name__)

@dataclass
class NestingConfig:
    """Configuration for hierarchical nesting"""
    max_depth: int = 5
    min_agents_per_nest: int = 3
    max_agents_per_nest: int = 10
    coherence_threshold: float = 0.8
    emergence_threshold: float = 0.7
    adaptation_rate: float = 0.01

class NestedUnipixel:
    """Represents a nested group of Unipixel agents"""
    
    def __init__(self, config: NestingConfig, depth: int = 0):
        self.config = config
        self.depth = depth
        self.agents: List[UnipixelAgent] = []
        self.subnests: List['NestedUnipixel'] = []
        self.category = FractalCategory()
        self.connector = UnipixelConnector(ConnectorConfig())
        self.state = np.array([])
        
    def add_agent(self, agent: UnipixelAgent) -> bool:
        """Add agent to optimal position in hierarchy"""
        if len(self.agents) < self.config.max_agents_per_nest:
            self._integrate_agent(agent)
            return True
            
        if self.depth < self.config.max_depth:
            return self._nest_agent(agent)
            
        return False
    
    def process(self, input_data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process input through nested hierarchy"""
        metrics = {'coherence': 0.0, 'emergence': 0.0}
        
        # Process through direct agents
        agent_outputs = []
        for agent in self.agents:
            output, agent_metrics = agent.process(input_data)
            agent_outputs.append(output)
            
        # Process through subnests
        subnest_outputs = []
        for subnest in self.subnests:
            output, nest_metrics = subnest.process(input_data)
            subnest_outputs.append(output)
            
        # Integrate outputs through categorical composition
        if agent_outputs or subnest_outputs:
            outputs = agent_outputs + subnest_outputs
            self.state = self._integrate_outputs(outputs)
            
            # Update metrics
            metrics['coherence'] = self._compute_coherence()
            metrics['emergence'] = self._compute_emergence(outputs)
            
        return self.state, metrics
    
    def _integrate_agent(self, agent: UnipixelAgent) -> None:
        """Integrate new agent into current nest level"""
        self.agents.append(agent)
        self.connector.add_agent(agent)
        
        # Create categorical connection
        self.category.neural_category.add_object(agent.neural_object)
        
        # Update nest state
        if len(self.state) == 0:
            self.state = agent.state.copy()
        else:
            self.state = np.mean([self.state, agent.state], axis=0)
    
    def _nest_agent(self, agent: UnipixelAgent) -> bool:
        """Place agent in optimal subnest"""
        # Find best matching subnest
        best_subnest = None
        best_harmony = -1.0
        
        for subnest in self.subnests:
            harmony = self._compute_nesting_harmony(agent, subnest)
            if harmony > best_harmony:
                best_harmony = harmony
                best_subnest = subnest
                
        # Create new subnest if necessary
        if best_harmony < self.config.coherence_threshold:
            if len(self.subnests) * self.config.max_agents_per_nest < self.config.max_agents_per_nest:
                best_subnest = NestedUnipixel(self.config, self.depth + 1)
                self.subnests.append(best_subnest)
                
        if best_subnest:
            return best_subnest.add_agent(agent)
            
        return False
    
    def _integrate_outputs(self, outputs: List[np.ndarray]) -> np.ndarray:
        """Integrate outputs using categorical structure"""
        # Create integration morphism
        if not outputs:
            return np.zeros_like(self.state) if self.state.size > 0 else np.array([])
            
        output_dim = outputs[0].shape[0]
        morphism = NeuralMorphism(output_dim, output_dim)
        
        # Apply morphism to each output
        transformed = []
        for output in outputs:
            transformed.append(morphism(output))
            
        # Combine using weighted average based on coherence
        weights = self._compute_output_weights(transformed)
        integrated = np.average(transformed, axis=0, weights=weights)
        
        return integrated
    
    def _compute_output_weights(self, outputs: List[np.ndarray]) -> np.ndarray:
        """Compute integration weights based on categorical coherence"""
        weights = np.ones(len(outputs))
        
        for i, output in enumerate(outputs):
            # Create temporary neural object for output
            temp_obj = NeuralObject(
                dimension=output.shape[0],
                activation='tanh',
                state=output
            )
            
            # Add to category temporarily
            self.category.neural_category.add_object(temp_obj)
            
            # Compute coherence
            weights[i] = float(self.category.verify_coherence())
            
            # Remove temporary object
            self.category.neural_category.objects.remove(temp_obj)
            
        # Normalize weights
        weights = np.maximum(weights, 1e-6)  # Prevent division by zero
        return weights / np.sum(weights)
    
    def _compute_nesting_harmony(self, agent: UnipixelAgent, 
                               nest: 'NestedUnipixel') -> float:
        """Compute harmony between agent and nest"""
        if len(nest.state) == 0:
            return 1.0  # Empty nest is maximally receptive
            
        # Compute state harmony
        state_harmony = float(np.abs(np.corrcoef(agent.state, nest.state)[0,1]))
        
        # Compute categorical harmony
        transformation = NaturalTransformation(
            agent.category.functors[0],
            nest.category.functors[0] if nest.category.functors else agent.category.functors[0]
        )
        categorical_harmony = float(transformation.is_natural())
        
        return (state_harmony + categorical_harmony) / 2
    
    def _compute_coherence(self) -> float:
        """Compute internal coherence of nest"""
        coherence_scores = []
        
        # Agent coherence
        for agent in self.agents:
            coherence_scores.append(float(agent.category.verify_coherence()))
            
        # Subnest coherence
        for subnest in self.subnests:
            metrics = subnest._compute_network_metrics()
            coherence_scores.append(metrics['coherence'])
            
        # Network coherence
        if self.agents:
            network_metrics = self.connector.update()
            coherence_scores.append(network_metrics['global_coherence'])
            
        return np.mean(coherence_scores) if coherence_scores else 1.0
    
    def _compute_emergence(self, outputs: List[np.ndarray]) -> float:
        """Compute emergence of higher-order patterns"""
        if len(outputs) < 2:
            return 0.0
            
        # Compute pairwise correlations
        correlations = []
        outputs = np.array(outputs)
        for i in range(len(outputs)):
            for j in range(i+1, len(outputs)):
                corr = np.abs(np.corrcoef(outputs[i], outputs[j])[0,1])
                correlations.append(corr)
                
        # Emergence is high when outputs are correlated but not identical
        mean_corr = np.mean(correlations)
        var_corr = np.var(correlations)
        
        return mean_corr * (1.0 - np.exp(-var_corr))
    
    def _compute_network_metrics(self) -> Dict:
        """Compute comprehensive nest metrics"""
        return {
            'coherence': self._compute_coherence(),
            'emergence': self._compute_emergence([a.state for a in self.agents]),
            'depth': self.depth,
            'total_agents': len(self.agents) + sum(len(n.agents) for n in self.subnests)
        } 