"""
UnipixelConnector.py

Implements an elegant, self-organizing connector system for Unipixel agents,
optimizing network topology through categorical functors and free energy minimization.
"""

import numpy as np
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
import logging
from Methods.UnipixelAgent import UnipixelAgent
from Methods.CategoryTheory import NeuralMorphism, NaturalTransformation
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)

@dataclass
class ConnectorConfig:
    """Configuration for Unipixel connection optimization"""
    max_connections: int = 50
    harmony_threshold: float = 0.75
    adaptation_rate: float = 0.01
    topology_update_interval: int = 100
    min_coherence: float = 0.8

class UnipixelConnector:
    """Manages optimal connectivity between Unipixel agents"""
    
    def __init__(self, config: ConnectorConfig):
        self.config = config
        self.agents: List[UnipixelAgent] = []
        self.connection_matrix: np.ndarray = np.array([])
        self.harmony_cache: Dict[Tuple[int, int], float] = {}
        self.update_counter: int = 0
        
    def add_agent(self, agent: UnipixelAgent) -> None:
        """Add new agent to network with optimal connections"""
        self.agents.append(agent)
        self._update_connection_matrix()
        self._optimize_connections(len(self.agents) - 1)
        
    def update(self) -> Dict:
        """Update network topology and measure coherence"""
        self.update_counter += 1
        
        if self.update_counter >= self.config.topology_update_interval:
            self._optimize_global_topology()
            self.update_counter = 0
            
        return self._compute_network_metrics()
    
    def _update_connection_matrix(self) -> None:
        """Update connection matrix dimensions"""
        n = len(self.agents)
        new_matrix = np.zeros((n, n))
        
        if self.connection_matrix.size > 0:
            old_n = self.connection_matrix.shape[0]
            new_matrix[:old_n, :old_n] = self.connection_matrix
            
        self.connection_matrix = new_matrix
        
    def _optimize_connections(self, agent_idx: int) -> None:
        """Optimize connections for specific agent using category theory"""
        agent = self.agents[agent_idx]
        potential_connections = []
        
        # Evaluate potential connections using categorical harmony
        for i, other_agent in enumerate(self.agents):
            if i != agent_idx:
                harmony = self._compute_categorical_harmony(agent, other_agent)
                self.harmony_cache[(agent_idx, i)] = harmony
                if harmony > self.config.harmony_threshold:
                    potential_connections.append((harmony, i))
        
        # Sort by harmony and establish optimal connections
        potential_connections.sort(reverse=True)
        for harmony, other_idx in potential_connections[:self.config.max_connections]:
            if len(agent.connections) < self.config.max_connections:
                if agent.connect(self.agents[other_idx]):
                    self.connection_matrix[agent_idx, other_idx] = harmony
                    self.connection_matrix[other_idx, agent_idx] = harmony
    
    def _optimize_global_topology(self) -> None:
        """Optimize global network topology using minimum spanning tree and free energy"""
        n = len(self.agents)
        if n < 2:
            return
            
        # Compute harmony-based distance matrix
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                harmony = self._compute_categorical_harmony(self.agents[i], self.agents[j])
                distance = 1.0 - harmony
                distances[i, j] = distance
                distances[j, i] = distance
                
        # Find minimum spanning tree as base topology
        mst = minimum_spanning_tree(distances).toarray()
        
        # Add additional high-harmony connections up to limit
        for i in range(n):
            current_connections = np.sum(mst[i] > 0)
            if current_connections < self.config.max_connections:
                # Get harmony-sorted potential connections
                potentials = [(self.harmony_cache.get((i,j), 0.0), j) 
                            for j in range(n) if j != i and mst[i,j] == 0]
                potentials.sort(reverse=True)
                
                # Add best connections
                for harmony, j in potentials:
                    if (harmony > self.config.harmony_threshold and 
                        current_connections < self.config.max_connections):
                        mst[i,j] = harmony
                        mst[j,i] = harmony
                        current_connections += 1
        
        # Update actual connections
        self._apply_topology_update(mst)
    
    def _apply_topology_update(self, new_topology: np.ndarray) -> None:
        """Apply new topology while maintaining categorical coherence"""
        changes = new_topology != self.connection_matrix
        
        if not np.any(changes):
            return
            
        # Apply changes gradually to maintain coherence
        for i in range(len(self.agents)):
            for j in range(i+1, len(self.agents)):
                if changes[i,j]:
                    if new_topology[i,j] > 0:
                        # Add connection
                        if self.agents[i].connect(self.agents[j]):
                            self.connection_matrix[i,j] = new_topology[i,j]
                            self.connection_matrix[j,i] = new_topology[i,j]
                    else:
                        # Remove connection if it reduces coherence
                        self._remove_connection(i, j)
    
    def _compute_categorical_harmony(self, agent1: UnipixelAgent, 
                                  agent2: UnipixelAgent) -> float:
        """Compute harmony between agents using categorical structure"""
        # Create natural transformation between agent categories
        transformation = NaturalTransformation(
            agent1.category.functors[0],
            agent2.category.functors[0]
        )
        
        # Add morphism components
        morphism = NeuralMorphism(agent1.config.dimensions, agent2.config.dimensions)
        transformation.add_component(agent1.neural_object, morphism)
        
        # Compute harmony based on naturality and state alignment
        naturality = float(transformation.is_natural())
        state_harmony = float(np.abs(np.corrcoef(agent1.state, agent2.state)[0,1]))
        
        return (naturality + state_harmony) / 2
    
    def _remove_connection(self, i: int, j: int) -> None:
        """Remove connection if it maintains coherence"""
        # Store original connection state
        original_matrix = self.connection_matrix.copy()
        
        # Temporarily remove connection
        self.connection_matrix[i,j] = 0
        self.connection_matrix[j,i] = 0
        
        # Check if removal maintains coherence
        if self._verify_network_coherence():
            # Remove connection from agents
            self.agents[i].connections.remove(self.agents[j])
            self.agents[j].connections.remove(self.agents[i])
        else:
            # Restore connection
            self.connection_matrix = original_matrix
    
    def _verify_network_coherence(self) -> bool:
        """Verify network maintains required coherence level"""
        metrics = self._compute_network_metrics()
        return metrics['global_coherence'] >= self.config.min_coherence
    
    def _compute_network_metrics(self) -> Dict:
        """Compute comprehensive network metrics"""
        n = len(self.agents)
        if n < 2:
            return {
                'global_coherence': 1.0,
                'connection_density': 0.0,
                'average_harmony': 0.0
            }
            
        # Compute metrics
        density = np.sum(self.connection_matrix > 0) / (n * (n-1))
        harmony = np.mean(self.connection_matrix[self.connection_matrix > 0])
        
        # Compute global coherence using categorical structure
        coherence_scores = []
        for agent in self.agents:
            coherence_scores.append(float(agent.category.verify_coherence()))
        global_coherence = np.mean(coherence_scores)
        
        return {
            'global_coherence': global_coherence,
            'connection_density': density,
            'average_harmony': harmony
        } 