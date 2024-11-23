"""
SocialFracti.py

Implements FractiAI principles for social network analysis and interaction modeling,
enabling fractal-based understanding of social dynamics and emergent behaviors.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import networkx as nx
from scipy import stats
import logging

logger = logging.getLogger(__name__)

@dataclass
class SocialConfig:
    """Configuration for social network analysis"""
    max_connections: int = 1000
    influence_threshold: float = 0.6
    harmony_weight: float = 0.8
    interaction_memory: int = 100
    community_threshold: float = 0.7

class SocialAgent:
    """Represents an individual in the social network"""
    
    def __init__(self, agent_id: str, traits: np.ndarray):
        self.agent_id = agent_id
        self.traits = traits  # Personality/behavior traits
        self.connections = {}  # {agent_id: connection_strength}
        self.influence_history = []
        self.state = np.zeros_like(traits)
        
    def interact(self, other: 'SocialAgent') -> float:
        """Interact with another agent"""
        compatibility = self._compute_compatibility(other)
        influence = self._compute_influence(other)
        
        # Update states based on interaction
        self._update_state(other, influence)
        
        return compatibility
    
    def _compute_compatibility(self, other: 'SocialAgent') -> float:
        """Compute compatibility between agents"""
        return float(np.abs(np.corrcoef(self.traits, other.traits)[0,1]))
    
    def _compute_influence(self, other: 'SocialAgent') -> float:
        """Compute influence level between agents"""
        return float(np.mean(np.abs(other.traits - self.traits)))
    
    def _update_state(self, other: 'SocialAgent', influence: float) -> None:
        """Update agent state based on interaction"""
        self.state = (1 - influence) * self.state + influence * other.traits

class SocialNetwork:
    """Models social network dynamics using fractal principles"""
    
    def __init__(self, config: SocialConfig):
        self.config = config
        self.agents: Dict[str, SocialAgent] = {}
        self.interaction_patterns = []
        self.community_structure = None
        
    def add_agent(self, agent_id: str, traits: np.ndarray) -> None:
        """Add new agent to network"""
        self.agents[agent_id] = SocialAgent(agent_id, traits)
        self._update_connections(agent_id)
        
    def simulate_interactions(self, steps: int) -> Dict[str, Any]:
        """Simulate network interactions"""
        metrics = []
        
        for _ in range(steps):
            step_metrics = self._simulate_step()
            metrics.append(step_metrics)
            
        return self._analyze_simulation(metrics)
    
    def analyze_communities(self) -> Dict[str, Any]:
        """Analyze community structure"""
        G = self._create_network_graph()
        communities = self._detect_communities(G)
        
        return {
            'communities': communities,
            'modularity': self._compute_modularity(G, communities),
            'cohesion': self._compute_community_cohesion(communities)
        }
    
    def _update_connections(self, agent_id: str) -> None:
        """Update agent connections"""
        agent = self.agents[agent_id]
        
        for other_id, other_agent in self.agents.items():
            if other_id != agent_id:
                compatibility = agent.interact(other_agent)
                
                if compatibility > self.config.influence_threshold:
                    agent.connections[other_id] = compatibility
                    other_agent.connections[agent_id] = compatibility
    
    def _simulate_step(self) -> Dict[str, float]:
        """Simulate one step of network interactions"""
        interaction_scores = []
        
        for agent_id, agent in self.agents.items():
            for connection_id, strength in agent.connections.items():
                if np.random.random() < strength:
                    interaction = agent.interact(self.agents[connection_id])
                    interaction_scores.append(interaction)
        
        metrics = {
            'mean_interaction': float(np.mean(interaction_scores)),
            'interaction_variance': float(np.var(interaction_scores)),
            'active_connections': len(interaction_scores)
        }
        
        self.interaction_patterns.append(metrics)
        return metrics
    
    def _analyze_simulation(self, metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """Analyze simulation results"""
        return {
            'network_stability': self._compute_stability(metrics),
            'interaction_patterns': self._analyze_patterns(metrics),
            'emergence_metrics': self._compute_emergence(metrics)
        }
    
    def _compute_stability(self, metrics: List[Dict[str, float]]) -> float:
        """Compute network stability"""
        mean_interactions = [m['mean_interaction'] for m in metrics]
        return float(1.0 - np.std(mean_interactions) / (np.mean(mean_interactions) + 1e-7))
    
    def _analyze_patterns(self, metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Analyze interaction patterns"""
        patterns = np.array([m['mean_interaction'] for m in metrics])
        return {
            'periodicity': self._detect_periodicity(patterns),
            'trend': float(np.mean(np.diff(patterns))),
            'complexity': float(stats.entropy(patterns))
        }
    
    def _compute_emergence(self, metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Compute emergent behavior metrics"""
        active_connections = np.array([m['active_connections'] for m in metrics])
        return {
            'connectivity_growth': float(np.mean(np.diff(active_connections))),
            'network_density': float(np.mean(active_connections) / len(self.agents)),
            'interaction_efficiency': float(np.mean([m['mean_interaction'] for m in metrics]))
        }
    
    def _create_network_graph(self) -> nx.Graph:
        """Create NetworkX graph from current state"""
        G = nx.Graph()
        
        for agent_id, agent in self.agents.items():
            G.add_node(agent_id, traits=agent.traits)
            
        for agent_id, agent in self.agents.items():
            for connection_id, strength in agent.connections.items():
                G.add_edge(agent_id, connection_id, weight=strength)
                
        return G
    
    def _detect_communities(self, G: nx.Graph) -> List[List[str]]:
        """Detect communities in network"""
        communities = list(nx.community.greedy_modularity_communities(G))
        return [list(community) for community in communities]
    
    def _compute_modularity(self, G: nx.Graph, communities: List[List[str]]) -> float:
        """Compute network modularity"""
        return float(nx.community.modularity(G, communities))
    
    def _compute_community_cohesion(self, communities: List[List[str]]) -> Dict[str, float]:
        """Compute cohesion metrics for communities"""
        cohesion_metrics = {}
        
        for i, community in enumerate(communities):
            internal_connections = 0
            external_connections = 0
            
            for agent_id in community:
                agent = self.agents[agent_id]
                for connection_id, strength in agent.connections.items():
                    if connection_id in community:
                        internal_connections += strength
                    else:
                        external_connections += strength
                        
            if internal_connections + external_connections > 0:
                cohesion = internal_connections / (internal_connections + external_connections)
            else:
                cohesion = 0.0
                
            cohesion_metrics[f'community_{i}'] = float(cohesion)
            
        return cohesion_metrics
    
    def _detect_periodicity(self, pattern: np.ndarray) -> float:
        """Detect periodic behavior in pattern"""
        if len(pattern) < 4:
            return 0.0
            
        autocorr = np.correlate(pattern, pattern, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks
        peaks = []
        for i in range(1, len(autocorr)-1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                peaks.append(i)
                
        if len(peaks) > 1:
            return float(np.mean(np.diff(peaks)))
        return 0.0
