"""
MultiAgentInference.py

Implements multi-agent active inference for FractiAI, enabling collective belief 
propagation and coordinated actions through fractal patterns.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy.stats import entropy
from scipy.special import softmax
import networkx as nx

logger = logging.getLogger(__name__)

@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent inference system"""
    num_agents: int = 10
    state_dim: int = 32
    action_dim: int = 16
    belief_dim: int = 64
    communication_range: float = 0.5
    learning_rate: float = 0.01
    fractal_depth: int = 3

class InferenceAgent:
    """Individual agent with active inference capabilities"""
    
    def __init__(self, agent_id: str, config: MultiAgentConfig):
        self.agent_id = agent_id
        self.config = config
        
        # Initialize states and beliefs
        self.state = np.zeros(config.state_dim)
        self.belief = np.zeros(config.belief_dim)
        self.action_space = np.zeros(config.action_dim)
        
        # Initialize models
        self.generative_model = np.random.randn(config.belief_dim, config.state_dim)
        self.recognition_model = np.random.randn(config.state_dim, config.belief_dim)
        
        # Initialize connections and history
        self.connections: Dict[str, float] = {}
        self.belief_history = []
        self.action_history = []
        
    def update_belief(self, observation: np.ndarray, 
                     neighbor_beliefs: Dict[str, np.ndarray]) -> None:
        """Update beliefs based on observation and neighbor beliefs"""
        # Generate prediction
        prediction = np.dot(self.generative_model, self.state)
        
        # Compute prediction error
        pred_error = observation - prediction
        
        # Update state using recognition model
        state_update = np.dot(self.recognition_model, pred_error)
        self.state += self.config.learning_rate * state_update
        
        # Incorporate neighbor beliefs
        if neighbor_beliefs:
            mean_neighbor_belief = np.mean(list(neighbor_beliefs.values()), axis=0)
            self.belief = 0.7 * self.belief + 0.3 * mean_neighbor_belief
            
        # Store history
        self.belief_history.append(self.belief.copy())
        
    def select_action(self, goal_state: Optional[np.ndarray] = None) -> np.ndarray:
        """Select action using active inference"""
        if goal_state is None:
            goal_state = np.zeros_like(self.state)
            
        # Compute expected free energy for actions
        action_values = []
        for action in self._generate_actions():
            # Simulate action effect
            next_state = self.state + action
            
            # Compute expected free energy
            ambiguity = self._compute_ambiguity(next_state)
            risk = np.sum((next_state - goal_state) ** 2)
            
            value = -(ambiguity + risk)
            action_values.append(value)
            
        # Select action using softmax
        action_probs = softmax(action_values)
        selected_idx = np.random.choice(len(action_values), p=action_probs)
        selected_action = self._generate_actions()[selected_idx]
        
        # Store action
        self.action_history.append(selected_action)
        
        return selected_action
        
    def _generate_actions(self) -> List[np.ndarray]:
        """Generate possible actions"""
        # Generate base actions
        base_actions = [
            np.random.randn(self.config.action_dim) 
            for _ in range(10)  # Consider 10 possible actions
        ]
        
        # Scale actions
        return [action * 0.1 for action in base_actions]
        
    def _compute_ambiguity(self, state: np.ndarray) -> float:
        """Compute ambiguity of state prediction"""
        prediction = np.dot(self.generative_model, state)
        return float(entropy(softmax(prediction)))
        
    def get_metrics(self) -> Dict[str, float]:
        """Get agent metrics"""
        return {
            'belief_entropy': float(entropy(softmax(self.belief))),
            'action_variance': float(np.var([np.mean(a) for a in self.action_history])),
            'belief_stability': self._compute_belief_stability()
        }
        
    def _compute_belief_stability(self) -> float:
        """Compute stability of beliefs over time"""
        if len(self.belief_history) < 2:
            return 1.0
            
        diffs = np.diff([b.mean() for b in self.belief_history])
        return float(1.0 - np.std(diffs))

class MultiAgentInference:
    """Multi-agent active inference system with fractal patterns"""
    
    def __init__(self, config: MultiAgentConfig):
        self.config = config
        self.agents: Dict[str, InferenceAgent] = {}
        self.communication_graph = nx.Graph()
        self.global_state = {}
        self.initialize_system()
        
    def initialize_system(self) -> None:
        """Initialize multi-agent system"""
        # Create agents
        for i in range(self.config.num_agents):
            agent_id = f"AGENT_{i}"
            self.agents[agent_id] = InferenceAgent(agent_id, self.config)
            
        # Initialize communication network
        self._initialize_communication()
        
    def _initialize_communication(self) -> None:
        """Initialize communication network between agents"""
        # Create nodes
        for agent_id in self.agents:
            self.communication_graph.add_node(agent_id)
            
        # Create edges based on communication range
        for i, agent_id in enumerate(self.agents):
            for j, other_id in enumerate(self.agents):
                if i < j:  # Avoid duplicate connections
                    if np.random.random() < self.config.communication_range:
                        self.communication_graph.add_edge(agent_id, other_id)
                        self.agents[agent_id].connections[other_id] = 1.0
                        self.agents[other_id].connections[agent_id] = 1.0
                        
    def process_observations(self, observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Process observations through multi-agent system"""
        actions = {}
        
        # Update beliefs
        for agent_id, agent in self.agents.items():
            if agent_id in observations:
                # Get neighbor beliefs
                neighbor_beliefs = self._get_neighbor_beliefs(agent_id)
                
                # Update agent belief
                agent.update_belief(observations[agent_id], neighbor_beliefs)
                
                # Select action
                actions[agent_id] = agent.select_action()
                
        # Update global state
        self._update_global_state()
        
        return actions
        
    def _get_neighbor_beliefs(self, agent_id: str) -> Dict[str, np.ndarray]:
        """Get beliefs of connected neighbors"""
        if agent_id not in self.communication_graph:
            return {}
            
        return {
            neighbor: self.agents[neighbor].belief
            for neighbor in self.communication_graph.neighbors(agent_id)
        }
        
    def _update_global_state(self) -> None:
        """Update global system state"""
        self.global_state = {
            'mean_belief': np.mean([agent.belief for agent in self.agents.values()], axis=0),
            'belief_variance': np.var([agent.belief for agent in self.agents.values()], axis=0),
            'network_density': nx.density(self.communication_graph),
            'agent_metrics': {
                agent_id: agent.get_metrics()
                for agent_id, agent in self.agents.items()
            }
        }
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        return {
            'global_state': self.global_state,
            'network_metrics': self._compute_network_metrics(),
            'collective_metrics': self._compute_collective_metrics()
        }
        
    def _compute_network_metrics(self) -> Dict[str, float]:
        """Compute network-level metrics"""
        return {
            'clustering': float(nx.average_clustering(self.communication_graph)),
            'path_length': float(nx.average_shortest_path_length(self.communication_graph)),
            'centrality': self._compute_centrality()
        }
        
    def _compute_centrality(self) -> float:
        """Compute average node centrality"""
        centralities = nx.eigenvector_centrality_numpy(self.communication_graph)
        return float(np.mean(list(centralities.values())))
        
    def _compute_collective_metrics(self) -> Dict[str, float]:
        """Compute collective behavior metrics"""
        belief_entropies = [m['belief_entropy'] 
                          for m in self.global_state['agent_metrics'].values()]
        action_variances = [m['action_variance']
                          for m in self.global_state['agent_metrics'].values()]
        
        return {
            'belief_coherence': float(1.0 - np.std(belief_entropies)),
            'action_coordination': float(1.0 - np.mean(action_variances)),
            'system_stability': float(np.mean([m['belief_stability']
                                             for m in self.global_state['agent_metrics'].values()]))
        } 