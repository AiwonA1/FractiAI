"""
NeuroFracti.py

Implements FractiAI principles for neural network optimization and brain-inspired
computing, enabling fractal-based neural architectures and learning processes.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

@dataclass
class NeuroConfig:
    """Configuration for neural system"""
    num_neurons: int = 1000
    connection_density: float = 0.1
    plasticity_rate: float = 0.01
    firing_threshold: float = 0.5
    refractory_period: int = 5
    dimension: int = 3  # Spatial dimension for neuron placement

class Neuron:
    """Individual neuron with fractal-based dynamics"""
    
    def __init__(self, neuron_id: int, position: np.ndarray):
        self.neuron_id = neuron_id
        self.position = position
        self.connections = {}  # {target_id: weight}
        self.state = 0.0
        self.potential = 0.0
        self.refractory_count = 0
        self.spike_history = []
        
    def update(self, inputs: Dict[int, float], config: NeuroConfig) -> float:
        """Update neuron state and generate output"""
        if self.refractory_count > 0:
            self.refractory_count -= 1
            return 0.0
            
        # Update membrane potential
        self.potential = self.potential * 0.9  # Decay
        for input_id, weight in inputs.items():
            self.potential += weight * self.state
            
        # Check for spike
        if self.potential > config.firing_threshold:
            self.spike()
            return 1.0
            
        return 0.0
    
    def spike(self) -> None:
        """Generate spike and enter refractory period"""
        self.spike_history.append(len(self.spike_history))
        self.potential = 0.0
        self.refractory_count = 5
        
    def adapt_connections(self, target_activity: Dict[int, float], 
                         config: NeuroConfig) -> None:
        """Adapt connection weights based on activity"""
        for target_id, activity in target_activity.items():
            if target_id in self.connections:
                # Hebbian-like plasticity
                weight_change = config.plasticity_rate * self.state * activity
                self.connections[target_id] += weight_change
                # Normalize weight
                self.connections[target_id] = np.clip(
                    self.connections[target_id], 0, 1
                )

class NeuralNetwork:
    """Neural network with fractal architecture"""
    
    def __init__(self, config: NeuroConfig):
        self.config = config
        self.neurons: Dict[int, Neuron] = {}
        self.activity_history = []
        self.connectivity_matrix = None
        self.initialize_network()
        
    def initialize_network(self) -> None:
        """Initialize neural network"""
        # Create neurons with spatial positions
        for i in range(self.config.num_neurons):
            position = np.random.randn(self.config.dimension)
            self.neurons[i] = Neuron(i, position)
            
        # Initialize connections
        self._initialize_connections()
        
    def _initialize_connections(self) -> None:
        """Initialize network connectivity"""
        positions = np.array([n.position for n in self.neurons.values()])
        distances = cdist(positions, positions)
        
        # Create connections based on distance and density
        for i in range(self.config.num_neurons):
            potential_connections = np.where(
                distances[i] < np.percentile(distances[i], 
                                          self.config.connection_density * 100)
            )[0]
            
            for j in potential_connections:
                if i != j:
                    weight = np.exp(-distances[i,j])
                    self.neurons[i].connections[j] = weight
                    
        # Create sparse connectivity matrix
        self._update_connectivity_matrix()
        
    def simulate(self, steps: int, input_pattern: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run network simulation"""
        activity_patterns = []
        metrics = []
        
        for step in range(steps):
            # Process external input if provided
            if input_pattern is not None:
                self._process_input(input_pattern)
                
            # Update network
            step_metrics = self._simulate_step()
            metrics.append(step_metrics)
            
            # Record activity pattern
            activity = self._get_activity_pattern()
            activity_patterns.append(activity)
            
            # Adapt connections
            self._adapt_network()
            
        return self._analyze_simulation(metrics, activity_patterns)
    
    def _simulate_step(self) -> Dict[str, float]:
        """Simulate one step of network activity"""
        neuron_activities = {}
        
        # Update each neuron
        for neuron_id, neuron in self.neurons.items():
            # Gather inputs from connected neurons
            inputs = {
                input_id: self.neurons[input_id].state * weight
                for input_id, weight in neuron.connections.items()
            }
            
            # Update neuron
            activity = neuron.update(inputs, self.config)
            neuron_activities[neuron_id] = activity
            
        metrics = {
            'mean_activity': float(np.mean(list(neuron_activities.values()))),
            'active_neurons': float(np.sum(list(neuron_activities.values()) > 0)),
            'network_energy': self._compute_network_energy()
        }
        
        self.activity_history.append(metrics)
        return metrics
    
    def _adapt_network(self) -> None:
        """Adapt network connections"""
        activities = {n_id: neuron.state 
                     for n_id, neuron in self.neurons.items()}
        
        for neuron in self.neurons.values():
            neuron.adapt_connections(activities, self.config)
            
        # Update connectivity matrix
        self._update_connectivity_matrix()
    
    def _update_connectivity_matrix(self) -> None:
        """Update sparse connectivity matrix"""
        rows, cols, data = [], [], []
        
        for i, neuron in self.neurons.items():
            for j, weight in neuron.connections.items():
                rows.append(i)
                cols.append(j)
                data.append(weight)
                
        self.connectivity_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(self.config.num_neurons, self.config.num_neurons)
        )
    
    def _get_activity_pattern(self) -> np.ndarray:
        """Get current network activity pattern"""
        return np.array([n.state for n in self.neurons.values()])
    
    def _compute_network_energy(self) -> float:
        """Compute network energy (sum of weighted activities)"""
        activities = self._get_activity_pattern()
        return float(np.sum(self.connectivity_matrix.dot(activities)))
    
    def _analyze_simulation(self, metrics: List[Dict[str, float]], 
                          patterns: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze simulation results"""
        return {
            'network_dynamics': self._analyze_dynamics(metrics),
            'activity_patterns': self._analyze_patterns(patterns),
            'structural_metrics': self._compute_structural_metrics()
        }
    
    def _analyze_dynamics(self, metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Analyze network dynamics"""
        activities = np.array([m['mean_activity'] for m in metrics])
        energies = np.array([m['network_energy'] for m in metrics])
        
        return {
            'activity_stability': float(1.0 - np.std(activities) / (np.mean(activities) + 1e-7)),
            'energy_stability': float(1.0 - np.std(energies) / (np.mean(energies) + 1e-7)),
            'activation_rate': float(np.mean([m['active_neurons'] for m in metrics]) / 
                                   self.config.num_neurons)
        }
    
    def _analyze_patterns(self, patterns: List[np.ndarray]) -> Dict[str, float]:
        """Analyze activity patterns"""
        patterns = np.array(patterns)
        return {
            'pattern_complexity': float(np.mean([np.std(p) for p in patterns])),
            'pattern_diversity': float(np.std([np.mean(p) for p in patterns])),
            'temporal_correlation': self._compute_temporal_correlation(patterns)
        }
    
    def _compute_temporal_correlation(self, patterns: np.ndarray) -> float:
        """Compute temporal correlation in activity patterns"""
        if len(patterns) < 2:
            return 0.0
            
        correlations = []
        for i in range(len(patterns)-1):
            corr = np.corrcoef(patterns[i], patterns[i+1])[0,1]
            correlations.append(abs(corr))
            
        return float(np.mean(correlations))
    
    def _compute_structural_metrics(self) -> Dict[str, float]:
        """Compute structural network metrics"""
        return {
            'connection_density': float(self.connectivity_matrix.nnz / 
                                     (self.config.num_neurons ** 2)),
            'average_weight': float(np.mean(self.connectivity_matrix.data)),
            'weight_variance': float(np.var(self.connectivity_matrix.data))
        } 