"""
AntFracti.py

Implements advanced FractiAI principles for ant colony optimization and swarm intelligence,
demonstrating emergent behavior and self-organization through fractal patterns and quantum-inspired dynamics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter
from collections import deque

logger = logging.getLogger(__name__)

@dataclass 
class AntConfig:
    """Configuration for advanced ant colony system"""
    # Colony parameters
    colony_size: int = 100
    sub_colonies: int = 3
    caste_types: int = 4  # Different ant specializations
    
    # Pheromone parameters
    pheromone_types: int = 3  # Multiple pheromone signals
    pheromone_decay: float = 0.1
    pheromone_diffusion: float = 0.05
    pheromone_strength: float = 1.0
    
    # Behavioral parameters
    exploration_rate: float = 0.2
    local_awareness: float = 0.7
    social_influence: float = 0.3
    memory_length: int = 50
    quantum_randomness: float = 0.1
    
    # Environment parameters
    max_steps: int = 1000
    resource_types: int = 3
    terrain_complexity: float = 0.5
    weather_effects: bool = True
    
    # Learning parameters
    adaptation_rate: float = 0.01
    reinforcement_factor: float = 0.2
    collective_memory: bool = True

@dataclass
class AntCaste:
    """Defines specialized ant roles"""
    name: str
    pheromone_sensitivity: float
    exploration_bias: float
    load_capacity: float
    energy_efficiency: float
    communication_range: float

class QuantumPheromoneField:
    """Implements quantum-inspired pheromone dynamics"""
    
    def __init__(self, world_size: Tuple[int, int], n_types: int):
        self.fields = np.zeros((n_types, *world_size), dtype=np.complex128)
        self.coherence = np.ones(n_types)
        self.interference_patterns = np.zeros_like(self.fields)
        
    def update(self, positions: np.ndarray, strengths: np.ndarray, 
              config: AntConfig) -> None:
        """Update quantum pheromone fields"""
        # Apply quantum wave function evolution
        self.fields *= np.exp(1j * config.pheromone_decay)
        
        # Add new pheromone contributions with quantum uncertainty
        for pos, strength in zip(positions, strengths):
            phase = 2 * np.pi * np.random.random()
            quantum_strength = strength * np.exp(1j * phase)
            pos = pos.astype(int)
            if self._valid_position(pos):
                self.fields[:, pos[0], pos[1]] += quantum_strength
                
        # Apply diffusion and decoherence
        self._apply_quantum_effects(config)
        
    def _apply_quantum_effects(self, config: AntConfig) -> None:
        """Apply quantum mechanical effects to pheromone fields"""
        for i in range(len(self.fields)):
            # Quantum diffusion
            self.fields[i] = gaussian_filter(self.fields[i], sigma=config.pheromone_diffusion)
            
            # Decoherence
            self.coherence[i] *= (1 - config.pheromone_decay)
            
            # Calculate interference patterns
            self.interference_patterns[i] = np.abs(self.fields[i])**2
            
    def _valid_position(self, pos: np.ndarray) -> bool:
        """Check if position is within field bounds"""
        return all(0 <= p < s for p, s in zip(pos, self.fields.shape[1:]))

class AntAgent:
    """Individual ant agent with quantum-aware fractal behavior patterns"""
    
    def __init__(self, ant_id: int, position: np.ndarray, caste: AntCaste):
        self.ant_id = ant_id
        self.position = position
        self.caste = caste
        self.path_history = deque(maxlen=50)
        self.quantum_state = np.random.random(3) + 1j * np.random.random(3)
        self.carried_resources = np.zeros(3)
        self.energy = 1.0
        self.experience = {}
        
    def move(self, pheromone_field: QuantumPheromoneField,
            resources: Dict[str, np.ndarray],
            colony_state: Dict[str, Any],
            config: AntConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Execute quantum-influenced movement"""
        self.path_history.append(self.position.copy())
        
        # Compute quantum-classical hybrid direction
        classical_direction = self._compute_classical_direction(
            pheromone_field, resources, colony_state, config)
        quantum_direction = self._compute_quantum_direction(config)
        
        # Combine directions with quantum weighting
        direction = (1 - config.quantum_randomness) * classical_direction + \
                   config.quantum_randomness * quantum_direction
                   
        # Update position with energy constraints
        movement_cost = np.linalg.norm(direction) * (1 / self.caste.energy_efficiency)
        if self.energy >= movement_cost:
            self.position += direction
            self.energy -= movement_cost
            
        # Update quantum state
        self._evolve_quantum_state(config)
        
        return self.position, self._compute_pheromone_contribution(direction, config)
    
    def _compute_classical_direction(self, pheromone_field: QuantumPheromoneField,
                                  resources: Dict[str, np.ndarray],
                                  colony_state: Dict[str, Any],
                                  config: AntConfig) -> np.ndarray:
        """Compute classical movement components"""
        # Get pheromone influence
        pheromone_direction = self._get_pheromone_gradient(pheromone_field)
        
        # Get resource influence
        resource_direction = self._get_resource_direction(resources)
        
        # Get social influence
        social_direction = self._get_social_influence(colony_state)
        
        # Combine influences based on caste and state
        direction = (self.caste.pheromone_sensitivity * pheromone_direction +
                    self.caste.exploration_bias * resource_direction +
                    config.social_influence * social_direction)
        
        return self._normalize_direction(direction)
    
    def _compute_quantum_direction(self, config: AntConfig) -> np.ndarray:
        """Compute quantum-influenced movement component"""
        # Generate quantum random walk
        phase = np.angle(self.quantum_state[0])
        amplitude = np.abs(self.quantum_state[1])
        
        direction = np.array([
            amplitude * np.cos(phase),
            amplitude * np.sin(phase)
        ])
        
        return self._normalize_direction(direction)
    
    def _evolve_quantum_state(self, config: AntConfig) -> None:
        """Evolve internal quantum state"""
        # Apply quantum walk operator
        phase_shift = np.exp(1j * np.pi * config.quantum_randomness)
        self.quantum_state *= phase_shift
        
        for step in range(steps):
            step_metrics = self._simulate_step()
            simulation_metrics.append(step_metrics)
            
            # Update pheromone map
            self._update_pheromones()
            
        return self._analyze_simulation(simulation_metrics)
    
    def _simulate_step(self) -> Dict[str, float]:
        """Simulate one step of colony behavior"""
        positions = []
        pheromone_contributions = []
        
        for ant in self.ants:
            position, pheromone = ant.move(
                self.pheromone_map,
                self.food_sources,
                self.config
            )
            positions.append(position)
            pheromone_contributions.append(pheromone)
            
        metrics = {
            'mean_pheromone': float(np.mean(pheromone_contributions)),
            'colony_spread': self._compute_colony_spread(positions),
            'foraging_efficiency': self._compute_foraging_efficiency(positions)
        }
        
        self.metrics.append(metrics)
        return metrics
    
    def _update_pheromones(self) -> None:
        """Update pheromone map"""
        # Apply decay
        self.pheromone_map *= (1 - self.config.pheromone_decay)
        
        # Add new pheromones
        for ant in self.ants:
            pos = ant.position.astype(int)
            if 0 <= pos[0] < self.world_size[0] and 0 <= pos[1] < self.world_size[1]:
                self.pheromone_map[pos[0], pos[1]] += ant.pheromone_contribution
                
    def _compute_colony_spread(self, positions: List[np.ndarray]) -> float:
        """Compute spatial spread of colony"""
        if not positions:
            return 0.0
            
        positions = np.array(positions)
        centroid = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        return float(np.mean(distances))
    
    def _compute_foraging_efficiency(self, positions: List[np.ndarray]) -> float:
        """Compute foraging efficiency"""
        if not positions or len(self.food_sources) == 0:
            return 0.0
            
        positions = np.array(positions)
        min_distances = np.min(cdist(positions, self.food_sources), axis=1)
        return float(1.0 / (np.mean(min_distances) + 1e-7))
    
    def _analyze_simulation(self, metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """Analyze simulation results"""
        return {
            'colony_behavior': self._analyze_colony_behavior(metrics),
            'pheromone_patterns': self._analyze_pheromone_patterns(),
            'emergence_metrics': self._compute_emergence_metrics(metrics)
        }
    
    def _analyze_colony_behavior(self, metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Analyze collective behavior patterns"""
        spreads = [m['colony_spread'] for m in metrics]
        efficiencies = [m['foraging_efficiency'] for m in metrics]
        
        return {
            'spatial_coherence': float(1.0 - np.std(spreads) / (np.mean(spreads) + 1e-7)),
            'foraging_stability': float(1.0 - np.std(efficiencies) / (np.mean(efficiencies) + 1e-7)),
            'coordination_score': float(np.mean(efficiencies) * np.mean(spreads))
        }
    
    def _analyze_pheromone_patterns(self) -> Dict[str, float]:
        """Analyze pheromone trail patterns"""
        return {
            'pattern_strength': float(np.mean(self.pheromone_map)),
            'pattern_complexity': float(np.std(self.pheromone_map)),
            'trail_coverage': float(np.sum(self.pheromone_map > 0) / self.pheromone_map.size)
        }
    
    def _compute_emergence_metrics(self, metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Compute emergence and self-organization metrics"""
        return {
            'collective_efficiency': float(np.mean([m['foraging_efficiency'] for m in metrics])),
            'organization_level': float(np.mean([m['colony_spread'] for m in metrics])),
            'adaptation_rate': float(np.mean(np.diff([m['mean_pheromone'] for m in metrics])))
        }
