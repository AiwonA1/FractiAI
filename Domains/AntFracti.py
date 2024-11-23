"""
AntFracti.py

Implements FractiAI principles for ant colony optimization and swarm intelligence,
demonstrating emergent behavior and self-organization through fractal patterns.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

@dataclass
class AntConfig:
    """Configuration for ant colony system"""
    colony_size: int = 100
    pheromone_decay: float = 0.1
    pheromone_strength: float = 1.0
    exploration_rate: float = 0.2
    local_awareness: float = 0.7
    max_steps: int = 1000

class AntAgent:
    """Individual ant agent with fractal behavior patterns"""
    
    def __init__(self, ant_id: int, position: np.ndarray):
        self.ant_id = ant_id
        self.position = position
        self.path_history = []
        self.pheromone_contribution = 0.0
        self.state = np.zeros(3)  # Internal state vector
        
    def move(self, pheromone_map: np.ndarray, 
            food_sources: np.ndarray,
            config: AntConfig) -> Tuple[np.ndarray, float]:
        """Move ant based on pheromones and food sources"""
        # Store current position
        self.path_history.append(self.position.copy())
        
        # Compute move direction
        direction = self._compute_direction(
            pheromone_map,
            food_sources,
            config
        )
        
        # Update position
        new_position = self.position + direction
        self.position = new_position
        
        # Compute pheromone contribution
        self.pheromone_contribution = self._compute_pheromone(direction, config)
        
        return new_position, self.pheromone_contribution
    
    def _compute_direction(self, pheromone_map: np.ndarray,
                         food_sources: np.ndarray,
                         config: AntConfig) -> np.ndarray:
        """Compute movement direction"""
        # Get local pheromone information
        local_pheromones = self._get_local_pheromones(pheromone_map)
        
        # Get direction to nearest food
        food_direction = self._get_food_direction(food_sources)
        
        # Combine influences with exploration
        if np.random.random() < config.exploration_rate:
            # Random exploration
            direction = np.random.randn(2)
        else:
            # Weighted combination of pheromones and food direction
            direction = (config.local_awareness * local_pheromones +
                       (1 - config.local_awareness) * food_direction)
            
        # Normalize direction
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm
            
        return direction
    
    def _get_local_pheromones(self, pheromone_map: np.ndarray) -> np.ndarray:
        """Get local pheromone gradient"""
        pos = self.position.astype(int)
        window_size = 3
        
        # Get local window
        x_min = max(0, pos[0] - window_size)
        x_max = min(pheromone_map.shape[0], pos[0] + window_size + 1)
        y_min = max(0, pos[1] - window_size)
        y_max = min(pheromone_map.shape[1], pos[1] + window_size + 1)
        
        local_window = pheromone_map[x_min:x_max, y_min:y_max]
        
        # Compute gradient
        gradient_y, gradient_x = np.gradient(local_window)
        
        return np.array([np.mean(gradient_x), np.mean(gradient_y)])
    
    def _get_food_direction(self, food_sources: np.ndarray) -> np.ndarray:
        """Compute direction to nearest food source"""
        if len(food_sources) == 0:
            return np.zeros(2)
            
        # Compute distances to all food sources
        distances = cdist([self.position], food_sources)
        nearest_idx = np.argmin(distances)
        
        # Get direction to nearest food
        direction = food_sources[nearest_idx] - self.position
        
        # Normalize
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm
            
        return direction
    
    def _compute_pheromone(self, direction: np.ndarray, 
                          config: AntConfig) -> float:
        """Compute pheromone contribution"""
        # Base pheromone on movement efficiency
        movement_efficiency = np.linalg.norm(direction)
        return config.pheromone_strength * movement_efficiency

class AntColony:
    """Ant colony system implementing fractal-based swarm intelligence"""
    
    def __init__(self, config: AntConfig, world_size: Tuple[int, int]):
        self.config = config
        self.world_size = world_size
        self.ants: List[AntAgent] = []
        self.pheromone_map = np.zeros(world_size)
        self.food_sources = np.array([])
        self.metrics = []
        
    def initialize_colony(self, nest_position: np.ndarray) -> None:
        """Initialize ant colony"""
        for i in range(self.config.colony_size):
            position = nest_position + np.random.randn(2) * 0.1
            self.ants.append(AntAgent(i, position))
            
    def add_food_source(self, position: np.ndarray, quantity: float) -> None:
        """Add food source to environment"""
        if self.food_sources.size == 0:
            self.food_sources = position.reshape(1, -1)
        else:
            self.food_sources = np.vstack([self.food_sources, position])
            
    def simulate(self, steps: int) -> Dict[str, Any]:
        """Run colony simulation"""
        simulation_metrics = []
        
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
