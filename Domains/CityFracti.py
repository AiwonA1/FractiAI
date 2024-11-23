"""
CityFracti.py

Implements FractiAI principles for urban systems modeling and optimization,
enabling fractal-based understanding of city dynamics and development patterns.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy.spatial import Voronoi
from scipy.sparse import csr_matrix
import networkx as nx

logger = logging.getLogger(__name__)

@dataclass
class CityConfig:
    """Configuration for urban system"""
    grid_size: Tuple[int, int] = (100, 100)
    num_districts: int = 10
    development_rate: float = 0.01
    connectivity_threshold: float = 0.7
    resource_efficiency: float = 0.8
    sustainability_factor: float = 0.9

class District:
    """Urban district with fractal development patterns"""
    
    def __init__(self, district_id: int, position: np.ndarray):
        self.district_id = district_id
        self.position = position
        self.population = 0.0
        self.resources = {}  # {resource_type: amount}
        self.connections = {}  # {district_id: connection_strength}
        self.development_history = []
        
    def develop(self, resources: Dict[str, float], 
               neighbors: Dict[int, float]) -> float:
        """Develop district based on resources and connections"""
        # Compute development potential
        resource_factor = np.mean(list(resources.values()))
        connectivity_factor = np.mean(list(neighbors.values()))
        
        development = resource_factor * connectivity_factor
        self.population *= (1 + development)
        
        self.development_history.append(development)
        return development
    
    def update_resources(self, new_resources: Dict[str, float]) -> None:
        """Update district resources"""
        for resource_type, amount in new_resources.items():
            if resource_type not in self.resources:
                self.resources[resource_type] = 0
            self.resources[resource_type] += amount

class UrbanSystem:
    """City system with fractal organization"""
    
    def __init__(self, config: CityConfig):
        self.config = config
        self.districts: Dict[int, District] = {}
        self.infrastructure = np.zeros(config.grid_size)
        self.resource_distribution = {}
        self.development_patterns = []
        self.initialize_city()
        
    def initialize_city(self) -> None:
        """Initialize city components"""
        # Create districts
        for i in range(self.config.num_districts):
            position = np.random.rand(2) * self.config.grid_size[0]
            self.districts[i] = District(i, position)
            
        # Initialize resources
        self._initialize_resources()
        
        # Initialize infrastructure
        self._initialize_infrastructure()
        
        # Initialize connections
        self._initialize_connections()
        
    def _initialize_resources(self) -> None:
        """Initialize resource distribution"""
        resource_types = ['energy', 'water', 'land']
        
        for resource in resource_types:
            self.resource_distribution[resource] = self._generate_resource_map()
            
    def _initialize_infrastructure(self) -> None:
        """Initialize infrastructure network"""
        # Create fractal infrastructure pattern
        self.infrastructure = self._generate_infrastructure_pattern()
        
    def _initialize_connections(self) -> None:
        """Initialize district connections"""
        positions = np.array([d.position for d in self.districts.values()])
        distances = np.sqrt(((positions[:, None] - positions) ** 2).sum(axis=2))
        
        for i, district in self.districts.items():
            for j, other in self.districts.items():
                if i != j:
                    # Compute connection strength based on distance
                    strength = np.exp(-distances[i,j] / self.config.grid_size[0])
                    if strength > self.config.connectivity_threshold:
                        district.connections[j] = strength
                        
    def simulate(self, steps: int) -> Dict[str, Any]:
        """Run urban system simulation"""
        metrics = []
        development_history = []
        
        for _ in range(steps):
            step_metrics = self._simulate_step()
            metrics.append(step_metrics)
            
            # Record development patterns
            development = self._get_development_pattern()
            development_history.append(development)
            
            # Update system
            self._update_system()
            
        return self._analyze_simulation(metrics, development_history)
    
    def _simulate_step(self) -> Dict[str, float]:
        """Simulate one step of urban development"""
        district_metrics = {}
        
        # Update each district
        for district_id, district in self.districts.items():
            # Get local resources
            local_resources = self._get_local_resources(district_id)
            
            # Get connected districts
            neighbors = {
                other_id: other.population
                for other_id, other in self.districts.items()
                if other_id in district.connections
            }
            
            # Compute development
            development = district.develop(local_resources, neighbors)
            
            district_metrics[district_id] = {
                'development': development,
                'population': float(district.population),
                'resource_usage': sum(district.resources.values())
            }
            
        metrics = {
            'mean_development': float(np.mean([m['development'] for m in district_metrics.values()])),
            'total_population': float(np.sum([m['population'] for m in district_metrics.values()])),
            'resource_efficiency': self._compute_resource_efficiency()
        }
        
        return metrics
    
    def _update_system(self) -> None:
        """Update urban system state"""
        # Update resources
        self._update_resources()
        
        # Update infrastructure
        self._update_infrastructure()
        
        # Update connections
        self._update_connections()
        
    def _update_resources(self) -> None:
        """Update resource distribution"""
        for resource_type in self.resource_distribution:
            # Apply resource consumption and regeneration
            consumption = self._compute_resource_consumption()
            regeneration = self._compute_resource_regeneration()
            
            self.resource_distribution[resource_type] *= (1 - consumption + regeneration)
            
    def _update_infrastructure(self) -> None:
        """Update infrastructure network"""
        # Apply development effects on infrastructure
        development_impact = self._compute_development_impact()
        self.infrastructure *= (1 + development_impact * self.config.development_rate)
        
    def _update_connections(self) -> None:
        """Update district connections"""
        for district in self.districts.values():
            for other_id in list(district.connections.keys()):
                # Update connection strength based on interaction
                strength = district.connections[other_id]
                interaction = min(district.population, 
                                self.districts[other_id].population)
                
                new_strength = strength * (1 + interaction * self.config.development_rate)
                district.connections[other_id] = new_strength
                
    def _get_local_resources(self, district_id: int) -> Dict[str, float]:
        """Get resources available to a district"""
        district = self.districts[district_id]
        local_resources = {}
        
        for resource_type, distribution in self.resource_distribution.items():
            pos = district.position.astype(int)
            local_resources[resource_type] = np.mean(
                distribution[max(0, pos[0]-2):pos[0]+3,
                           max(0, pos[1]-2):pos[1]+3]
            )
            
        return local_resources
    
    def _get_development_pattern(self) -> np.ndarray:
        """Get current development pattern"""
        pattern = np.zeros(self.config.grid_size)
        
        for district in self.districts.values():
            pos = district.position.astype(int)
            if 0 <= pos[0] < self.config.grid_size[0] and \
               0 <= pos[1] < self.config.grid_size[1]:
                pattern[pos[0], pos[1]] = district.population
                
        return pattern
    
    def _compute_resource_efficiency(self) -> float:
        """Compute system-wide resource efficiency"""
        total_resources = sum(np.sum(dist) for dist in self.resource_distribution.values())
        total_population = sum(d.population for d in self.districts.values())
        
        if total_resources == 0:
            return 0.0
            
        return float(total_population / total_resources)
    
    def _compute_resource_consumption(self) -> float:
        """Compute resource consumption rate"""
        total_population = sum(d.population for d in self.districts.values())
        return float(total_population * (1 - self.config.resource_efficiency))
    
    def _compute_resource_regeneration(self) -> float:
        """Compute resource regeneration rate"""
        return float(self.config.sustainability_factor * 
                    (1 - self._compute_resource_consumption()))
    
    def _compute_development_impact(self) -> float:
        """Compute development impact on infrastructure"""
        total_development = sum(len(d.development_history) 
                              for d in self.districts.values())
        return float(total_development / (self.config.num_districts * 100))
    
    def _generate_resource_map(self) -> np.ndarray:
        """Generate fractal resource distribution"""
        return self._generate_fractal_pattern(
            self.config.grid_size,
            octaves=5
        )
    
    def _generate_infrastructure_pattern(self) -> np.ndarray:
        """Generate fractal infrastructure pattern"""
        return self._generate_fractal_pattern(
            self.config.grid_size,
            octaves=3
        )
    
    def _generate_fractal_pattern(self, size: Tuple[int, int], 
                                octaves: int) -> np.ndarray:
        """Generate fractal pattern using Diamond-Square algorithm"""
        N = max(size)
        size = 2**int(np.ceil(np.log2(N)))
        grid = np.random.randn(size, size)
        
        scale = 1.0
        for _ in range(octaves):
            grid = self._diamond_square_step(grid, scale)
            scale *= 0.5
            
        return grid[:size[0], :size[1]]
    
    def _diamond_square_step(self, grid: np.ndarray, scale: float) -> np.ndarray:
        """Perform one step of Diamond-Square algorithm"""
        size = len(grid)
        half = size // 2
        
        if half < 1:
            return grid
            
        # Diamond step
        for i in range(half, size, size):
            for j in range(half, size, size):
                grid[i,j] = (grid[i-half,j-half] + grid[i-half,j+half] +
                            grid[i+half,j-half] + grid[i+half,j+half]) / 4.0 + \
                           np.random.randn() * scale
                           
        # Square step
        for i in range(0, size, half):
            for j in range((i + half) % size, size, size):
                grid[i,j] = (grid[i-half,j] + grid[i+half,j] +
                            grid[i,j-half] + grid[i,j+half]) / 4.0 + \
                           np.random.randn() * scale
                           
        return grid
    
    def _analyze_simulation(self, metrics: List[Dict[str, float]], 
                          patterns: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze simulation results"""
        return {
            'development_metrics': self._analyze_development(metrics),
            'spatial_patterns': self._analyze_patterns(patterns),
            'system_efficiency': self._compute_system_efficiency(metrics)
        }
    
    def _analyze_development(self, metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Analyze development metrics"""
        developments = [m['mean_development'] for m in metrics]
        populations = [m['total_population'] for m in metrics]
        
        return {
            'development_stability': float(1.0 - np.std(developments) / 
                                        (np.mean(developments) + 1e-7)),
            'population_growth': float(np.mean(np.diff(populations))),
            'development_rate': float(np.mean(developments))
        }
    
    def _analyze_patterns(self, patterns: List[np.ndarray]) -> Dict[str, float]:
        """Analyze spatial development patterns"""
        patterns = np.array(patterns)
        return {
            'spatial_complexity': float(np.mean([np.std(p) for p in patterns])),
            'pattern_evolution': float(np.mean(np.diff(patterns, axis=0).ravel())),
            'density_distribution': float(np.mean([np.sum(p > 0) / p.size 
                                                 for p in patterns]))
        }
    
    def _compute_system_efficiency(self, metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Compute overall system efficiency"""
        return {
            'resource_efficiency': float(np.mean([m['resource_efficiency'] 
                                                for m in metrics])),
            'development_efficiency': float(np.mean([m['mean_development'] / 
                                                   (m['total_population'] + 1e-7)
                                                   for m in metrics])),
            'sustainability_score': float(self.config.sustainability_factor * 
                                       np.mean([m['resource_efficiency'] 
                                              for m in metrics]))
        } 