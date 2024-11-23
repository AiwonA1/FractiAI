"""
EcoFracti.py

Implements FractiAI principles for ecological system modeling and analysis,
enabling fractal-based understanding of ecosystem dynamics and interactions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy.spatial import Voronoi
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

@dataclass
class EcoConfig:
    """Configuration for ecological system"""
    grid_size: Tuple[int, int] = (100, 100)
    num_species: int = 10
    interaction_range: float = 5.0
    resource_diffusion: float = 0.1
    mutation_rate: float = 0.01
    carrying_capacity: float = 100.0

class Species:
    """Individual species with trait-based interactions"""
    
    def __init__(self, species_id: int, traits: np.ndarray):
        self.species_id = species_id
        self.traits = traits  # Ecological traits
        self.population = np.zeros(1)  # Population distribution
        self.interactions = {}  # {species_id: interaction_strength}
        self.fitness_history = []
        
    def compute_fitness(self, resources: np.ndarray, 
                       competitors: Dict[int, float]) -> float:
        """Compute species fitness based on resources and competition"""
        # Resource utilization
        resource_fitness = np.sum(self.traits * resources)
        
        # Competition effects
        competition = sum(
            strength * self.compute_competition(other_id)
            for other_id, strength in competitors.items()
        )
        
        fitness = resource_fitness * (1 - competition)
        self.fitness_history.append(fitness)
        
        return float(fitness)
    
    def compute_competition(self, other_id: int) -> float:
        """Compute competition strength with another species"""
        if other_id in self.interactions:
            return self.interactions[other_id]
        return 0.0
    
    def mutate(self, rate: float) -> None:
        """Mutate species traits"""
        mutation = np.random.randn(*self.traits.shape) * rate
        self.traits += mutation

class Ecosystem:
    """Ecosystem with fractal-based species interactions"""
    
    def __init__(self, config: EcoConfig):
        self.config = config
        self.species: Dict[int, Species] = {}
        self.resource_grid = np.zeros(config.grid_size)
        self.interaction_matrix = None
        self.spatial_structure = None
        self.initialize_ecosystem()
        
    def initialize_ecosystem(self) -> None:
        """Initialize ecosystem components"""
        # Initialize species
        for i in range(self.config.num_species):
            traits = np.random.randn(4)  # 4 basic ecological traits
            self.species[i] = Species(i, traits)
            
        # Initialize resources
        self._initialize_resources()
        
        # Initialize interactions
        self._initialize_interactions()
        
        # Initialize spatial structure
        self._initialize_spatial_structure()
        
    def _initialize_resources(self) -> None:
        """Initialize resource distribution"""
        # Create fractal-like resource distribution
        self.resource_grid = self._generate_fractal_landscape(
            self.config.grid_size,
            octaves=5
        )
        
    def _initialize_interactions(self) -> None:
        """Initialize species interactions"""
        for i, species in self.species.items():
            for j, other_species in self.species.items():
                if i != j:
                    # Compute interaction strength based on trait similarity
                    similarity = np.abs(
                        np.corrcoef(species.traits, other_species.traits)[0,1]
                    )
                    species.interactions[j] = similarity
                    
        self._update_interaction_matrix()
        
    def _initialize_spatial_structure(self) -> None:
        """Initialize spatial structure of ecosystem"""
        # Create Voronoi tessellation for spatial partitioning
        points = np.random.rand(self.config.num_species, 2)
        self.spatial_structure = Voronoi(points)
        
    def simulate(self, steps: int) -> Dict[str, Any]:
        """Run ecosystem simulation"""
        metrics = []
        population_history = []
        
        for _ in range(steps):
            step_metrics = self._simulate_step()
            metrics.append(step_metrics)
            
            # Record population distributions
            populations = self._get_population_distribution()
            population_history.append(populations)
            
            # Update ecosystem
            self._update_ecosystem()
            
        return self._analyze_simulation(metrics, population_history)
    
    def _simulate_step(self) -> Dict[str, float]:
        """Simulate one step of ecosystem dynamics"""
        species_metrics = {}
        
        # Update each species
        for species_id, species in self.species.items():
            # Get local resources
            local_resources = self._get_local_resources(species_id)
            
            # Get competing species
            competitors = {
                other_id: other.population.mean()
                for other_id, other in self.species.items()
                if other_id != species_id
            }
            
            # Compute fitness
            fitness = species.compute_fitness(local_resources, competitors)
            
            # Update population
            growth_rate = fitness * (1 - species.population.sum() / 
                                   self.config.carrying_capacity)
            species.population += growth_rate
            
            species_metrics[species_id] = {
                'fitness': fitness,
                'population': float(species.population.mean()),
                'growth_rate': float(growth_rate)
            }
            
        metrics = {
            'mean_fitness': float(np.mean([m['fitness'] for m in species_metrics.values()])),
            'total_population': float(np.sum([m['population'] for m in species_metrics.values()])),
            'diversity': self._compute_diversity()
        }
        
        return metrics
    
    def _update_ecosystem(self) -> None:
        """Update ecosystem state"""
        # Diffuse resources
        self._diffuse_resources()
        
        # Apply species mutations
        for species in self.species.values():
            species.mutate(self.config.mutation_rate)
            
        # Update interactions
        self._update_interaction_matrix()
        
    def _diffuse_resources(self) -> None:
        """Apply resource diffusion"""
        # Simple diffusion using convolution
        kernel = np.array([[0.05, 0.2, 0.05],
                          [0.2, 0.0, 0.2],
                          [0.05, 0.2, 0.05]])
        
        from scipy.signal import convolve2d
        self.resource_grid = convolve2d(
            self.resource_grid,
            kernel,
            mode='same',
            boundary='wrap'
        )
        
    def _update_interaction_matrix(self) -> None:
        """Update species interaction matrix"""
        size = self.config.num_species
        rows, cols, data = [], [], []
        
        for i, species in self.species.items():
            for j, strength in species.interactions.items():
                rows.append(i)
                cols.append(j)
                data.append(strength)
                
        self.interaction_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(size, size)
        )
    
    def _get_local_resources(self, species_id: int) -> np.ndarray:
        """Get resources available to a species"""
        # Simplified resource access based on spatial structure
        return np.mean(self.resource_grid)
    
    def _get_population_distribution(self) -> np.ndarray:
        """Get current population distribution"""
        return np.array([s.population.mean() for s in self.species.values()])
    
    def _compute_diversity(self) -> float:
        """Compute ecosystem diversity"""
        populations = self._get_population_distribution()
        total = np.sum(populations)
        if total == 0:
            return 0.0
        proportions = populations / total
        return float(-np.sum(proportions * np.log(proportions + 1e-10)))
    
    def _generate_fractal_landscape(self, size: Tuple[int, int], 
                                  octaves: int) -> np.ndarray:
        """Generate fractal landscape using Diamond-Square algorithm"""
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
                          populations: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze simulation results"""
        return {
            'ecosystem_stability': self._analyze_stability(metrics),
            'population_dynamics': self._analyze_populations(populations),
            'interaction_metrics': self._analyze_interactions()
        }
    
    def _analyze_stability(self, metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Analyze ecosystem stability"""
        diversities = [m['diversity'] for m in metrics]
        populations = [m['total_population'] for m in metrics]
        
        return {
            'diversity_stability': float(1.0 - np.std(diversities) / (np.mean(diversities) + 1e-7)),
            'population_stability': float(1.0 - np.std(populations) / (np.mean(populations) + 1e-7)),
            'mean_diversity': float(np.mean(diversities))
        }
    
    def _analyze_populations(self, populations: List[np.ndarray]) -> Dict[str, float]:
        """Analyze population dynamics"""
        populations = np.array(populations)
        return {
            'population_growth': float(np.mean(np.diff(populations.mean(axis=1)))),
            'species_evenness': float(np.mean([np.std(p) for p in populations])),
            'coexistence_index': float(np.mean(populations > 0, axis=1).mean())
        }
    
    def _analyze_interactions(self) -> Dict[str, float]:
        """Analyze species interactions"""
        return {
            'interaction_density': float(self.interaction_matrix.nnz / 
                                      (self.config.num_species ** 2)),
            'mean_interaction': float(np.mean(self.interaction_matrix.data)),
            'interaction_variance': float(np.var(self.interaction_matrix.data))
        }