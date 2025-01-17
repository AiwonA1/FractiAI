"""
FractalFieldTheory.py

Implements quantum field theoretic principles for pattern evolution and resonance,
enabling long-range correlations and emergent collective behavior through field interactions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy.fft import fft2, ifft2
from scipy.special import gamma
from Methods.CategoryTheory import FractalCategory, NeuralObject
from Methods.FreeEnergyPrinciple import FreeEnergyMinimizer
from Methods.FractalResonance import ResonanceConfig

logger = logging.getLogger(__name__)

@dataclass
class FieldConfig:
    """Configuration for fractal field theory"""
    field_dimension: int = 64
    coupling_strength: float = 0.1
    quantum_precision: float = 0.01
    correlation_length: float = 1.0
    dissipation_rate: float = 0.05
    field_resolution: int = 32
    vacuum_energy: float = 0.01

class QuantumField:
    """Quantum field implementation for pattern dynamics"""
    
    def __init__(self, config: FieldConfig):
        self.config = config
        self.field_state = np.zeros((config.field_dimension, config.field_dimension), 
                                  dtype=np.complex128)
        self.momentum_state = np.zeros_like(self.field_state)
        self.vacuum_state = self._initialize_vacuum()
        
    def _initialize_vacuum(self) -> np.ndarray:
        """Initialize quantum vacuum state"""
        k_x = np.fft.fftfreq(self.config.field_dimension)
        k_y = np.fft.fftfreq(self.config.field_dimension)
        kx, ky = np.meshgrid(k_x, k_y)
        k_squared = kx**2 + ky**2
        
        # Zero-point energy fluctuations
        vacuum = np.sqrt(self.config.vacuum_energy / (2 * np.sqrt(k_squared + 1e-6)))
        vacuum *= np.exp(2j * np.pi * np.random.rand(*k_squared.shape))
        return vacuum
    
    def evolve(self, dt: float) -> np.ndarray:
        """Evolve quantum field state"""
        # Update momentum
        self.momentum_state += -self._compute_potential_gradient() * dt
        
        # Update field
        self.field_state += self.momentum_state * dt
        
        # Add quantum fluctuations
        self.field_state += np.sqrt(dt) * self.vacuum_state * np.random.randn(
            *self.field_state.shape)
        
        # Apply dissipation
        self.field_state *= (1 - self.config.dissipation_rate * dt)
        self.momentum_state *= (1 - self.config.dissipation_rate * dt)
        
        return np.abs(self.field_state)
    
    def _compute_potential_gradient(self) -> np.ndarray:
        """Compute gradient of field potential"""
        # Nonlinear φ⁴ potential
        return (self.field_state**3 - self.field_state + 
                self.config.coupling_strength * self._compute_laplacian())
    
    def _compute_laplacian(self) -> np.ndarray:
        """Compute Laplacian in Fourier space"""
        field_k = fft2(self.field_state)
        k_x = np.fft.fftfreq(self.config.field_dimension)
        k_y = np.fft.fftfreq(self.config.field_dimension)
        kx, ky = np.meshgrid(k_x, k_y)
        k_squared = kx**2 + ky**2
        return ifft2(-k_squared * field_k)

class FieldCorrelator:
    """Analyzes quantum field correlations"""
    
    def __init__(self, config: FieldConfig):
        self.config = config
        self.correlation_history = []
        
    def compute_correlations(self, field_state: np.ndarray) -> Dict[str, float]:
        """Compute quantum field correlations"""
        # Compute two-point correlation function
        correlation_function = self._compute_two_point_correlation(field_state)
        
        # Compute connected correlator
        connected_correlator = self._compute_connected_correlator(field_state)
        
        # Store in history
        self.correlation_history.append({
            'correlation_length': float(self._extract_correlation_length(
                correlation_function)),
            'connected_strength': float(np.mean(np.abs(connected_correlator)))
        })
        
        return {
            'correlation_function': correlation_function,
            'connected_correlator': connected_correlator,
            'correlation_length': float(self._extract_correlation_length(
                correlation_function))
        }
    
    def _compute_two_point_correlation(self, field: np.ndarray) -> np.ndarray:
        """Compute two-point correlation function"""
        field_k = fft2(field)
        return np.real(ifft2(np.abs(field_k)**2))
    
    def _compute_connected_correlator(self, field: np.ndarray) -> np.ndarray:
        """Compute connected correlation function"""
        mean_field = np.mean(field)
        fluctuations = field - mean_field
        return self._compute_two_point_correlation(fluctuations)
    
    def _extract_correlation_length(self, correlation: np.ndarray) -> float:
        """Extract correlation length from correlation function"""
        # Compute radial average
        center = correlation.shape[0] // 2
        y, x = np.ogrid[-center:center, -center:center]
        r = np.sqrt(x*x + y*y)
        r_bins = np.arange(0, center)
        radial_mean = np.array([np.mean(correlation[r >= r_bin-0.5][r < r_bin+0.5]) 
                               for r_bin in r_bins])
        
        # Fit exponential decay
        valid = radial_mean > 0
        if np.sum(valid) > 1:
            log_corr = np.log(radial_mean[valid])
            distances = r_bins[valid]
            slope = np.polyfit(distances, log_corr, 1)[0]
            return -1.0 / slope if slope < 0 else self.config.correlation_length
        return self.config.correlation_length

class FractalFieldTheory:
    """Manages quantum field theoretic pattern evolution"""
    
    def __init__(self, config: FieldConfig):
        self.config = config
        self.quantum_field = QuantumField(config)
        self.correlator = FieldCorrelator(config)
        self.category = FractalCategory()
        self.fep = FreeEnergyMinimizer(
            state_dim=config.field_dimension,
            obs_dim=config.field_dimension
        )
        
    def inject_pattern(self, pattern: np.ndarray) -> None:
        """Inject pattern into quantum field"""
        # Resize pattern to field dimensions if needed
        if pattern.shape != self.quantum_field.field_state.shape:
            pattern = self._resize_pattern(pattern)
            
        # Create neural object for categorical structure
        pattern_obj = NeuralObject(
            dimension=pattern.size,
            activation='tanh',
            state=pattern.flatten()
        )
        self.category.neural_category.add_object(pattern_obj)
        
        # Inject pattern into field
        self.quantum_field.field_state += pattern
        
    def evolve_field(self, dt: float = 0.1) -> Dict[str, Any]:
        """Evolve quantum field and analyze patterns"""
        # Evolve field
        field_state = self.quantum_field.evolve(dt)
        
        # Analyze correlations
        correlations = self.correlator.compute_correlations(field_state)
        
        # Compute free energy
        free_energy = self.fep.compute_free_energy(field_state.flatten())
        
        return {
            'field_state': field_state,
            'correlations': correlations,
            'free_energy': free_energy,
            'coherence': float(self.category.verify_coherence())
        }
    
    def _resize_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Resize pattern to match field dimensions"""
        from scipy.ndimage import zoom
        
        target_shape = (self.config.field_dimension, self.config.field_dimension)
        zoom_factors = (target_shape[0] / pattern.shape[0],
                       target_shape[1] / pattern.shape[1])
        return zoom(pattern, zoom_factors, order=1) 