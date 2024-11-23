"""
CrossDimensionalInference.py

Implements cross-dimensional active inference for FractiAI, enabling inference and 
pattern matching across different dimensions and scales through fractal patterns.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy.stats import entropy
from scipy.special import softmax
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class DimensionConfig:
    """Configuration for cross-dimensional inference"""
    dimensions: List[str] = None  # List of dimension names
    dim_sizes: Dict[str, int] = None  # Size of each dimension
    shared_dim: int = 64  # Shared representation dimension
    scale_levels: int = 5  # Number of scale levels
    fractal_depth: int = 3
    learning_rate: float = 0.01

    def __post_init__(self):
        if self.dimensions is None:
            self.dimensions = ['organic', 'inorganic', 'abstract']
        if self.dim_sizes is None:
            self.dim_sizes = {dim: 32 for dim in self.dimensions}

class DimensionalSpace:
    """Represents a dimensional space with fractal patterns"""
    
    def __init__(self, name: str, size: int, shared_dim: int):
        self.name = name
        self.size = size
        self.shared_dim = shared_dim
        
        # Initialize transformations
        scale = np.sqrt(2.0 / (size + shared_dim))
        self.to_shared = np.random.randn(shared_dim, size) * scale
        self.from_shared = np.random.randn(size, shared_dim) * scale
        
        # Initialize patterns and history
        self.patterns = []
        self.transformations = []
        
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode dimension-specific data to shared space"""
        shared = np.dot(self.to_shared, data)
        self.patterns.append(data)
        return shared
        
    def decode(self, shared: np.ndarray) -> np.ndarray:
        """Decode from shared space to dimension-specific space"""
        decoded = np.dot(self.from_shared, shared)
        self.transformations.append({
            'shared': shared,
            'decoded': decoded
        })
        return decoded
        
    def adapt(self, error: np.ndarray, learning_rate: float) -> None:
        """Adapt transformations based on error"""
        if self.patterns:
            # Update transformations
            self.to_shared -= learning_rate * np.outer(error, self.patterns[-1])
            self.from_shared -= learning_rate * np.outer(error, error)
            
    def get_metrics(self) -> Dict[str, float]:
        """Get dimensional space metrics"""
        if not self.patterns or not self.transformations:
            return {}
            
        return {
            'pattern_complexity': self._compute_pattern_complexity(),
            'transformation_stability': self._compute_transformation_stability(),
            'representation_quality': self._compute_representation_quality()
        }
        
    def _compute_pattern_complexity(self) -> float:
        """Compute complexity of patterns"""
        patterns = np.array(self.patterns)
        return float(entropy(softmax(patterns.mean(axis=0))))
        
    def _compute_transformation_stability(self) -> float:
        """Compute stability of transformations"""
        if len(self.transformations) < 2:
            return 1.0
            
        shared_states = [t['shared'] for t in self.transformations]
        diffs = np.diff(shared_states, axis=0)
        return float(1.0 - np.std(diffs) / (np.mean(np.abs(diffs)) + 1e-7))
        
    def _compute_representation_quality(self) -> float:
        """Compute quality of shared representations"""
        if not self.transformations:
            return 0.0
            
        # Compare original patterns with reconstructions
        errors = []
        for pattern, transform in zip(self.patterns, self.transformations):
            error = np.mean((pattern - transform['decoded']) ** 2)
            errors.append(error)
            
        return float(1.0 - np.mean(errors))

class ScaleSpace:
    """Manages multi-scale representations"""
    
    def __init__(self, base_size: int, num_levels: int):
        self.base_size = base_size
        self.num_levels = num_levels
        self.scales = [base_size // (2**i) for i in range(num_levels)]
        self.scale_patterns = {scale: [] for scale in self.scales}
        
    def decompose(self, pattern: np.ndarray) -> Dict[int, np.ndarray]:
        """Decompose pattern into multiple scales"""
        decomposition = {}
        current = pattern
        
        for scale in self.scales:
            # Downsample to scale
            scaled = self._rescale(current, scale)
            decomposition[scale] = scaled
            self.scale_patterns[scale].append(scaled)
            current = scaled
            
        return decomposition
        
    def reconstruct(self, decomposition: Dict[int, np.ndarray]) -> np.ndarray:
        """Reconstruct pattern from multi-scale decomposition"""
        reconstruction = decomposition[min(self.scales)]
        
        for scale in sorted(self.scales)[1:]:
            upscaled = self._rescale(reconstruction, scale)
            reconstruction = upscaled
            
        return reconstruction
        
    def _rescale(self, pattern: np.ndarray, target_size: int) -> np.ndarray:
        """Rescale pattern to target size"""
        if len(pattern) == target_size:
            return pattern
            
        # Simple averaging for downscaling
        if len(pattern) > target_size:
            ratio = len(pattern) // target_size
            return np.mean(pattern.reshape(-1, ratio), axis=1)
            
        # Simple interpolation for upscaling
        ratio = target_size // len(pattern)
        return np.repeat(pattern, ratio)
        
    def get_scale_metrics(self) -> Dict[int, Dict[str, float]]:
        """Get metrics for each scale"""
        metrics = {}
        
        for scale, patterns in self.scale_patterns.items():
            if patterns:
                patterns = np.array(patterns)
                metrics[scale] = {
                    'complexity': float(entropy(softmax(patterns.mean(axis=0)))),
                    'stability': float(1.0 - np.std(patterns) / 
                                    (np.mean(np.abs(patterns)) + 1e-7)),
                    'pattern_count': len(patterns)
                }
                
        return metrics

class CrossDimensionalInference:
    """Cross-dimensional active inference system"""
    
    def __init__(self, config: DimensionConfig):
        self.config = config
        self.dimensions: Dict[str, DimensionalSpace] = {}
        self.scale_spaces: Dict[str, ScaleSpace] = {}
        self.shared_patterns = []
        self.cross_mappings = {}
        self.initialize_spaces()
        
    def initialize_spaces(self) -> None:
        """Initialize dimensional and scale spaces"""
        # Create dimensional spaces
        for dim_name in self.config.dimensions:
            dim_size = self.config.dim_sizes[dim_name]
            self.dimensions[dim_name] = DimensionalSpace(
                dim_name,
                dim_size,
                self.config.shared_dim
            )
            
            # Create scale space for dimension
            self.scale_spaces[dim_name] = ScaleSpace(
                dim_size,
                self.config.scale_levels
            )
            
    def process(self, inputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Process inputs from multiple dimensions"""
        shared_states = {}
        scale_decompositions = {}
        
        # Process each dimension
        for dim_name, input_data in inputs.items():
            if dim_name in self.dimensions:
                # Multi-scale decomposition
                decomposition = self.scale_spaces[dim_name].decompose(input_data)
                scale_decompositions[dim_name] = decomposition
                
                # Encode to shared space
                shared = self.dimensions[dim_name].encode(input_data)
                shared_states[dim_name] = shared
                
        # Store shared patterns
        self.shared_patterns.append(shared_states)
        
        # Update cross-dimensional mappings
        self._update_cross_mappings(shared_states)
        
        # Generate cross-dimensional predictions
        predictions = self._generate_predictions(shared_states)
        
        return {
            'shared_states': shared_states,
            'predictions': predictions,
            'scale_decompositions': scale_decompositions,
            'metrics': self._compute_metrics()
        }
        
    def _update_cross_mappings(self, shared_states: Dict[str, np.ndarray]) -> None:
        """Update mappings between dimensions"""
        for dim1 in self.config.dimensions:
            for dim2 in self.config.dimensions:
                if dim1 < dim2:  # Avoid duplicates
                    if dim1 in shared_states and dim2 in shared_states:
                        mapping_key = f"{dim1}->{dim2}"
                        if mapping_key not in self.cross_mappings:
                            self.cross_mappings[mapping_key] = []
                            
                        self.cross_mappings[mapping_key].append({
                            'source': shared_states[dim1],
                            'target': shared_states[dim2]
                        })
                        
    def _generate_predictions(self, 
                            shared_states: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Generate predictions for each dimension"""
        predictions = {}
        
        for dim_name, dimension in self.dimensions.items():
            if dim_name in shared_states:
                # Generate prediction from shared state
                prediction = dimension.decode(shared_states[dim_name])
                predictions[dim_name] = prediction
                
                # Adapt dimension based on prediction error
                if dim_name in shared_states:
                    error = shared_states[dim_name] - prediction
                    dimension.adapt(error, self.config.learning_rate)
                    
        return predictions
        
    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute comprehensive metrics"""
        return {
            'dimensional_metrics': {
                dim_name: space.get_metrics()
                for dim_name, space in self.dimensions.items()
            },
            'scale_metrics': {
                dim_name: space.get_scale_metrics()
                for dim_name, space in self.scale_spaces.items()
            },
            'cross_dimensional_metrics': self._compute_cross_dimensional_metrics(),
            'integration_metrics': self._compute_integration_metrics()
        }
        
    def _compute_cross_dimensional_metrics(self) -> Dict[str, float]:
        """Compute metrics for cross-dimensional relationships"""
        metrics = {}
        
        for mapping_key, mappings in self.cross_mappings.items():
            if mappings:
                correlations = []
                for mapping in mappings:
                    corr = np.corrcoef(
                        mapping['source'],
                        mapping['target']
                    )[0,1]
                    correlations.append(abs(corr))
                    
                metrics[mapping_key] = {
                    'correlation': float(np.mean(correlations)),
                    'stability': float(1.0 - np.std(correlations)),
                    'mapping_count': len(mappings)
                }
                
        return metrics
        
    def _compute_integration_metrics(self) -> Dict[str, float]:
        """Compute metrics about dimensional integration"""
        if not self.shared_patterns:
            return {}
            
        # Compute integration across dimensions
        shared_states = np.array([
            [pattern[dim] for dim in self.config.dimensions]
            for pattern in self.shared_patterns
        ])
        
        return {
            'integration_strength': float(np.mean([
                np.abs(np.corrcoef(shared_states[:, i], shared_states[:, j])[0,1])
                for i in range(len(self.config.dimensions))
                for j in range(i+1, len(self.config.dimensions))
            ])),
            'dimensional_coherence': float(1.0 - np.std(shared_states) / 
                                        (np.mean(np.abs(shared_states)) + 1e-7)),
            'pattern_complexity': float(entropy(softmax(shared_states.mean(axis=0))))
        } 