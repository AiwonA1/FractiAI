"""
CrossDomainInference.py

Implements cross-domain active inference for FractiAI, enabling inference and 
transfer across different domains and modalities through fractal patterns.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy.stats import entropy
from scipy.special import softmax

logger = logging.getLogger(__name__)

@dataclass
class DomainConfig:
    """Configuration for cross-domain inference"""
    domain_dims: Dict[str, int] = None  # Dimensions for each domain
    shared_dim: int = 64  # Shared latent dimension
    adaptation_rate: float = 0.01
    transfer_strength: float = 0.7
    fractal_depth: int = 3

    def __post_init__(self):
        if self.domain_dims is None:
            self.domain_dims = {
                'organic': 32,
                'inorganic': 32,
                'abstract': 32
            }

class DomainAdapter:
    """Adapts between domain-specific and shared representations"""
    
    def __init__(self, domain_name: str, domain_dim: int, shared_dim: int):
        self.domain_name = domain_name
        self.domain_dim = domain_dim
        self.shared_dim = shared_dim
        
        # Initialize transformation matrices
        scale = np.sqrt(2.0 / (domain_dim + shared_dim))
        self.encoder = np.random.randn(shared_dim, domain_dim) * scale
        self.decoder = np.random.randn(domain_dim, shared_dim) * scale
        
        # Initialize domain-specific patterns
        self.patterns = []
        self.adaptation_history = []
        
    def encode(self, domain_data: np.ndarray) -> np.ndarray:
        """Encode domain-specific data to shared space"""
        shared = np.dot(self.encoder, domain_data)
        self.patterns.append(domain_data)
        return shared
        
    def decode(self, shared_data: np.ndarray) -> np.ndarray:
        """Decode shared representation to domain-specific space"""
        return np.dot(self.decoder, shared_data)
        
    def adapt(self, error: np.ndarray, learning_rate: float) -> None:
        """Adapt domain mappings based on reconstruction error"""
        # Update transformation matrices
        self.encoder -= learning_rate * np.outer(error, self.patterns[-1])
        self.decoder -= learning_rate * np.outer(error, error)
        
        # Store adaptation
        self.adaptation_history.append(np.mean(np.abs(error)))
        
    def get_domain_patterns(self) -> Dict[str, float]:
        """Get domain-specific pattern metrics"""
        if not self.patterns:
            return {}
            
        patterns = np.array(self.patterns)
        return {
            'complexity': float(entropy(softmax(patterns.mean(axis=0)))),
            'stability': float(1.0 - np.std(patterns) / (np.mean(np.abs(patterns)) + 1e-7)),
            'adaptation': float(np.mean(self.adaptation_history) if self.adaptation_history else 0)
        }

class CrossDomainInference:
    """Cross-domain active inference system with fractal patterns"""
    
    def __init__(self, config: DomainConfig):
        self.config = config
        self.adapters: Dict[str, DomainAdapter] = {}
        self.shared_patterns = []
        self.transfer_history = []
        self.initialize_adapters()
        
    def initialize_adapters(self) -> None:
        """Initialize domain adapters"""
        for domain, dim in self.config.domain_dims.items():
            self.adapters[domain] = DomainAdapter(
                domain,
                dim,
                self.config.shared_dim
            )
            
    def process_domain(self, domain: str, 
                      data: np.ndarray) -> Dict[str, Any]:
        """Process data from specific domain"""
        if domain not in self.adapters:
            raise ValueError(f"Unknown domain: {domain}")
            
        # Encode to shared space
        adapter = self.adapters[domain]
        shared = adapter.encode(data)
        
        # Store shared pattern
        self.shared_patterns.append(shared)
        
        # Transfer to other domains
        transfers = self._transfer_to_domains(shared, exclude=domain)
        
        # Compute reconstruction
        reconstruction = adapter.decode(shared)
        
        # Compute error and adapt
        error = data - reconstruction
        adapter.adapt(error, self.config.adaptation_rate)
        
        return {
            'shared': shared,
            'reconstruction': reconstruction,
            'transfers': transfers,
            'metrics': self._compute_metrics(domain, error)
        }
        
    def _transfer_to_domains(self, shared: np.ndarray, 
                           exclude: str) -> Dict[str, np.ndarray]:
        """Transfer shared representation to other domains"""
        transfers = {}
        
        for domain, adapter in self.adapters.items():
            if domain != exclude:
                # Apply transfer with strength parameter
                domain_transfer = adapter.decode(
                    shared * self.config.transfer_strength
                )
                transfers[domain] = domain_transfer
                
                # Store transfer
                self.transfer_history.append({
                    'from': exclude,
                    'to': domain,
                    'pattern': domain_transfer
                })
                
        return transfers
        
    def _compute_metrics(self, domain: str, 
                        error: np.ndarray) -> Dict[str, float]:
        """Compute domain-specific metrics"""
        adapter = self.adapters[domain]
        
        return {
            'reconstruction_error': float(np.mean(np.abs(error))),
            'pattern_metrics': adapter.get_domain_patterns(),
            'transfer_efficiency': self._compute_transfer_efficiency(domain)
        }
        
    def _compute_transfer_efficiency(self, domain: str) -> float:
        """Compute efficiency of transfers from domain"""
        if not self.transfer_history:
            return 0.0
            
        # Get recent transfers from domain
        relevant_transfers = [
            t for t in self.transfer_history[-100:]  # Look at recent history
            if t['from'] == domain
        ]
        
        if not relevant_transfers:
            return 0.0
            
        # Compute transfer pattern stability
        patterns = np.array([t['pattern'] for t in relevant_transfers])
        return float(1.0 - np.std(patterns) / (np.mean(np.abs(patterns)) + 1e-7))
        
    def get_shared_metrics(self) -> Dict[str, Any]:
        """Get metrics about shared representations"""
        if not self.shared_patterns:
            return {}
            
        patterns = np.array(self.shared_patterns)
        
        return {
            'shared_complexity': float(entropy(softmax(patterns.mean(axis=0)))),
            'shared_stability': float(1.0 - np.std(patterns) / 
                                   (np.mean(np.abs(patterns)) + 1e-7)),
            'domain_coherence': self._compute_domain_coherence()
        }
        
    def _compute_domain_coherence(self) -> float:
        """Compute coherence between domain representations"""
        coherence_scores = []
        
        for d1 in self.adapters:
            for d2 in self.adapters:
                if d1 < d2:  # Avoid duplicates
                    # Compare domain patterns
                    patterns1 = self.adapters[d1].patterns
                    patterns2 = self.adapters[d2].patterns
                    
                    if patterns1 and patterns2:
                        # Use latest patterns
                        correlation = np.corrcoef(
                            patterns1[-1],
                            patterns2[-1]
                        )[0,1]
                        coherence_scores.append(abs(correlation))
                        
        return float(np.mean(coherence_scores)) if coherence_scores else 0.0
        
    def analyze_transfers(self) -> Dict[str, Dict[str, float]]:
        """Analyze transfer patterns between domains"""
        transfer_metrics = {}
        
        for source in self.adapters:
            transfer_metrics[source] = {}
            
            for target in self.adapters:
                if source != target:
                    # Get relevant transfers
                    transfers = [
                        t for t in self.transfer_history
                        if t['from'] == source and t['to'] == target
                    ]
                    
                    if transfers:
                        patterns = np.array([t['pattern'] for t in transfers])
                        transfer_metrics[source][target] = {
                            'efficiency': float(1.0 - np.std(patterns) / 
                                             (np.mean(np.abs(patterns)) + 1e-7)),
                            'complexity': float(entropy(softmax(patterns.mean(axis=0)))),
                            'count': len(transfers)
                        }
                        
        return transfer_metrics 