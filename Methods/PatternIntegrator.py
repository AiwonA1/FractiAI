"""
PatternIntegrator.py

Implements advanced pattern integration capabilities for cross-domain pattern recognition,
transformation, and harmonization in the FractiAI system.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from scipy.stats import entropy
from Methods.FractalTransformer import FractalTransformer, FractalTransformerConfig
from Methods.MetaCognition import MetaCognition, MetaCogConfig

@dataclass
class PatternIntegratorConfig:
    """Configuration for pattern integration"""
    pattern_dim: int = 256
    num_domains: int = 4
    integration_layers: int = 3
    coherence_threshold: float = 0.7
    adaptation_rate: float = 0.01
    resonance_factor: float = 0.1
    max_patterns: int = 1000
    
class DomainAdapter:
    """Adapts patterns between different domains"""
    
    def __init__(self, config: PatternIntegratorConfig):
        self.config = config
        self.domain_mappings: Dict[Tuple[str, str], nn.Module] = {}
        self.domain_stats = defaultdict(list)
        
    def register_domain_pair(self, domain1: str, domain2: str) -> None:
        """Register a new domain pair for adaptation"""
        if (domain1, domain2) not in self.domain_mappings:
            # Create bidirectional mappings
            self.domain_mappings[(domain1, domain2)] = nn.Sequential(
                nn.Linear(self.config.pattern_dim, self.config.pattern_dim * 2),
                nn.GELU(),
                nn.Linear(self.config.pattern_dim * 2, self.config.pattern_dim)
            )
            self.domain_mappings[(domain2, domain1)] = nn.Sequential(
                nn.Linear(self.config.pattern_dim, self.config.pattern_dim * 2),
                nn.GELU(),
                nn.Linear(self.config.pattern_dim * 2, self.config.pattern_dim)
            )
            
    def adapt_pattern(self, pattern: torch.Tensor,
                     source_domain: str,
                     target_domain: str) -> torch.Tensor:
        """Adapt pattern from source to target domain"""
        if (source_domain, target_domain) not in self.domain_mappings:
            self.register_domain_pair(source_domain, target_domain)
            
        # Apply domain mapping
        mapping = self.domain_mappings[(source_domain, target_domain)]
        adapted = mapping(pattern)
        
        # Track adaptation stats
        similarity = F.cosine_similarity(pattern, adapted, dim=-1).mean()
        self.domain_stats[(source_domain, target_domain)].append(float(similarity))
        
        return adapted
        
class PatternHarmonizer:
    """Harmonizes patterns across domains"""
    
    def __init__(self, config: PatternIntegratorConfig):
        self.config = config
        self.pattern_graph = nx.Graph()
        self.domain_patterns: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        
    def add_pattern(self, pattern_id: str,
                   pattern: torch.Tensor,
                   domain: str) -> None:
        """Add pattern to domain"""
        self.domain_patterns[domain][pattern_id] = pattern
        
        # Update pattern graph
        self.pattern_graph.add_node(
            (domain, pattern_id),
            pattern=pattern.detach().numpy()
        )
        
        # Calculate cross-domain similarities
        for other_domain, patterns in self.domain_patterns.items():
            if other_domain != domain:
                for other_id, other_pattern in patterns.items():
                    similarity = float(F.cosine_similarity(
                        pattern.unsqueeze(0),
                        other_pattern.unsqueeze(0)
                    ))
                    if similarity > self.config.coherence_threshold:
                        self.pattern_graph.add_edge(
                            (domain, pattern_id),
                            (other_domain, other_id),
                            weight=similarity
                        )
                        
    def harmonize_patterns(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Harmonize patterns across all domains"""
        harmonized = defaultdict(dict)
        
        # Find pattern communities
        communities = list(nx.community.greedy_modularity_communities(
            self.pattern_graph, weight='weight'))
            
        # Harmonize within each community
        for community in communities:
            patterns = []
            for domain, pattern_id in community:
                pattern = self.domain_patterns[domain][pattern_id]
                patterns.append(pattern)
                
            if patterns:
                # Calculate mean pattern
                mean_pattern = torch.mean(torch.stack(patterns), dim=0)
                
                # Update patterns with resonance
                for domain, pattern_id in community:
                    original = self.domain_patterns[domain][pattern_id]
                    resonance = torch.sigmoid(
                        F.cosine_similarity(original, mean_pattern, dim=0)
                    ) * self.config.resonance_factor
                    harmonized[domain][pattern_id] = (
                        original * (1 - resonance) + mean_pattern * resonance
                    )
                    
        return dict(harmonized)
        
class PatternIntegrator:
    """Main pattern integration system"""
    
    def __init__(self, config: PatternIntegratorConfig):
        self.config = config
        
        # Initialize components
        self.adapter = DomainAdapter(config)
        self.harmonizer = PatternHarmonizer(config)
        
        # Initialize transformers for each domain
        self.domain_transformers: Dict[str, FractalTransformer] = {}
        
        # Initialize metacognition
        metacog_config = MetaCogConfig(
            state_dim=config.pattern_dim,
            pattern_threshold=config.coherence_threshold,
            adaptation_rate=config.adaptation_rate
        )
        self.metacognition = MetaCognition(metacog_config)
        
        # Pattern tracking
        self.pattern_history: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.integration_stats = defaultdict(list)
        
    def register_domain(self, domain: str) -> None:
        """Register new domain for integration"""
        if domain not in self.domain_transformers:
            # Create domain transformer
            transformer_config = FractalTransformerConfig(
                hidden_dim=self.config.pattern_dim,
                coherence_threshold=self.config.coherence_threshold
            )
            self.domain_transformers[domain] = FractalTransformer(transformer_config)
            
    def process_pattern(self, pattern: torch.Tensor,
                       domain: str,
                       pattern_id: Optional[str] = None) -> Dict[str, Any]:
        """Process pattern from domain"""
        if domain not in self.domain_transformers:
            self.register_domain(domain)
            
        # Transform pattern
        transformer = self.domain_transformers[domain]
        transformed = transformer.transform_pattern(pattern)
        
        # Generate pattern ID if not provided
        if pattern_id is None:
            pattern_id = f"pattern_{len(self.pattern_history[domain])}"
            
        # Add to harmonizer
        self.harmonizer.add_pattern(pattern_id, transformed, domain)
        
        # Track pattern
        self.pattern_history[domain].append(transformed)
        
        # Update metacognition
        metacog_result = self.metacognition.process_state(
            transformed.detach().numpy(),
            {'domain': domain, 'pattern_id': pattern_id}
        )
        
        # Harmonize patterns
        harmonized = self.harmonizer.harmonize_patterns()
        
        # Track integration stats
        if domain in harmonized and pattern_id in harmonized[domain]:
            similarity = float(F.cosine_similarity(
                transformed,
                harmonized[domain][pattern_id],
                dim=0
            ))
            self.integration_stats[domain].append(similarity)
            
        return {
            'pattern_id': pattern_id,
            'transformed': transformed,
            'harmonized': harmonized.get(domain, {}).get(pattern_id),
            'metacognition': metacog_result,
            'cross_domain_coherence': np.mean([
                np.mean(stats) for stats in self.integration_stats.values()
            ])
        }
        
    def adapt_pattern(self, pattern: torch.Tensor,
                     source_domain: str,
                     target_domain: str) -> torch.Tensor:
        """Adapt pattern between domains"""
        return self.adapter.adapt_pattern(pattern, source_domain, target_domain)
        
    def get_domain_patterns(self, domain: str) -> Dict[str, torch.Tensor]:
        """Get all patterns for domain"""
        return self.harmonizer.domain_patterns.get(domain, {})
        
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get integration performance metrics"""
        return {
            'num_patterns': {
                domain: len(patterns)
                for domain, patterns in self.harmonizer.domain_patterns.items()
            },
            'mean_coherence': {
                domain: float(np.mean(stats))
                for domain, stats in self.integration_stats.items()
            },
            'cross_domain_connections': len(self.harmonizer.pattern_graph.edges()),
            'metacognition': self.metacognition.get_metrics()
        } 