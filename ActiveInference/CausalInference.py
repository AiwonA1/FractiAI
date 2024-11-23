"""
CausalInference.py

Implements causal inference for FractiAI, enabling causal discovery, intervention 
modeling, and counterfactual reasoning through fractal patterns.
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
class CausalConfig:
    """Configuration for causal inference system"""
    num_variables: int = 20
    temporal_horizon: int = 10
    intervention_strength: float = 0.7
    confidence_threshold: float = 0.8
    fractal_depth: int = 3
    learning_rate: float = 0.01

class CausalVariable:
    """Represents a variable in the causal system"""
    
    def __init__(self, var_id: str, dim: int):
        self.var_id = var_id
        self.dim = dim
        self.state = np.zeros(dim)
        self.history = []
        self.causes = {}  # {var_id: strength}
        self.effects = {}  # {var_id: strength}
        
    def update(self, inputs: Dict[str, np.ndarray], 
               intervention: Optional[np.ndarray] = None) -> np.ndarray:
        """Update variable state based on causal inputs"""
        # Compute causal influence
        causal_state = np.zeros_like(self.state)
        for var_id, strength in self.causes.items():
            if var_id in inputs:
                causal_state += strength * inputs[var_id]
                
        # Apply intervention if provided
        if intervention is not None:
            self.state = intervention
        else:
            # Update state with causal influences
            self.state = np.tanh(causal_state)
            
        # Store history
        self.history.append(self.state.copy())
        
        return self.state
        
    def add_cause(self, var_id: str, strength: float) -> None:
        """Add causal relationship"""
        self.causes[var_id] = strength
        
    def add_effect(self, var_id: str, strength: float) -> None:
        """Add effect relationship"""
        self.effects[var_id] = strength
        
    def get_causal_metrics(self) -> Dict[str, float]:
        """Get metrics about causal relationships"""
        return {
            'num_causes': len(self.causes),
            'num_effects': len(self.effects),
            'cause_strength': float(np.mean(list(self.causes.values()))),
            'effect_strength': float(np.mean(list(self.effects.values())))
        }

class CausalGraph:
    """Represents and analyzes causal relationships"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.strengths = {}  # {(cause, effect): strength}
        
    def add_relationship(self, cause: str, effect: str, 
                        strength: float) -> None:
        """Add causal relationship to graph"""
        self.graph.add_edge(cause, effect)
        self.strengths[(cause, effect)] = strength
        
    def remove_relationship(self, cause: str, effect: str) -> None:
        """Remove causal relationship"""
        if self.graph.has_edge(cause, effect):
            self.graph.remove_edge(cause, effect)
            del self.strengths[(cause, effect)]
            
    def get_causes(self, var_id: str) -> Dict[str, float]:
        """Get direct causes of variable"""
        return {
            pred: self.strengths[(pred, var_id)]
            for pred in self.graph.predecessors(var_id)
        }
        
    def get_effects(self, var_id: str) -> Dict[str, float]:
        """Get direct effects of variable"""
        return {
            succ: self.strengths[(var_id, succ)]
            for succ in self.graph.successors(var_id)
        }
        
    def get_metrics(self) -> Dict[str, float]:
        """Get graph metrics"""
        return {
            'num_edges': self.graph.number_of_edges(),
            'avg_degree': float(np.mean([d for _, d in self.graph.degree()])),
            'transitivity': nx.transitivity(self.graph),
            'avg_strength': float(np.mean(list(self.strengths.values())))
        }

class CausalInference:
    """Causal inference system with fractal patterns"""
    
    def __init__(self, config: CausalConfig):
        self.config = config
        self.variables: Dict[str, CausalVariable] = {}
        self.causal_graph = CausalGraph()
        self.interventions = []
        self.counterfactuals = []
        self.initialize_system()
        
    def initialize_system(self) -> None:
        """Initialize causal system"""
        # Create variables
        for i in range(self.config.num_variables):
            var_id = f"VAR_{i}"
            self.variables[var_id] = CausalVariable(var_id, dim=4)
            
        # Discover initial causal relationships
        self._discover_causal_relationships()
        
    def _discover_causal_relationships(self) -> None:
        """Discover causal relationships between variables"""
        for var1_id in self.variables:
            for var2_id in self.variables:
                if var1_id != var2_id:
                    # Compute causal strength
                    strength = self._compute_causal_strength(var1_id, var2_id)
                    
                    # Add relationship if strong enough
                    if strength > self.config.confidence_threshold:
                        self.causal_graph.add_relationship(var1_id, var2_id, strength)
                        self.variables[var2_id].add_cause(var1_id, strength)
                        self.variables[var1_id].add_effect(var2_id, strength)
                        
    def _compute_causal_strength(self, cause_id: str, 
                               effect_id: str) -> float:
        """Compute strength of causal relationship"""
        cause = self.variables[cause_id]
        effect = self.variables[effect_id]
        
        if not cause.history or not effect.history:
            return 0.0
            
        # Use time-lagged correlation as simple causal measure
        cause_states = np.array(cause.history[:-1])
        effect_states = np.array(effect.history[1:])
        
        if len(cause_states) > 1 and len(effect_states) > 1:
            correlation = np.corrcoef(
                cause_states.mean(axis=1),
                effect_states.mean(axis=1)
            )[0,1]
            return float(max(0, correlation))
            
        return 0.0
        
    def update(self, observations: Dict[str, np.ndarray],
              interventions: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """Update causal system"""
        # Update each variable
        updates = {}
        for var_id, variable in self.variables.items():
            # Get causal inputs
            inputs = {
                cause_id: self.variables[cause_id].state
                for cause_id in variable.causes
            }
            
            # Apply intervention if exists
            intervention = interventions.get(var_id) if interventions else None
            
            # Update variable
            new_state = variable.update(inputs, intervention)
            updates[var_id] = new_state
            
        # Update causal relationships periodically
        if len(self.variables[list(self.variables.keys())[0]].history) % 10 == 0:
            self._update_causal_relationships()
            
        return {
            'updates': updates,
            'metrics': self._compute_metrics()
        }
        
    def _update_causal_relationships(self) -> None:
        """Update causal relationships based on new data"""
        for var1_id in self.variables:
            for var2_id in self.variables:
                if var1_id != var2_id:
                    # Compute new strength
                    strength = self._compute_causal_strength(var1_id, var2_id)
                    
                    # Update relationship
                    if strength > self.config.confidence_threshold:
                        if not self.causal_graph.graph.has_edge(var1_id, var2_id):
                            self.causal_graph.add_relationship(var1_id, var2_id, strength)
                            self.variables[var2_id].add_cause(var1_id, strength)
                            self.variables[var1_id].add_effect(var2_id, strength)
                    else:
                        if self.causal_graph.graph.has_edge(var1_id, var2_id):
                            self.causal_graph.remove_relationship(var1_id, var2_id)
                            
    def intervene(self, interventions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Perform interventions on system"""
        # Store intervention
        self.interventions.append(interventions)
        
        # Update system with interventions
        result = self.update({}, interventions)
        
        # Compute intervention effects
        effects = self._compute_intervention_effects(interventions)
        
        return {
            'immediate_effects': result['updates'],
            'causal_effects': effects
        }
        
    def _compute_intervention_effects(self, 
                                    interventions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Compute effects of interventions"""
        effects = {}
        
        for var_id in interventions:
            # Get all downstream variables
            descendants = nx.descendants(self.causal_graph.graph, var_id)
            
            effects[var_id] = {}
            for desc_id in descendants:
                # Compute effect strength
                path_strength = self._compute_path_strength(var_id, desc_id)
                effects[var_id][desc_id] = path_strength
                
        return effects
        
    def _compute_path_strength(self, start: str, end: str) -> float:
        """Compute strength of causal path between variables"""
        paths = list(nx.all_simple_paths(self.causal_graph.graph, start, end))
        
        if not paths:
            return 0.0
            
        # Compute strength of each path
        path_strengths = []
        for path in paths:
            strength = 1.0
            for i in range(len(path)-1):
                strength *= self.causal_graph.strengths[(path[i], path[i+1])]
            path_strengths.append(strength)
            
        return float(np.max(path_strengths))
        
    def counterfactual(self, variable: str, 
                      value: np.ndarray) -> Dict[str, Any]:
        """Perform counterfactual reasoning"""
        # Store current state
        original_state = {
            var_id: var.state.copy()
            for var_id, var in self.variables.items()
        }
        
        # Intervene with counterfactual
        result = self.intervene({variable: value})
        
        # Restore original state
        for var_id, state in original_state.items():
            self.variables[var_id].state = state
            
        # Store counterfactual
        self.counterfactuals.append({
            'variable': variable,
            'value': value,
            'effects': result['causal_effects']
        })
        
        return result
        
    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute system metrics"""
        return {
            'graph_metrics': self.causal_graph.get_metrics(),
            'variable_metrics': {
                var_id: var.get_causal_metrics()
                for var_id, var in self.variables.items()
            },
            'intervention_count': len(self.interventions),
            'counterfactual_count': len(self.counterfactuals)
        } 