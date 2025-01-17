"""
ActiveInference.py

Implements sophisticated Active Inference for FractiAI agents, enabling goal-directed behavior through
policy selection based on expected free energy minimization. Uses advanced intelligence methods including
hierarchical inference, adaptive exploration, and meta-learning.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
import logging
from scipy.stats import entropy
from Methods.FreeEnergyPrinciple import FreeEnergyMinimizer, FEPConfig
from scipy.special import softmax

logger = logging.getLogger(__name__)

@dataclass 
class ActiveInferenceConfig:
    """Enhanced configuration for Active Inference"""
    n_policies: int = 10  # Number of policies to consider
    policy_horizon: int = 5  # Time horizon for policy evaluation
    exploration_factor: float = 1.0  # Controls exploration vs exploitation
    goal_precision: float = 2.0  # Precision of goal-directed behavior
    
    # Advanced parameters
    hierarchical_levels: int = 3  # Number of hierarchical inference levels
    memory_capacity: int = 1000  # Size of episodic memory
    adaptation_rate: float = 0.1  # Rate of policy adaptation
    meta_learning_rate: float = 0.01  # Rate of hyperparameter adaptation
    inference_iterations: int = 100  # Iterations for variational inference

class HierarchicalPolicy:
    """Represents a hierarchical temporal sequence of actions"""
    
    def __init__(self, action_dim: int, horizon: int, levels: int):
        # Initialize hierarchical action sequences
        self.actions = [np.random.randn(horizon, action_dim) for _ in range(levels)]
        self.values = np.zeros(levels)  # Expected free energy per level
        self.uncertainty = np.ones(levels)  # Uncertainty estimates
        self.success_history = []  # Track success of policy
        
    def sample_action(self, t: int, level: int = 0) -> np.ndarray:
        """Sample action at time t and hierarchical level"""
        action = self.actions[level][t]
        # Add contributions from higher levels
        for l in range(level + 1, len(self.actions)):
            action += self.actions[l][t] * (0.5 ** (l - level))
        return action

    def update(self, success: float, learning_rate: float) -> None:
        """Update policy based on success"""
        self.success_history.append(success)
        if len(self.success_history) > 100:
            self.success_history.pop(0)
        
        # Adapt actions based on success
        for level in range(len(self.actions)):
            self.actions[level] += learning_rate * success * np.random.randn(*self.actions[level].shape)

class ActiveInferenceAgent:
    """Implements advanced active inference for goal-directed behavior"""
    
    def __init__(self, config: ActiveInferenceConfig, fep_config: FEPConfig):
        self.config = config
        self.fep = FreeEnergyMinimizer(fep_config)
        self.policies = []
        self.action_dim = fep_config.state_dim
        
        # Enhanced components
        self.episodic_memory = []
        self.meta_parameters = {'exploration': config.exploration_factor,
                              'precision': config.goal_precision}
        self.context_model = self._initialize_context_model()
        
        self.initialize_policies()
        
    def _initialize_context_model(self) -> Dict[str, Any]:
        """Initialize context learning model"""
        return {
            'contexts': [],  # Known contexts
            'context_policies': {},  # Context-specific policies
            'context_values': {},  # Value estimates per context
            'transition_model': np.zeros((0, 0))  # Context transitions
        }
        
    def initialize_policies(self) -> None:
        """Initialize hierarchical set of policies"""
        self.policies = [
            HierarchicalPolicy(self.action_dim, self.config.policy_horizon, 
                             self.config.hierarchical_levels)
            for _ in range(self.config.n_policies)
        ]
        
    def infer_state(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Hierarchical state inference"""
        state_estimates = []
        uncertainties = []
        
        # Infer state at each level
        for level in range(self.config.hierarchical_levels):
            state, uncertainty = self.fep.infer_state(
                observation, 
                n_iterations=self.config.inference_iterations
            )
            state_estimates.append(state)
            uncertainties.append(uncertainty)
            
            # Generate prediction for next level
            observation = self.fep.model.generate_observation(state)
            
        return np.mean(state_estimates, axis=0), np.mean(uncertainties, axis=0)
        
    def select_action(self, observation: np.ndarray, goal_state: np.ndarray) -> np.ndarray:
        """Select action using sophisticated active inference"""
        # Infer current state and context
        current_state, uncertainty = self.infer_state(observation)
        context = self._infer_context(current_state, observation)
        
        # Update context model
        self._update_context_model(context, current_state)
        
        # Evaluate policies considering context
        self._evaluate_policies(current_state, goal_state, context)
        
        # Adaptive exploration
        self._adapt_exploration(uncertainty)
        
        # Sample policy using sophisticated selection
        selected_policy = self._sample_policy(context)
        
        # Store in episodic memory
        self._update_memory(observation, current_state, context)
        
        return selected_policy.sample_action(0)
    
    def _infer_context(self, state: np.ndarray, observation: np.ndarray) -> int:
        """Infer current context using Bayesian inference"""
        if not self.context_model['contexts']:
            self.context_model['contexts'].append(state.copy())
            return 0
            
        # Compute likelihood for each context
        likelihoods = []
        for context in self.context_model['contexts']:
            diff = state - context
            likelihood = np.exp(-0.5 * np.sum(diff ** 2))
            likelihoods.append(likelihood)
            
        # Normalize and sample
        likelihoods = np.array(likelihoods)
        likelihoods /= np.sum(likelihoods)
        
        # Create new context if all likelihoods are low
        if np.max(likelihoods) < 0.2:
            self.context_model['contexts'].append(state.copy())
            return len(self.context_model['contexts']) - 1
            
        return np.argmax(likelihoods)
    
    def _evaluate_policies(self, current_state: np.ndarray,
                         goal_state: np.ndarray,
                         context: int) -> None:
        """Evaluate expected free energy hierarchically"""
        for policy in self.policies:
            total_value = 0
            for level in range(self.config.hierarchical_levels):
                value = self._compute_expected_free_energy(
                    current_state, goal_state, policy, level, context)
                policy.values[level] = value
                total_value += value * (0.5 ** level)  # Weight higher levels less
            policy.value = total_value
    
    def _compute_expected_free_energy(self, current_state: np.ndarray,
                                    goal_state: np.ndarray,
                                    policy: HierarchicalPolicy,
                                    level: int,
                                    context: int) -> float:
        """Compute sophisticated expected free energy"""
        expected_free_energy = 0.0
        state = current_state.copy()
        uncertainty = np.ones_like(state)
        
        for t in range(self.config.policy_horizon):
            # Get hierarchical action
            action = policy.sample_action(t, level)
            
            # Predict next state with uncertainty propagation
            next_state, next_uncertainty = self._predict_next_state(
                state, action, uncertainty, context)
            
            # Generate expected observation
            expected_obs = self.fep.model.generate_observation(next_state)
            
            # Compute sophisticated free energy components
            ambiguity = self._compute_ambiguity(next_state, next_uncertainty)
            risk = self._compute_risk(next_state, goal_state, next_uncertainty)
            novelty = self._compute_novelty(next_state, context)
            
            # Accumulate expected free energy
            expected_free_energy += (ambiguity + 
                                   self.config.goal_precision * risk -
                                   self.meta_parameters['exploration'] * novelty)
            
            state = next_state
            uncertainty = next_uncertainty
            
        return float(expected_free_energy)
    
    def _predict_next_state(self, state: np.ndarray, action: np.ndarray,
                          uncertainty: np.ndarray, context: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sophisticated state prediction with uncertainty"""
        # Get context-specific dynamics
        if context in self.context_model['context_policies']:
            dynamics = self.context_model['context_policies'][context]
        else:
            dynamics = self.fep.model.predict_state
            
        # Predict next state
        next_state = dynamics(state) + action
        
        # Propagate uncertainty
        process_noise = self.fep.model.process_noise
        next_uncertainty = uncertainty + process_noise
        
        return next_state, next_uncertainty
    
    def _compute_ambiguity(self, state: np.ndarray, uncertainty: np.ndarray) -> float:
        """Compute sophisticated ambiguity measure"""
        obs_cov = np.linalg.inv(self.fep.model.Pi_w)
        total_cov = obs_cov + np.diag(uncertainty)
        entropy_val = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * total_cov))
        return float(entropy_val)
    
    def _compute_risk(self, predicted_state: np.ndarray,
                     goal_state: np.ndarray,
                     uncertainty: np.ndarray) -> float:
        """Compute sophisticated risk measure"""
        diff = predicted_state - goal_state
        precision = np.linalg.inv(np.diag(uncertainty))
        return float(0.5 * np.dot(diff.T, np.dot(precision, diff)))
    
    def _compute_novelty(self, state: np.ndarray, context: int) -> float:
        """Compute novelty of state for exploration"""
        if not self.episodic_memory:
            return 1.0
            
        # Compute distance to memory states
        distances = []
        for mem_state in self.episodic_memory:
            diff = state - mem_state
            distance = np.sqrt(np.sum(diff ** 2))
            distances.append(distance)
            
        # Novelty is inverse of minimum distance
        return 1.0 / (1.0 + min(distances))
    
    def _adapt_exploration(self, uncertainty: np.ndarray) -> None:
        """Adapt exploration based on uncertainty"""
        avg_uncertainty = np.mean(uncertainty)
        self.meta_parameters['exploration'] *= (1.0 - self.config.meta_learning_rate)
        self.meta_parameters['exploration'] += self.config.meta_learning_rate * avg_uncertainty
        
    def _update_memory(self, observation: np.ndarray, state: np.ndarray, context: int) -> None:
        """Update episodic memory"""
        self.episodic_memory.append(state.copy())
        if len(self.episodic_memory) > self.config.memory_capacity:
            self.episodic_memory.pop(0)
            
    def _update_context_model(self, context: int, state: np.ndarray) -> None:
        """Update context transition model"""
        n_contexts = len(self.context_model['contexts'])
        if n_contexts > 1:
            if self.context_model['transition_model'].shape[0] < n_contexts:
                # Expand transition matrix
                new_transitions = np.zeros((n_contexts, n_contexts))
                old_size = self.context_model['transition_model'].shape[0]
                new_transitions[:old_size, :old_size] = self.context_model['transition_model']
                self.context_model['transition_model'] = new_transitions
                
            # Update transition probabilities
            if hasattr(self, '_last_context'):
                self.context_model['transition_model'][self._last_context, context] += 1
                
        self._last_context = context
        
    def _sample_policy(self, context: int) -> HierarchicalPolicy:
        """Sophisticated policy sampling"""
        # Get context-specific values if available
        if context in self.context_model['context_values']:
            base_values = self.context_model['context_values'][context]
        else:
            base_values = np.array([p.value for p in self.policies])
            
        # Compute policy probabilities
        values = np.array([p.value for p in self.policies])
        combined_values = 0.5 * base_values + 0.5 * values
        probabilities = np.exp(-self.meta_parameters['exploration'] * combined_values)
        probabilities /= np.sum(probabilities)
        
        # Sample policy
        idx = np.random.choice(len(self.policies), p=probabilities)
        return self.policies[idx]
    
    def update_model(self, observations: List[np.ndarray],
                    actions: List[np.ndarray],
                    states: List[np.ndarray]) -> None:
        """Update all internal models based on experience"""
        # Update base FEP model
        self.fep.update_model(observations, states)
        
        # Update successful policies
        if len(actions) > 1:
            successful_policy = HierarchicalPolicy(
                self.action_dim, len(actions), self.config.hierarchical_levels)
            
            # Set actions at each level
            for level in range(self.config.hierarchical_levels):
                successful_policy.actions[level] = np.array(actions)
                
            # Compute success metric
            final_state = states[-1]
            initial_state = states[0]
            success = -np.sum((final_state - initial_state) ** 2)
            
            # Update policy
            successful_policy.update(success, self.config.adaptation_rate)
            
            # Replace worst performing policy
            worst_idx = np.argmax([p.value for p in self.policies])
            self.policies[worst_idx] = successful_policy

"""
RecursiveMetaLearner: Implements hierarchical meta-learning through recursive
pattern discovery and optimization.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy.stats import entropy
from scipy.special import softmax

logger = logging.getLogger(__name__)

@dataclass
class MetaLearnerConfig:
    """Configuration for recursive meta-learning"""
    max_depth: int = 5
    pattern_dim: int = 64
    learning_rate: float = 0.01
    memory_size: int = 1000
    batch_size: int = 32

class RecursiveMetaLearner:
    """Implements recursive meta-learning"""
    
    def __init__(self, config: MetaLearnerConfig):
        self.config = config
        self.pattern_hierarchy = {}
        self.meta_patterns = {}
        self.learning_history = []
        
    def learn_pattern(self, pattern: np.ndarray, depth: int = 0) -> Dict[str, Any]:
        """Learn pattern at specified depth"""
        if depth >= self.config.max_depth:
            return {'pattern': pattern, 'depth': depth}
            
        # Store pattern at current depth
        pattern_id = f"pattern_{depth}_{len(self.pattern_hierarchy)}"
        self.pattern_hierarchy[pattern_id] = {
            'pattern': pattern,
            'depth': depth,
            'children': [],
            'meta_patterns': []
        }
        
        # Extract sub-patterns
        sub_patterns = self._extract_sub_patterns(pattern)
        
        # Recursively learn sub-patterns
        child_results = []
        for sub_pattern in sub_patterns:
            child_result = self.learn_pattern(sub_pattern, depth + 1)
            if child_result:
                self.pattern_hierarchy[pattern_id]['children'].append(
                    child_result['pattern_id']
                )
                child_results.append(child_result)
                
        # Generate meta-pattern
        if child_results:
            meta_pattern = self._generate_meta_pattern(child_results)
            meta_pattern_id = f"meta_{pattern_id}"
            self.meta_patterns[meta_pattern_id] = {
                'pattern': meta_pattern,
                'source_patterns': [r['pattern_id'] for r in child_results],
                'depth': depth
            }
            self.pattern_hierarchy[pattern_id]['meta_patterns'].append(
                meta_pattern_id
            )
            
        # Store learning experience
        experience = {
            'pattern_id': pattern_id,
            'depth': depth,
            'child_results': child_results,
            'meta_pattern_id': meta_pattern_id if child_results else None
        }
        self.learning_history.append(experience)
        
        return experience
        
    def _extract_sub_patterns(self, pattern: np.ndarray) -> List[np.ndarray]:
        """Extract sub-patterns from pattern"""
        sub_patterns = []
        
        # Split pattern into segments
        segment_size = len(pattern) // 2
        for i in range(0, len(pattern) - segment_size + 1, segment_size // 2):
            sub_pattern = pattern[i:i + segment_size]
            if len(sub_pattern) == segment_size:
                sub_patterns.append(sub_pattern)
                
        return sub_patterns
        
    def _generate_meta_pattern(self, child_results: List[Dict[str, Any]]) -> np.ndarray:
        """Generate meta-pattern from child patterns"""
        patterns = [r['pattern'] for r in child_results]
        
        # Resize patterns if needed
        target_size = self.config.pattern_dim
        resized_patterns = []
        for pattern in patterns:
            if len(pattern) != target_size:
                pattern = np.interp(
                    np.linspace(0, 1, target_size),
                    np.linspace(0, 1, len(pattern)),
                    pattern
                )
            resized_patterns.append(pattern)
            
        # Compute weighted average
        weights = softmax([1.0 / (r['depth'] + 1) for r in child_results])
        meta_pattern = np.average(resized_patterns, axis=0, weights=weights)
        
        return meta_pattern
        
    def find_similar_patterns(self, pattern: np.ndarray,
                           depth: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find similar patterns at specified depth"""
        similar_patterns = []
        
        # Filter patterns by depth if specified
        patterns_to_check = self.pattern_hierarchy
        if depth is not None:
            patterns_to_check = {
                k: v for k, v in self.pattern_hierarchy.items()
                if v['depth'] == depth
            }
            
        # Check similarity with stored patterns
        for pattern_id, pattern_data in patterns_to_check.items():
            similarity = self._compute_similarity(pattern, pattern_data['pattern'])
            
            if similarity > 0.7:  # Similarity threshold
                similar_patterns.append({
                    'pattern_id': pattern_id,
                    'similarity': similarity,
                    'depth': pattern_data['depth'],
                    'pattern': pattern_data['pattern']
                })
                
        return similar_patterns
        
    def _compute_similarity(self, pattern1: np.ndarray,
                         pattern2: np.ndarray) -> float:
        """Compute similarity between patterns"""
        # Resize patterns if needed
        if pattern1.shape != pattern2.shape:
            pattern1 = np.interp(
                np.linspace(0, 1, self.config.pattern_dim),
                np.linspace(0, 1, len(pattern1)),
                pattern1
            )
            pattern2 = np.interp(
                np.linspace(0, 1, self.config.pattern_dim),
                np.linspace(0, 1, len(pattern2)),
                pattern2
            )
            
        # Compute correlation
        return float(np.abs(np.corrcoef(pattern1, pattern2)[0,1]))
        
    def optimize_pattern(self, pattern: np.ndarray,
                       target_pattern: np.ndarray) -> np.ndarray:
        """Optimize pattern towards target"""
        # Resize patterns if needed
        if pattern.shape != target_pattern.shape:
            pattern = np.interp(
                np.linspace(0, 1, len(target_pattern)),
                np.linspace(0, 1, len(pattern)),
                pattern
            )
            
        # Compute gradient
        error = target_pattern - pattern
        gradient = error * self.config.learning_rate
        
        # Update pattern
        optimized_pattern = pattern + gradient
        
        return optimized_pattern