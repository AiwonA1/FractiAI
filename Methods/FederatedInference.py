"""
FederatedInference.py

Implements advanced federated inference capabilities for FractiAI, enabling privacy-preserving
distributed learning and pattern aggregation across multiple nodes while maintaining
fractal coherence and quantum resonance properties. Integrates sophisticated active inference
for distributed goal-directed learning and adaptation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import logging
from scipy.stats import entropy
from cryptography.fernet import Fernet
from Methods.MetaCognition import MetaCogConfig, CognitiveState
from Methods.FractalTransformer import FractalTransformer, FractalTransformerConfig
from Methods.ActiveInference import ActiveInferenceAgent, ActiveInferenceConfig
from Methods.FreeEnergyPrinciple import FreeEnergyMinimizer, FEPConfig

logger = logging.getLogger(__name__)

@dataclass
class FederatedConfig:
    """Configuration for federated inference system"""
    num_nodes: int = 10
    aggregation_rounds: int = 5
    min_node_samples: int = 100
    privacy_budget: float = 1.0
    noise_scale: float = 0.1
    pattern_compression: float = 0.8
    quantum_sync_rate: float = 0.1
    coherence_threshold: float = 0.7
    max_divergence: float = 0.3
    active_inference_horizon: int = 5
    goal_precision: float = 2.0
    exploration_factor: float = 1.0
    hierarchical_levels: int = 3
    meta_learning_rate: float = 0.01
    
class SecureAggregation:
    """Implements secure aggregation protocol for federated learning"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.encryption_keys = {}
        self.noise_masks = {}
        self.initialize_security()
        
    def initialize_security(self) -> None:
        """Initialize security components"""
        # Generate encryption keys for each node
        for i in range(self.config.num_nodes):
            self.encryption_keys[f"node_{i}"] = Fernet.generate_key()
            
        # Initialize pairwise noise masks
        for i in range(self.config.num_nodes):
            for j in range(i + 1, self.config.num_nodes):
                mask = np.random.normal(0, self.config.noise_scale, 1000)
                self.noise_masks[f"node_{i}_{j}"] = mask
                self.noise_masks[f"node_{j}_{i}"] = -mask
                
    def encrypt_pattern(self, pattern: np.ndarray, node_id: str) -> bytes:
        """Encrypt pattern updates"""
        f = Fernet(self.encryption_keys[node_id])
        return f.encrypt(pattern.tobytes())
        
    def decrypt_pattern(self, encrypted_pattern: bytes, node_id: str) -> np.ndarray:
        """Decrypt pattern updates"""
        f = Fernet(self.encryption_keys[node_id])
        decrypted = f.decrypt(encrypted_pattern)
        return np.frombuffer(decrypted).reshape(-1)
        
    def add_noise_mask(self, pattern: np.ndarray, 
                      node_id: str, round_id: int) -> np.ndarray:
        """Add noise mask to pattern"""
        total_noise = np.zeros_like(pattern)
        for other_id in self.noise_masks:
            if node_id in other_id:
                mask = self.noise_masks[other_id]
                if len(mask) > len(pattern):
                    mask = mask[:len(pattern)]
                else:
                    mask = np.pad(mask, (0, len(pattern) - len(mask)))
                total_noise += mask
                
        return pattern + total_noise * self.config.noise_scale

class QuantumFederatedState:
    """Manages quantum state synchronization across federated nodes"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.node_states = {}
        self.entanglement_maps = {}
        self.coherence_history = defaultdict(list)
        
    def update_node_state(self, node_id: str, 
                         cognitive_state: CognitiveState) -> None:
        """Update quantum state for node"""
        # Generate quantum state
        quantum_state = self._generate_quantum_state(cognitive_state)
        self.node_states[node_id] = quantum_state
        
        # Update entanglement maps
        self._update_entanglement(node_id, quantum_state)
        
    def _generate_quantum_state(self, cognitive_state: CognitiveState) -> np.ndarray:
        """Generate quantum state from cognitive state"""
        # Create superposition
        amplitude = np.fft.fft(cognitive_state.state_pattern)
        phase = np.angle(amplitude)
        quantum_state = amplitude * np.exp(1j * phase)
        return quantum_state / np.linalg.norm(quantum_state)
        
    def _update_entanglement(self, node_id: str, 
                           quantum_state: np.ndarray) -> None:
        """Update entanglement maps between nodes"""
        for other_id, other_state in self.node_states.items():
            if other_id != node_id:
                correlation = np.abs(np.dot(quantum_state.conj(), other_state))
                self.entanglement_maps[f"{node_id}_{other_id}"] = correlation
                
    def synchronize_states(self) -> Dict[str, float]:
        """Synchronize quantum states across nodes"""
        coherence_scores = {}
        
        # Compute coherence for each node
        for node_id in self.node_states:
            entanglements = [
                score for pair, score in self.entanglement_maps.items()
                if node_id in pair
            ]
            coherence = np.mean(entanglements) if entanglements else 0.0
            coherence_scores[node_id] = coherence
            self.coherence_history[node_id].append(coherence)
            
        return coherence_scores

class FederatedActiveInference:
    """Implements federated active inference across nodes"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        
        # Initialize active inference components
        active_config = ActiveInferenceConfig(
            n_policies=10,
            policy_horizon=config.active_inference_horizon,
            exploration_factor=config.exploration_factor,
            goal_precision=config.goal_precision,
            hierarchical_levels=config.hierarchical_levels
        )
        fep_config = FEPConfig(
            state_dim=1000,  # Match pattern dimension
            obs_dim=1000,
            learning_rate=config.meta_learning_rate
        )
        
        # Create active inference agents for each node
        self.node_agents: Dict[str, ActiveInferenceAgent] = {
            f"node_{i}": ActiveInferenceAgent(active_config, fep_config)
            for i in range(config.num_nodes)
        }
        
        # Shared components
        self.global_goal_state = np.zeros(1000)  # Initialize neutral goal
        self.policy_library = []
        self.context_models = {}
        self.inference_history = []
        
    def process_node_state(self, node_id: str, 
                          state: np.ndarray,
                          goal_state: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Process node state using active inference"""
        agent = self.node_agents[node_id]
        
        # Use provided or global goal state
        target_goal = goal_state if goal_state is not None else self.global_goal_state
        
        # Select action using active inference
        action = agent.select_action(state, target_goal)
        
        # Update context model
        context = agent._infer_context(state, state)  # Use state as observation
        self._update_shared_context_model(node_id, context, state)
        
        return {
            'action': action,
            'context': context,
            'value': agent.policies[0].value  # Use best policy's value
        }
        
    def _update_shared_context_model(self, node_id: str,
                                   context: int,
                                   state: np.ndarray) -> None:
        """Update shared context model across nodes"""
        if node_id not in self.context_models:
            self.context_models[node_id] = {
                'contexts': [],
                'transitions': np.zeros((0, 0)),
                'states': []
            }
            
        model = self.context_models[node_id]
        
        # Update context states
        if context >= len(model['contexts']):
            model['contexts'].append(state.copy())
            
            # Expand transition matrix
            n_contexts = len(model['contexts'])
            new_transitions = np.zeros((n_contexts, n_contexts))
            if model['transitions'].size > 0:
                old_size = model['transitions'].shape[0]
                new_transitions[:old_size, :old_size] = model['transitions']
            model['transitions'] = new_transitions
            
        # Update transitions
        if hasattr(self, f'_last_context_{node_id}'):
            last_context = getattr(self, f'_last_context_{node_id}')
            model['transitions'][last_context, context] += 1
            
        setattr(self, f'_last_context_{node_id}', context)
        
    def synchronize_policies(self) -> None:
        """Synchronize successful policies across nodes"""
        # Collect successful policies from all nodes
        all_policies = []
        for node_id, agent in self.node_agents.items():
            successful = [p for p in agent.policies if np.mean(p.success_history) > 0.7]
            all_policies.extend(successful)
            
        if not all_policies:
            return
            
        # Select top policies based on success
        top_policies = sorted(
            all_policies,
            key=lambda p: np.mean(p.success_history),
            reverse=True
        )[:5]  # Keep top 5
        
        # Share with all nodes
        for agent in self.node_agents.values():
            # Replace worst policies with top ones
            worst_indices = np.argsort([p.value for p in agent.policies])[:len(top_policies)]
            for idx, policy in zip(worst_indices, top_policies):
                agent.policies[idx] = policy.copy()  # Assuming policy has copy method

class FederatedNode:
    """Individual node in federated learning system"""
    
    def __init__(self, node_id: str, config: FederatedConfig):
        self.node_id = node_id
        self.config = config
        self.cognitive_state = CognitiveState(1000)  # Default dimension
        self.local_patterns = {}
        self.update_history = []
        self.transformer = FractalTransformer(FractalTransformerConfig())
        
        # Add active inference components
        active_config = ActiveInferenceConfig(
            n_policies=5,
            policy_horizon=config.active_inference_horizon,
            exploration_factor=config.exploration_factor,
            goal_precision=config.goal_precision
        )
        fep_config = FEPConfig(state_dim=1000, obs_dim=1000)
        self.active_inference = ActiveInferenceAgent(active_config, fep_config)
        
    def process_local_batch(self, data: torch.Tensor,
                          goal_state: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Process local data batch with active inference"""
        # Transform patterns
        transformed = self.transformer.transform_pattern(data)
        transformed_np = transformed.detach().numpy()
        
        # Apply active inference
        if goal_state is not None:
            action = self.active_inference.select_action(transformed_np, goal_state)
            transformed_np = transformed_np + action
            
        # Update local patterns
        pattern_id = f"pattern_{len(self.local_patterns)}"
        self.local_patterns[pattern_id] = {
            'pattern': transformed_np,
            'timestamp': np.datetime64('now'),
            'goal_state': goal_state
        }
        
        # Update cognitive state
        self._update_cognitive_state(transformed_np)
        
        return {
            'pattern_id': pattern_id,
            'transformed': torch.from_numpy(transformed_np),
            'cognitive_state': self.cognitive_state,
            'action': action if goal_state is not None else None
        }
        
    def _update_cognitive_state(self, pattern: np.ndarray) -> None:
        """Update node's cognitive state with active inference"""
        if len(pattern.shape) > 1:
            pattern = pattern.mean(axis=0)
            
        # Infer state using active inference
        state, uncertainty = self.active_inference.infer_state(pattern)
        
        self.cognitive_state.update(state, {
            'meta_state': state,
            'uncertainty': uncertainty,
            'coherence': float(np.mean(np.abs(state)))
        })

class FederatedInference:
    """Main federated inference system"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.nodes: Dict[str, FederatedNode] = {}
        self.secure_aggregation = SecureAggregation(config)
        self.quantum_state = QuantumFederatedState(config)
        self.active_inference = FederatedActiveInference(config)
        self.global_patterns = {}
        self.round_history = []
        
        self.initialize_nodes()
        
    def initialize_nodes(self) -> None:
        """Initialize federated nodes"""
        for i in range(self.config.num_nodes):
            node_id = f"node_{i}"
            self.nodes[node_id] = FederatedNode(node_id, self.config)
            
    def federated_round(self, goal_state: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Execute one round of federated learning with active inference"""
        # Update global goal state if provided
        if goal_state is not None:
            self.active_inference.global_goal_state = goal_state
            
        # Collect updates from nodes
        node_updates = {}
        node_actions = {}
        for node_id, node in self.nodes.items():
            # Get node update
            pattern, metadata = node.prepare_update()
            if len(pattern) > 0:
                # Apply active inference
                inference_result = self.active_inference.process_node_state(
                    node_id, pattern, goal_state)
                
                # Add security measures
                noised_pattern = self.secure_aggregation.add_noise_mask(
                    pattern + inference_result['action'],  # Add selected action
                    node_id, len(self.round_history))
                encrypted_pattern = self.secure_aggregation.encrypt_pattern(
                    noised_pattern, node_id)
                    
                node_updates[node_id] = {
                    'pattern': encrypted_pattern,
                    'metadata': metadata
                }
                node_actions[node_id] = inference_result
                
        # Aggregate updates
        aggregated_pattern = self._aggregate_updates(node_updates)
        
        # Synchronize active inference policies
        self.active_inference.synchronize_policies()
        
        # Update quantum states
        self._update_quantum_states()
        
        # Store round results
        round_results = {
            'round_id': len(self.round_history),
            'num_updates': len(node_updates),
            'aggregated_pattern': aggregated_pattern,
            'quantum_coherence': self.quantum_state.synchronize_states(),
            'node_actions': node_actions
        }
        self.round_history.append(round_results)
        
        return round_results
        
    def _aggregate_updates(self, 
                         node_updates: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """Aggregate encrypted updates from nodes"""
        if not node_updates:
            return np.array([])
            
        # Decrypt and aggregate patterns
        patterns = []
        for node_id, update in node_updates.items():
            decrypted = self.secure_aggregation.decrypt_pattern(
                update['pattern'], node_id)
            patterns.append(decrypted)
            
        # Weighted average based on number of local patterns
        weights = [
            update['metadata']['num_patterns'] 
            for update in node_updates.values()
        ]
        weights = np.array(weights) / np.sum(weights)
        
        return np.average(patterns, axis=0, weights=weights)
        
    def _update_quantum_states(self) -> None:
        """Update quantum states for all nodes"""
        for node_id, node in self.nodes.items():
            self.quantum_state.update_node_state(
                node_id, node.cognitive_state)
            
    def get_federation_metrics(self) -> Dict[str, Any]:
        """Get enhanced metrics about federation performance"""
        if not self.round_history:
            return {}
            
        # Get base metrics
        metrics = super().get_federation_metrics()
        
        # Add active inference metrics
        action_metrics = defaultdict(list)
        for round_data in self.round_history:
            if 'node_actions' in round_data:
                for node_id, action_data in round_data['node_actions'].items():
                    action_metrics[node_id].append(action_data['value'])
                    
        # Compute action value trends
        action_trends = {}
        for node_id, values in action_metrics.items():
            if len(values) > 1:
                trend = np.polyfit(np.arange(len(values)), values, 1)[0]
                action_trends[node_id] = {
                    'mean_value': float(np.mean(values)),
                    'trend': float(trend)
                }
                
        metrics.update({
            'action_metrics': action_trends,
            'policy_coherence': self._compute_policy_coherence(),
            'goal_achievement': self._compute_goal_achievement()
        })
        
        return metrics
        
    def _compute_policy_coherence(self) -> float:
        """Compute coherence of policies across nodes"""
        if not self.nodes:
            return 0.0
            
        # Collect all policy values
        all_values = []
        for node_id, node in self.nodes.items():
            values = [p.value for p in node.active_inference.policies]
            all_values.append(values)
            
        # Compute correlation between node policies
        correlations = []
        for i in range(len(all_values)):
            for j in range(i + 1, len(all_values)):
                corr = np.corrcoef(all_values[i], all_values[j])[0,1]
                correlations.append(corr)
                
        return float(np.mean(correlations)) if correlations else 0.0
        
    def _compute_goal_achievement(self) -> float:
        """Compute average goal achievement across nodes"""
        if not hasattr(self.active_inference, 'global_goal_state'):
            return 0.0
            
        # Compute distance to goal for each node's latest pattern
        distances = []
        for node_id, node in self.nodes.items():
            if node.local_patterns:
                latest_pattern = list(node.local_patterns.values())[-1]['pattern']
                distance = np.mean((latest_pattern - 
                                  self.active_inference.global_goal_state) ** 2)
                distances.append(distance)
                
        return float(1.0 / (1.0 + np.mean(distances))) if distances else 0.0 