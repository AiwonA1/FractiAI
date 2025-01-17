"""
ActiveInference.py

Implements Active Inference for FractiAI agents, enabling goal-directed behavior through
policy selection based on expected free energy minimization.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
from Methods.FreeEnergyPrinciple import FreeEnergyMinimizer, FEPConfig

logger = logging.getLogger(__name__)

@dataclass
class ActiveInferenceConfig:
    """Configuration for Active Inference"""
    n_policies: int = 10  # Number of policies to consider
    policy_horizon: int = 5  # Time horizon for policy evaluation
    exploration_factor: float = 1.0  # Controls exploration vs exploitation
    goal_precision: float = 2.0  # Precision of goal-directed behavior

class Policy:
    """Represents a temporal sequence of actions"""
    
    def __init__(self, action_dim: int, horizon: int):
        self.actions = np.random.randn(horizon, action_dim)
        self.value = 0.0  # Expected free energy
        
    def sample_action(self, t: int) -> np.ndarray:
        """Sample action at time t"""
        return self.actions[t]

class ActiveInferenceAgent:
    """Implements active inference for goal-directed behavior"""
    
    def __init__(self, config: ActiveInferenceConfig, fep_config: FEPConfig):
        self.config = config
        self.fep = FreeEnergyMinimizer(fep_config)
        self.policies = []
        self.action_dim = fep_config.state_dim
        self.initialize_policies()
        
    def initialize_policies(self) -> None:
        """Initialize set of policies"""
        self.policies = [
            Policy(self.action_dim, self.config.policy_horizon)
            for _ in range(self.config.n_policies)
        ]
        
    def infer_state(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Infer current state using FEP"""
        return self.fep.infer_state(observation)
        
    def select_action(self, observation: np.ndarray, goal_state: np.ndarray) -> np.ndarray:
        """Select action using active inference"""
        # Infer current state
        current_state, uncertainty = self.infer_state(observation)
        
        # Evaluate policies
        self._evaluate_policies(current_state, goal_state)
        
        # Sample policy based on expected free energy
        selected_policy = self._sample_policy()
        
        # Return first action from selected policy
        return selected_policy.sample_action(0)
    
    def _evaluate_policies(self, current_state: np.ndarray, 
                         goal_state: np.ndarray) -> None:
        """Evaluate expected free energy for each policy"""
        for policy in self.policies:
            policy.value = self._compute_expected_free_energy(
                current_state, goal_state, policy)
    
    def _compute_expected_free_energy(self, current_state: np.ndarray,
                                    goal_state: np.ndarray,
                                    policy: Policy) -> float:
        """Compute expected free energy for a policy"""
        expected_free_energy = 0.0
        state = current_state.copy()
        
        for t in range(self.config.policy_horizon):
            # Get action from policy
            action = policy.sample_action(t)
            
            # Predict next state
            next_state = self._predict_next_state(state, action)
            
            # Generate expected observation
            expected_obs = self.fep.model.generate_observation(next_state)
            
            # Compute expected free energy components
            ambiguity = self._compute_ambiguity(next_state)
            risk = self._compute_risk(next_state, goal_state)
            
            # Accumulate expected free energy
            expected_free_energy += ambiguity + self.config.goal_precision * risk
            
            state = next_state
            
        return float(expected_free_energy)
    
    def _predict_next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Predict next state given current state and action"""
        # Combine state transition with action influence
        next_state = self.fep.model.predict_state(state)
        return next_state + action
    
    def _compute_ambiguity(self, state: np.ndarray) -> float:
        """Compute ambiguity (expected uncertainty) of future state"""
        # Expected entropy of observations given state
        obs_cov = np.linalg.inv(self.fep.model.Pi_w)
        entropy = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * obs_cov))
        return float(entropy)
    
    def _compute_risk(self, predicted_state: np.ndarray, 
                     goal_state: np.ndarray) -> float:
        """Compute risk (divergence from goal state)"""
        # KL divergence from predicted state to goal state
        diff = predicted_state - goal_state
        return float(0.5 * np.dot(diff.T, np.dot(self.fep.model.Pi_z, diff)))
    
    def _sample_policy(self) -> Policy:
        """Sample policy based on expected free energy"""
        # Convert expected free energies to probabilities
        energies = np.array([p.value for p in self.policies])
        probabilities = np.exp(-self.config.exploration_factor * energies)
        probabilities /= np.sum(probabilities)
        
        # Sample policy
        idx = np.random.choice(len(self.policies), p=probabilities)
        return self.policies[idx]
    
    def update_model(self, observations: List[np.ndarray], 
                    actions: List[np.ndarray],
                    states: List[np.ndarray]) -> None:
        """Update internal models based on experience"""
        self.fep.update_model(observations, states)
        
        # Update policy distribution based on successful actions
        if len(actions) > 1:
            successful_policy = Policy(self.action_dim, len(actions))
            successful_policy.actions = np.array(actions)
            
            # Replace worst performing policy
            worst_idx = np.argmax([p.value for p in self.policies])
            self.policies[worst_idx] = successful_policy 