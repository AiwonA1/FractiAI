"""
ActiveFractiference.py

Implements active inference principles for FractiAI, enabling predictive processing
and belief updating through fractal patterns and variational inference.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy.stats import entropy
from scipy.special import softmax

logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """Configuration for active inference system"""
    state_dim: int = 64
    action_dim: int = 32
    belief_dim: int = 128
    precision: float = 1.0
    learning_rate: float = 0.01
    temporal_horizon: int = 5
    fractal_depth: int = 3

class GenerativeModel:
    """Fractal-based generative model for state and observation prediction"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.state_transitions = np.random.randn(config.state_dim, config.state_dim)
        self.observation_map = np.random.randn(config.state_dim, config.belief_dim)
        self.action_effects = np.random.randn(config.action_dim, config.state_dim)
        self.precision_matrix = np.eye(config.belief_dim) * config.precision
        
    def predict_state(self, current_state: np.ndarray, 
                     action: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict next state given current state and action"""
        prediction = np.dot(current_state, self.state_transitions.T)
        
        if action is not None:
            action_effect = np.dot(action, self.action_effects)
            prediction += action_effect
            
        return prediction
        
    def predict_observation(self, state: np.ndarray) -> np.ndarray:
        """Predict observation from state"""
        return np.dot(state, self.observation_map)
        
    def update_model(self, prediction_error: np.ndarray, 
                    learning_rate: Optional[float] = None) -> None:
        """Update model parameters based on prediction error"""
        if learning_rate is None:
            learning_rate = self.config.learning_rate
            
        # Update transition and observation mappings
        self.state_transitions -= learning_rate * prediction_error
        self.observation_map -= learning_rate * prediction_error
        
        # Update precision matrix
        precision_error = np.outer(prediction_error, prediction_error) - self.precision_matrix
        self.precision_matrix += learning_rate * precision_error

class BeliefState:
    """Represents beliefs about system state with uncertainty"""
    
    def __init__(self, dim: int):
        self.mean = np.zeros(dim)
        self.variance = np.ones(dim)
        self.history = []
        
    def update(self, observation: np.ndarray, 
              prediction: np.ndarray, 
              precision: np.ndarray) -> None:
        """Update beliefs using precision-weighted prediction errors"""
        prediction_error = observation - prediction
        kalman_gain = self.variance * precision / \
                     (self.variance * precision + 1)
                     
        # Update mean and variance
        self.mean += kalman_gain * prediction_error
        self.variance *= (1 - kalman_gain)
        
        # Store history
        self.history.append({
            'mean': self.mean.copy(),
            'variance': self.variance.copy(),
            'prediction_error': prediction_error
        })
        
    def sample(self) -> np.ndarray:
        """Sample from belief distribution"""
        return np.random.normal(self.mean, np.sqrt(self.variance))

class PolicySelection:
    """Selects actions using active inference principles"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.policy_library = []
        self.expected_free_energy = []
        
    def evaluate_policies(self, current_belief: BeliefState,
                         generative_model: GenerativeModel,
                         goal_state: Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate policies using expected free energy"""
        policy_values = []
        
        for policy in self.policy_library:
            # Simulate policy execution
            predicted_states = self._simulate_policy(
                policy, 
                current_belief,
                generative_model
            )
            
            # Compute expected free energy
            efe = self._compute_expected_free_energy(
                predicted_states,
                goal_state,
                generative_model
            )
            
            policy_values.append(-efe)  # Negative because we want to minimize free energy
            
        return softmax(np.array(policy_values))
        
    def _simulate_policy(self, policy: List[np.ndarray],
                        belief: BeliefState,
                        model: GenerativeModel) -> List[np.ndarray]:
        """Simulate policy execution for temporal horizon"""
        states = []
        current_state = belief.mean
        
        for action in policy[:self.config.temporal_horizon]:
            next_state = model.predict_state(current_state, action)
            states.append(next_state)
            current_state = next_state
            
        return states
        
    def _compute_expected_free_energy(self, 
                                    predicted_states: List[np.ndarray],
                                    goal_state: Optional[np.ndarray],
                                    model: GenerativeModel) -> float:
        """Compute expected free energy for predicted states"""
        total_efe = 0.0
        
        for state in predicted_states:
            # Compute predicted observation
            predicted_obs = model.predict_observation(state)
            
            # Ambiguity term (entropy of predicted observations)
            ambiguity = entropy(softmax(predicted_obs))
            
            # Risk term (if goal state is provided)
            risk = 0.0
            if goal_state is not None:
                risk = np.sum((state - goal_state) ** 2)
                
            total_efe += ambiguity + risk
            
        return total_efe / len(predicted_states)
    
    def add_policy(self, policy: List[np.ndarray]) -> None:
        """Add new policy to library"""
        self.policy_library.append(policy)

class ActiveInference:
    """Main active inference system with fractal patterns"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.generative_model = GenerativeModel(config)
        self.belief_state = BeliefState(config.state_dim)
        self.policy_selection = PolicySelection(config)
        self.action_history = []
        self.prediction_errors = []
        
    def update_beliefs(self, observation: np.ndarray) -> None:
        """Update beliefs based on new observation"""
        # Generate predictions
        predicted_state = self.generative_model.predict_state(self.belief_state.mean)
        predicted_obs = self.generative_model.predict_observation(predicted_state)
        
        # Update beliefs
        self.belief_state.update(
            observation,
            predicted_obs,
            self.generative_model.precision_matrix
        )
        
        # Store prediction error
        self.prediction_errors.append(
            np.mean((observation - predicted_obs) ** 2)
        )
        
    def select_action(self, goal_state: Optional[np.ndarray] = None) -> np.ndarray:
        """Select action using active inference"""
        # Evaluate policies
        policy_probs = self.policy_selection.evaluate_policies(
            self.belief_state,
            self.generative_model,
            goal_state
        )
        
        # Sample policy
        selected_idx = np.random.choice(len(self.policy_selection.policy_library),
                                      p=policy_probs)
        selected_policy = self.policy_selection.policy_library[selected_idx]
        
        # Store selected action
        self.action_history.append(selected_policy[0])
        
        return selected_policy[0]
    
    def learn_from_experience(self, learning_rate: Optional[float] = None) -> None:
        """Update model based on accumulated experience"""
        if not self.prediction_errors:
            return
            
        # Compute average prediction error
        mean_error = np.mean(self.prediction_errors)
        
        # Update generative model
        self.generative_model.update_model(
            mean_error,
            learning_rate
        )
        
        # Clear history
        self.prediction_errors = []
        
    def get_inference_metrics(self) -> Dict[str, float]:
        """Get metrics about inference performance"""
        if not self.belief_state.history:
            return {}
            
        belief_history = self.belief_state.history
        prediction_errors = [h['prediction_error'] for h in belief_history]
        
        return {
            'mean_prediction_error': float(np.mean(np.abs(prediction_errors))),
            'belief_uncertainty': float(np.mean(self.belief_state.variance)),
            'learning_progress': float(np.std([h['mean'] for h in belief_history])),
            'model_complexity': self._compute_model_complexity()
        }
        
    def _compute_model_complexity(self) -> float:
        """Compute complexity of generative model"""
        # Use eigenvalue spread as complexity measure
        transition_eigs = np.linalg.eigvals(self.generative_model.state_transitions)
        observation_eigs = np.linalg.eigvals(
            np.dot(self.generative_model.observation_map,
                  self.generative_model.observation_map.T)
        )
        
        # Compute spectral entropy
        transition_entropy = entropy(softmax(np.abs(transition_eigs)))
        observation_entropy = entropy(softmax(np.abs(observation_eigs)))
        
        return float(transition_entropy + observation_entropy)
