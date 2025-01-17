"""
FreeEnergyPrinciple.py

Implements the Free Energy Principle (FEP) for FractiAI agents, providing a mathematical
framework for perception, learning, and action based on minimizing variational free energy.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy.stats import multivariate_normal

logger = logging.getLogger(__name__)

@dataclass
class FEPConfig:
    """Configuration for Free Energy Principle implementation"""
    state_dim: int = 3  # Dimensions of hidden states
    obs_dim: int = 3    # Dimensions of observations
    temp: float = 1.0   # Temperature parameter for precision
    learning_rate: float = 0.01
    inference_steps: int = 50

class GenerativeModel:
    """Implements the generative model for FEP"""
    
    def __init__(self, state_dim: int, obs_dim: int):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        # Initialize transformation matrices
        self.A = np.random.randn(obs_dim, state_dim)  # Observation model
        self.B = np.eye(state_dim)  # State transition model
        self.Pi_z = np.eye(state_dim)  # Prior precision over states
        self.Pi_w = np.eye(obs_dim)   # Sensory precision
        
    def generate_observation(self, state: np.ndarray) -> np.ndarray:
        """Generate observation from hidden state"""
        mean = np.dot(self.A, state)
        return mean + np.random.multivariate_normal(np.zeros(self.obs_dim), 
                                                  np.linalg.inv(self.Pi_w))
    
    def predict_state(self, prev_state: np.ndarray) -> np.ndarray:
        """Predict next state using transition model"""
        return np.dot(self.B, prev_state)

class FreeEnergyMinimizer:
    """Implements variational free energy minimization"""
    
    def __init__(self, config: FEPConfig):
        self.config = config
        self.model = GenerativeModel(config.state_dim, config.obs_dim)
        self.mu = np.zeros(config.state_dim)  # Expected state
        self.Sigma = np.eye(config.state_dim)  # State uncertainty
        
    def infer_state(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform variational inference to minimize free energy"""
        mu = self.mu.copy()
        Sigma = self.Sigma.copy()
        
        for _ in range(self.config.inference_steps):
            # Compute prediction error
            pred_error = observation - np.dot(self.model.A, mu)
            
            # Update expected state (gradient descent on free energy)
            mu_dot = (np.dot(self.model.A.T, np.dot(self.model.Pi_w, pred_error)) - 
                     np.dot(self.model.Pi_z, mu))
            mu += self.config.learning_rate * mu_dot
            
            # Update uncertainty
            Sigma_dot = (np.dot(self.model.A.T, np.dot(self.model.Pi_w, self.model.A)) + 
                        self.model.Pi_z)
            Sigma = np.linalg.inv(Sigma_dot)
        
        self.mu = mu
        self.Sigma = Sigma
        return mu, Sigma
    
    def compute_free_energy(self, observation: np.ndarray) -> float:
        """Compute variational free energy"""
        # Accuracy term
        pred_error = observation - np.dot(self.model.A, self.mu)
        accuracy = 0.5 * np.dot(pred_error.T, np.dot(self.model.Pi_w, pred_error))
        
        # Complexity term (KL divergence from prior)
        complexity = 0.5 * (np.trace(np.dot(self.model.Pi_z, self.Sigma)) + 
                          np.dot(self.mu.T, np.dot(self.model.Pi_z, self.mu)) -
                          self.config.state_dim + 
                          np.log(np.linalg.det(self.Sigma)))
        
        return float(accuracy + complexity)
    
    def update_model(self, observations: List[np.ndarray], 
                    inferred_states: List[np.ndarray]) -> None:
        """Update generative model parameters"""
        observations = np.array(observations)
        inferred_states = np.array(inferred_states)
        
        # Update observation model (A)
        A_num = np.dot(observations.T, inferred_states)
        A_den = np.dot(inferred_states.T, inferred_states)
        self.model.A = np.dot(A_num, np.linalg.inv(A_den))
        
        # Update state transition model (B)
        if len(inferred_states) > 1:
            B_num = np.dot(inferred_states[1:].T, inferred_states[:-1])
            B_den = np.dot(inferred_states[:-1].T, inferred_states[:-1])
            self.model.B = np.dot(B_num, np.linalg.inv(B_den))
        
        # Update precisions
        pred_errors = observations - np.dot(inferred_states, self.model.A.T)
        self.model.Pi_w = np.linalg.inv(np.cov(pred_errors.T))
        
        state_errors = inferred_states[1:] - np.dot(inferred_states[:-1], self.model.B.T)
        self.model.Pi_z = np.linalg.inv(np.cov(state_errors.T)) 