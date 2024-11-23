"""
CryptoFracti.py

Implements FractiAI principles for cryptocurrency analysis and prediction,
enabling fractal-based understanding of crypto market dynamics and patterns.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from collections import deque
from scipy.stats import entropy

logger = logging.getLogger(__name__)

@dataclass
class CryptoConfig:
    """Configuration for crypto system"""
    num_tokens: int = 50
    history_length: int = 1000
    volatility_window: int = 20
    network_depth: int = 3
    fractal_scale: float = 0.7
    liquidity_threshold: float = 0.5

class Token:
    """Individual cryptocurrency with fractal dynamics"""
    
    def __init__(self, token_id: str, initial_price: float):
        self.token_id = token_id
        self.price = initial_price
        self.volume = 0.0
        self.liquidity = 0.0
        self.price_history = deque(maxlen=1000)
        self.volume_history = deque(maxlen=1000)
        self.network_metrics = {}
        self.fractal_patterns = {}
        
    def update(self, market_data: Dict[str, float], 
              network_state: Dict[str, float]) -> Dict[str, float]:
        """Update token state based on market and network data"""
        # Update price using fractal dynamics
        price_change = self._compute_price_change(market_data, network_state)
        self.price *= (1 + price_change)
        
        # Update volume and liquidity
        self.volume = self._compute_volume(market_data)
        self.liquidity = self._compute_liquidity()
        
        # Store history
        self.price_history.append(self.price)
        self.volume_history.append(self.volume)
        
        # Update metrics
        self._update_metrics(market_data)
        
        return {
            'price_change': price_change,
            'volume': self.volume,
            'liquidity': self.liquidity
        }
        
    def _compute_price_change(self, market_data: Dict[str, float],
                            network_state: Dict[str, float]) -> float:
        """Compute price change using fractal patterns"""
        market_influence = np.mean(list(market_data.values()))
        network_influence = np.mean(list(network_state.values()))
        
        # Combine influences with fractal scaling
        base_change = (market_influence + network_influence) / 2
        noise = np.random.normal(0, self._compute_volatility())
        
        return base_change + noise
        
    def _compute_volume(self, market_data: Dict[str, float]) -> float:
        """Compute trading volume"""
        base_volume = market_data.get('market_volume', 0) * \
                     market_data.get('token_share', 1)
        return base_volume * (1 + np.random.normal(0, 0.1))
        
    def _compute_liquidity(self) -> float:
        """Compute token liquidity"""
        if len(self.volume_history) < 2:
            return 0.0
            
        recent_volumes = list(self.volume_history)[-10:]
        return np.mean(recent_volumes) / (self.price + 1e-10)
        
    def _compute_volatility(self) -> float:
        """Compute price volatility"""
        if len(self.price_history) < 2:
            return 0.1
            
        returns = np.diff(list(self.price_history)) / \
                 np.array(list(self.price_history)[:-1])
        return float(np.std(returns))
        
    def _update_metrics(self, market_data: Dict[str, float]) -> None:
        """Update token metrics"""
        self.network_metrics = {
            'centrality': market_data.get('network_centrality', 0),
            'connectivity': market_data.get('network_connectivity', 0),
            'stability': self._compute_stability()
        }
        
        self.fractal_patterns = {
            'hurst_exponent': self._compute_hurst_exponent(),
            'fractal_dimension': self._compute_fractal_dimension(),
            'pattern_strength': self._compute_pattern_strength()
        }
        
    def _compute_stability(self) -> float:
        """Compute token stability metric"""
        if len(self.price_history) < 10:
            return 1.0
            
        prices = np.array(list(self.price_history))
        return float(1.0 - np.std(prices) / (np.mean(prices) + 1e-10))
        
    def _compute_hurst_exponent(self) -> float:
        """Compute Hurst exponent of price series"""
        if len(self.price_history) < 20:
            return 0.5
            
        prices = np.array(list(self.price_history))
        lags = range(2, min(20, len(prices) // 2))
        tau = [np.std(np.subtract(prices[lag:], prices[:-lag])) 
               for lag in lags]
        
        if not tau or len(lags) < 2:
            return 0.5
            
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return float(reg[0] / 2.0)
        
    def _compute_fractal_dimension(self) -> float:
        """Compute fractal dimension of price series"""
        if len(self.price_history) < 10:
            return 1.0
            
        prices = np.array(list(self.price_history))
        scales = [2, 4, 8]
        counts = []
        
        for scale in scales:
            scaled = np.floor(prices * scale)
            counts.append(len(np.unique(scaled)))
            
        if len(counts) > 1:
            coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
            return float(-coeffs[0])
        return 1.0
        
    def _compute_pattern_strength(self) -> float:
        """Compute strength of fractal patterns"""
        if len(self.price_history) < 20:
            return 0.0
            
        prices = np.array(list(self.price_history))
        patterns = []
        
        for i in range(len(prices) - 10):
            pattern = prices[i:i+10]
            patterns.append(pattern / np.mean(pattern))
            
        similarities = []
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                similarity = np.corrcoef(patterns[i], patterns[j])[0,1]
                similarities.append(abs(similarity))
                
        return float(np.mean(similarities))

class CryptoMarket:
    """Cryptocurrency market with fractal dynamics"""
    
    def __init__(self, config: CryptoConfig):
        self.config = config
        self.tokens: Dict[str, Token] = {}
        self.market_state = {}
        self.network_state = {}
        self.metrics_history = []
        self.initialize_market()
        
    def initialize_market(self) -> None:
        """Initialize crypto market"""
        # Create tokens
        for i in range(self.config.num_tokens):
            token_id = f"TOKEN_{i}"
            initial_price = 100 * (1 + np.random.randn() * 0.1)
            self.tokens[token_id] = Token(token_id, initial_price)
            
        # Initialize market state
        self._initialize_market_state()
        
        # Initialize network state
        self._initialize_network_state()
        
    def _initialize_market_state(self) -> None:
        """Initialize market-wide state"""
        self.market_state = {
            'market_volume': 1000000,
            'market_sentiment': 0.0,
            'global_volatility': 0.1,
            'network_health': 1.0
        }
        
    def _initialize_network_state(self) -> None:
        """Initialize network state"""
        for token in self.tokens.values():
            self.network_state[token.token_id] = {
                'network_centrality': np.random.random(),
                'network_connectivity': np.random.random(),
                'token_share': 1.0 / self.config.num_tokens
            }
            
    def simulate(self, steps: int) -> Dict[str, Any]:
        """Run market simulation"""
        metrics = []
        state_history = []
        
        for _ in range(steps):
            step_metrics = self._simulate_step()
            metrics.append(step_metrics)
            
            # Record state
            state = self._get_market_snapshot()
            state_history.append(state)
            
            # Update market
            self._update_market()
            
        return self._analyze_simulation(metrics, state_history)
        
    def _simulate_step(self) -> Dict[str, float]:
        """Simulate one market step"""
        token_metrics = {}
        
        # Update each token
        for token_id, token in self.tokens.items():
            market_data = {**self.market_state, 
                          **self.network_state[token_id]}
            
            metrics = token.update(
                market_data,
                self.network_state[token_id]
            )
            
            token_metrics[token_id] = metrics
            
        # Compute market-wide metrics
        metrics = {
            'market_volume': float(np.sum([m['volume'] for m in token_metrics.values()])),
            'market_liquidity': float(np.mean([m['liquidity'] for m in token_metrics.values()])),
            'price_volatility': self._compute_market_volatility()
        }
        
        return metrics
        
    def _update_market(self) -> None:
        """Update market state"""
        # Update market factors
        self._update_market_factors()
        
        # Update network state
        self._update_network_state()
        
    def _update_market_factors(self) -> None:
        """Update market-wide factors"""
        self.market_state['market_sentiment'] = self._compute_market_sentiment()
        self.market_state['global_volatility'] = self._compute_global_volatility()
        self.market_state['network_health'] = self._compute_network_health()
        
    def _update_network_state(self) -> None:
        """Update network state"""
        total_volume = sum(t.volume for t in self.tokens.values())
        
        for token_id, token in self.tokens.items():
            self.network_state[token_id]['token_share'] = \
                token.volume / (total_volume + 1e-10)
            
            self.network_state[token_id]['network_centrality'] = \
                self._compute_token_centrality(token)
            
            self.network_state[token_id]['network_connectivity'] = \
                self._compute_token_connectivity(token)
            
    def _compute_market_sentiment(self) -> float:
        """Compute overall market sentiment"""
        price_changes = [
            (token.price - token.price_history[0]) / token.price_history[0]
            if len(token.price_history) > 0 else 0.0
            for token in self.tokens.values()
        ]
        
        return float(np.mean(price_changes))
        
    def _compute_global_volatility(self) -> float:
        """Compute market-wide volatility"""
        volatilities = [token._compute_volatility() 
                       for token in self.tokens.values()]
        return float(np.mean(volatilities))
        
    def _compute_network_health(self) -> float:
        """Compute overall network health"""
        metrics = []
        for token in self.tokens.values():
            metrics.extend([
                token.network_metrics.get('stability', 0),
                token.network_metrics.get('connectivity', 0)
            ])
        return float(np.mean(metrics))
        
    def _compute_token_centrality(self, token: Token) -> float:
        """Compute token's centrality in network"""
        volume_share = token.volume / \
                      (sum(t.volume for t in self.tokens.values()) + 1e-10)
        price_impact = abs(token.price_history[-1] - token.price_history[0]) \
                      if len(token.price_history) > 1 else 0
                      
        return float((volume_share + price_impact) / 2)
        
    def _compute_token_connectivity(self, token: Token) -> float:
        """Compute token's network connectivity"""
        correlations = []
        for other in self.tokens.values():
            if other.token_id != token.token_id:
                corr = np.corrcoef(
                    list(token.price_history),
                    list(other.price_history)
                )[0,1]
                correlations.append(abs(corr))
                
        return float(np.mean(correlations))
        
    def _compute_market_volatility(self) -> float:
        """Compute market-wide volatility"""
        return float(np.mean([token._compute_volatility() 
                            for token in self.tokens.values()]))
        
    def _get_market_snapshot(self) -> Dict[str, Dict[str, float]]:
        """Get current market state snapshot"""
        return {
            'tokens': {
                token_id: {
                    'price': token.price,
                    'volume': token.volume,
                    'liquidity': token.liquidity
                }
                for token_id, token in self.tokens.items()
            },
            'market_state': self.market_state.copy(),
            'network_state': {
                token_id: state.copy()
                for token_id, state in self.network_state.items()
            }
        }
        
    def _analyze_simulation(self, metrics: List[Dict[str, float]],
                          states: List[Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
        """Analyze simulation results"""
        return {
            'market_metrics': self._analyze_market_metrics(metrics),
            'network_metrics': self._analyze_network_metrics(states),
            'token_metrics': self._analyze_token_metrics(states)
        }
        
    def _analyze_market_metrics(self, 
                              metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Analyze market-wide metrics"""
        volumes = [m['market_volume'] for m in metrics]
        volatilities = [m['price_volatility'] for m in metrics]
        
        return {
            'volume_growth': float(np.mean(np.diff(volumes)) / np.mean(volumes)),
            'volatility_stability': float(1.0 - np.std(volatilities) / 
                                       (np.mean(volatilities) + 1e-10)),
            'market_efficiency': float(np.mean([m['market_liquidity'] 
                                              for m in metrics]))
        }
        
    def _analyze_network_metrics(self, 
                               states: List[Dict[str, Dict[str, float]]]) -> Dict[str, float]:
        """Analyze network metrics"""
        network_states = [state['network_state'] for state in states]
        
        centralities = [[s[token_id]['network_centrality'] 
                        for token_id in self.tokens.keys()]
                       for s in network_states]
        
        connectivities = [[s[token_id]['network_connectivity']
                          for token_id in self.tokens.keys()]
                         for s in network_states]
                         
        return {
            'centralization': float(np.mean([np.std(c) for c in centralities])),
            'connectivity': float(np.mean([np.mean(c) for c in connectivities])),
            'network_stability': float(1.0 - np.std([np.mean(c) 
                                                    for c in connectivities]))
        }
        
    def _analyze_token_metrics(self, 
                             states: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        """Analyze individual token metrics"""
        token_metrics = {}
        
        for token_id in self.tokens:
            prices = [s['tokens'][token_id]['price'] for s in states]
            volumes = [s['tokens'][token_id]['volume'] for s in states]
            
            token_metrics[token_id] = {
                'price_stability': float(1.0 - np.std(prices) / (np.mean(prices) + 1e-10)),
                'volume_growth': float(np.mean(np.diff(volumes)) / np.mean(volumes)),
                'fractal_metrics': self.tokens[token_id].fractal_patterns
            }
            
        return token_metrics
