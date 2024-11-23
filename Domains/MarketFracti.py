"""
MarketFracti.py

Implements FractiAI principles for financial market analysis and prediction,
enabling fractal-based understanding of market dynamics and patterns.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy.stats import entropy
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class MarketConfig:
    """Configuration for market system"""
    num_assets: int = 100
    history_length: int = 1000
    volatility_threshold: float = 0.2
    correlation_window: int = 50
    fractal_depth: int = 5
    risk_tolerance: float = 0.7

class Asset:
    """Individual financial asset with fractal price dynamics"""
    
    def __init__(self, asset_id: str, initial_price: float):
        self.asset_id = asset_id
        self.price = initial_price
        self.returns = deque(maxlen=1000)
        self.volatility = 0.0
        self.correlations = {}
        self.fractal_metrics = {}
        
    def update_price(self, market_factors: Dict[str, float], 
                    sentiment: float) -> float:
        """Update asset price based on market factors and sentiment"""
        # Compute price change using fractal dynamics
        factor_impact = np.mean(list(market_factors.values()))
        noise = np.random.normal(0, self.volatility)
        
        # Update price with fractal scaling
        price_change = (factor_impact + sentiment + noise) * self.price
        self.price += price_change
        
        # Store return
        if len(self.returns) > 0:
            self.returns.append(price_change / self.price)
            
        # Update volatility
        if len(self.returns) > 1:
            self.volatility = np.std(list(self.returns))
            
        return price_change
    
    def compute_correlation(self, other: 'Asset') -> float:
        """Compute correlation with another asset"""
        if len(self.returns) < 2:
            return 0.0
            
        corr = np.corrcoef(list(self.returns), list(other.returns))[0,1]
        self.correlations[other.asset_id] = corr
        return float(corr)
    
    def analyze_patterns(self) -> Dict[str, float]:
        """Analyze price patterns using fractal metrics"""
        if len(self.returns) < 10:
            return {}
            
        returns = np.array(list(self.returns))
        self.fractal_metrics = {
            'hurst_exponent': self._compute_hurst(returns),
            'fractal_dimension': self._compute_fractal_dimension(returns),
            'multifractality': self._compute_multifractality(returns)
        }
        return self.fractal_metrics
    
    def _compute_hurst(self, returns: np.ndarray) -> float:
        """Compute Hurst exponent for returns"""
        lags = range(2, min(20, len(returns) // 2))
        tau = [np.std(np.subtract(returns[lag:], returns[:-lag])) 
               for lag in lags]
        
        if not tau or len(lags) < 2:
            return 0.5
            
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return float(reg[0] / 2.0)
    
    def _compute_fractal_dimension(self, returns: np.ndarray) -> float:
        """Compute fractal dimension of return series"""
        if len(returns) < 4:
            return 1.0
            
        # Box counting dimension
        scales = [2, 4, 8]
        counts = []
        
        for scale in scales:
            boxes = np.floor(returns * scale)
            count = len(np.unique(boxes))
            counts.append(count)
            
        if len(counts) > 1:
            coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
            return float(-coeffs[0])
        return 1.0
    
    def _compute_multifractality(self, returns: np.ndarray) -> float:
        """Compute degree of multifractality"""
        q_values = [-2, 0, 2]  # Different moments
        taus = []
        
        for q in q_values:
            if q != 0:
                tau = np.mean(np.abs(returns) ** q)
            else:
                tau = np.exp(np.mean(np.log(np.abs(returns) + 1e-10)))
            taus.append(tau)
            
        return float(np.std(taus))

class MarketSystem:
    """Financial market system with fractal dynamics"""
    
    def __init__(self, config: MarketConfig):
        self.config = config
        self.assets: Dict[str, Asset] = {}
        self.market_factors = {}
        self.sentiment_history = deque(maxlen=config.history_length)
        self.systemic_risk = 0.0
        self.initialize_market()
        
    def initialize_market(self) -> None:
        """Initialize market components"""
        # Create assets
        for i in range(self.config.num_assets):
            asset_id = f"ASSET_{i}"
            initial_price = 100 * (1 + np.random.randn() * 0.1)
            self.assets[asset_id] = Asset(asset_id, initial_price)
            
        # Initialize market factors
        self._initialize_market_factors()
        
    def _initialize_market_factors(self) -> None:
        """Initialize market-wide factors"""
        self.market_factors = {
            'economic_growth': 0.02,
            'interest_rate': 0.03,
            'market_sentiment': 0.0,
            'systemic_risk': 0.1
        }
        
    def simulate(self, steps: int) -> Dict[str, Any]:
        """Run market simulation"""
        metrics = []
        price_history = []
        
        for _ in range(steps):
            step_metrics = self._simulate_step()
            metrics.append(step_metrics)
            
            # Record price history
            prices = self._get_price_snapshot()
            price_history.append(prices)
            
            # Update market state
            self._update_market()
            
        return self._analyze_simulation(metrics, price_history)
    
    def _simulate_step(self) -> Dict[str, float]:
        """Simulate one market step"""
        asset_metrics = {}
        
        # Update market sentiment
        sentiment = self._compute_market_sentiment()
        self.sentiment_history.append(sentiment)
        
        # Update each asset
        for asset_id, asset in self.assets.items():
            price_change = asset.update_price(
                self.market_factors,
                sentiment
            )
            
            asset_metrics[asset_id] = {
                'price_change': price_change,
                'volatility': asset.volatility,
                'patterns': asset.analyze_patterns()
            }
            
        # Compute system-wide metrics
        metrics = {
            'market_sentiment': sentiment,
            'systemic_risk': self._compute_systemic_risk(),
            'market_efficiency': self._compute_market_efficiency(asset_metrics)
        }
        
        return metrics
    
    def _update_market(self) -> None:
        """Update market state"""
        # Update market factors
        self._update_market_factors()
        
        # Update correlations
        self._update_correlations()
        
        # Update systemic risk
        self.systemic_risk = self._compute_systemic_risk()
        
    def _update_market_factors(self) -> None:
        """Update market-wide factors"""
        for factor in self.market_factors:
            # Add random walk with mean reversion
            change = np.random.normal(0, 0.01)
            self.market_factors[factor] *= (1 + change)
            
    def _update_correlations(self) -> None:
        """Update asset correlations"""
        for asset_id, asset in self.assets.items():
            for other_id, other_asset in self.assets.items():
                if asset_id != other_id:
                    asset.compute_correlation(other_asset)
                    
    def _compute_market_sentiment(self) -> float:
        """Compute overall market sentiment"""
        # Combine price momentum and volatility
        price_changes = [
            (asset.price - asset.returns[-1]) / asset.price
            if len(asset.returns) > 0 else 0.0
            for asset in self.assets.values()
        ]
        
        momentum = np.mean(price_changes)
        avg_volatility = np.mean([a.volatility for a in self.assets.values()])
        
        sentiment = momentum * (1 - avg_volatility)
        return float(sentiment)
    
    def _compute_systemic_risk(self) -> float:
        """Compute system-wide risk level"""
        # Consider correlations and volatilities
        correlations = []
        for asset in self.assets.values():
            correlations.extend(list(asset.correlations.values()))
            
        avg_correlation = np.mean(correlations) if correlations else 0.0
        avg_volatility = np.mean([a.volatility for a in self.assets.values()])
        
        return float(avg_correlation * avg_volatility)
    
    def _compute_market_efficiency(self, 
                                 metrics: Dict[str, Dict]) -> float:
        """Compute market efficiency measure"""
        # Use entropy of price changes as efficiency measure
        price_changes = [m['price_change'] for m in metrics.values()]
        if not price_changes:
            return 0.0
            
        # Normalize price changes
        normalized = np.abs(price_changes) / (np.max(np.abs(price_changes)) + 1e-10)
        return float(entropy(normalized + 1e-10))
    
    def _get_price_snapshot(self) -> Dict[str, float]:
        """Get current price snapshot"""
        return {asset_id: asset.price 
                for asset_id, asset in self.assets.items()}
    
    def _analyze_simulation(self, metrics: List[Dict[str, float]], 
                          price_history: List[Dict[str, float]]) -> Dict[str, Any]:
        """Analyze simulation results"""
        return {
            'market_metrics': self._analyze_market_metrics(metrics),
            'price_dynamics': self._analyze_price_dynamics(price_history),
            'risk_analysis': self._analyze_risk_metrics(metrics)
        }
    
    def _analyze_market_metrics(self, 
                              metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Analyze market-wide metrics"""
        sentiments = [m['market_sentiment'] for m in metrics]
        risks = [m['systemic_risk'] for m in metrics]
        
        return {
            'sentiment_stability': float(1.0 - np.std(sentiments) / 
                                      (np.mean(np.abs(sentiments)) + 1e-7)),
            'risk_trend': float(np.mean(np.diff(risks))),
            'market_efficiency': float(np.mean([m['market_efficiency'] 
                                              for m in metrics]))
        }
    
    def _analyze_price_dynamics(self, 
                              price_history: List[Dict[str, float]]) -> Dict[str, float]:
        """Analyze price dynamics"""
        # Convert to numpy array for efficient computation
        prices = np.array([[p[asset_id] for asset_id in self.assets.keys()]
                          for p in price_history])
        
        return {
            'price_volatility': float(np.mean(np.std(prices, axis=0))),
            'return_autocorrelation': self._compute_autocorrelation(prices),
            'cross_sectional_dispersion': float(np.mean(np.std(prices, axis=1)))
        }
    
    def _analyze_risk_metrics(self, 
                            metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Analyze risk-related metrics"""
        risks = np.array([m['systemic_risk'] for m in metrics])
        
        return {
            'risk_level': float(np.mean(risks)),
            'risk_volatility': float(np.std(risks)),
            'tail_risk': float(np.percentile(risks, 95))
        }
    
    def _compute_autocorrelation(self, prices: np.ndarray) -> float:
        """Compute return autocorrelation"""
        if len(prices) < 2:
            return 0.0
            
        returns = np.diff(prices, axis=0) / prices[:-1]
        autocorr = np.array([np.corrcoef(returns[:-1, i], returns[1:, i])[0,1]
                           for i in range(returns.shape[1])])
        
        return float(np.mean(np.abs(autocorr[~np.isnan(autocorr)]))) 