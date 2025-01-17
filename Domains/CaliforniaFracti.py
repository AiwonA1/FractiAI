"""
CaliforniaFracti.py

Implements a comprehensive California-specific FractiAI simulation that models complex
interactions between natural systems, government, technology, and socioeconomic domains.
Uses quantum field theoretic principles and categorical structures to model emergent
behavior and cross-domain resonance patterns.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import logging
from collections import defaultdict
from enum import Enum, auto
import scipy.stats as stats
import scipy.optimize as optimize

from Methods.QuantumFieldMorphism import (
    QuantumFieldMorphism, FieldMorphismConfig,
    QuantumFieldState, FieldMorphismFunctor
)
from Methods.Method_Integrator import MethodIntegrator, UnipixelConfig

logger = logging.getLogger(__name__)

class RegionType(Enum):
    """California region types"""
    COASTAL = auto()
    INLAND = auto()
    MOUNTAIN = auto()
    DESERT = auto()
    URBAN = auto()
    AGRICULTURAL = auto()

class EcosystemType(Enum):
    """California ecosystem types"""
    COASTAL_SAGE = auto()
    CHAPARRAL = auto()
    OAK_WOODLAND = auto()
    REDWOOD_FOREST = auto()
    ALPINE_FOREST = auto()
    DESERT_SCRUB = auto()
    WETLAND = auto()
    GRASSLAND = auto()

class WaterSystemType(Enum):
    """California water system types"""
    SNOWPACK = auto()
    GROUNDWATER = auto()
    RESERVOIR = auto()
    RIVER = auto()
    DELTA = auto()
    AQUEDUCT = auto()

@dataclass
class CaliforniaDomainConfig:
    """Enhanced configuration for California domain simulation"""
    # Natural Systems
    climate_resolution: int = 64
    ecosystem_types: int = len(EcosystemType)
    water_systems: int = len(WaterSystemType)
    region_types: int = len(RegionType)
    wildfire_risk_levels: int = 5
    drought_severity_levels: int = 4
    
    # Climate Parameters
    baseline_temperature: float = 15.0  # Celsius
    temperature_variation: float = 10.0
    annual_precipitation: float = 600.0  # mm
    drought_threshold: float = 0.3
    
    # Ecosystem Parameters
    biodiversity_index: float = 0.8
    ecosystem_resilience: float = 0.7
    species_interaction_strength: float = 0.4
    
    # Water System Parameters
    reservoir_capacity: float = 1000.0  # Units: billion gallons
    groundwater_recharge_rate: float = 0.02
    water_quality_threshold: float = 0.75
    
    # Government & Policy
    policy_dimensions: int = 32
    agency_count: int = 12
    regulation_types: int = 24
    policy_update_interval: int = 10
    policy_effectiveness_threshold: float = 0.6
    
    # Technology & Innovation
    tech_sectors: int = 16
    innovation_rate: float = 0.1
    startup_density: float = 0.05
    tech_adoption_rate: float = 0.15
    innovation_threshold: float = 0.7
    
    # Socioeconomic
    population_groups: int = 10
    economic_sectors: int = 20
    urban_regions: int = 12
    income_brackets: int = 5
    education_levels: int = 4
    
    # Economic Parameters
    gdp_growth_rate: float = 0.03
    unemployment_threshold: float = 0.05
    inflation_target: float = 0.02
    interest_rate: float = 0.03
    
    # Interaction Parameters
    coupling_strength: float = 0.15
    adaptation_rate: float = 0.05
    resonance_threshold: float = 0.75
    field_dimension: int = 128
    
    # Analysis Parameters
    confidence_level: float = 0.95
    significance_threshold: float = 0.05
    forecast_horizon: int = 52  # weeks
    
    # Utility Functions
    risk_aversion: float = 0.7
    discount_rate: float = 0.05
    social_welfare_weight: float = 0.6

class AdvancedMetrics:
    """Advanced metric computation and analysis"""
    
    @staticmethod
    def compute_resilience(state_history: List[np.ndarray]) -> float:
        """Compute system resilience using state history"""
        if not state_history:
            return 0.0
        
        # Calculate recovery rate from perturbations
        variations = np.diff([np.mean(state) for state in state_history], axis=0)
        recovery_rate = np.mean([1 / (1 + np.abs(v)) for v in variations])
        return float(recovery_rate)
    
    @staticmethod
    def compute_sustainability(
        resource_state: np.ndarray,
        consumption_rate: np.ndarray,
        regeneration_rate: np.ndarray
    ) -> float:
        """Compute sustainability index"""
        sustainability = np.mean(
            resource_state * regeneration_rate / (consumption_rate + 1e-8)
        )
        return float(np.clip(sustainability, 0, 1))
    
    @staticmethod
    def compute_inequality(economic_state: np.ndarray) -> float:
        """Compute Gini coefficient for inequality"""
        sorted_state = np.sort(economic_state)
        n = len(sorted_state)
        index = np.arange(1, n + 1)
        return float(((2 * index - n - 1) * sorted_state).sum() / (n * sorted_state.sum()))
    
    @staticmethod
    def compute_innovation_impact(
        innovation_state: np.ndarray,
        economic_state: np.ndarray,
        environmental_state: np.ndarray
    ) -> Dict[str, float]:
        """Compute innovation impact metrics"""
        economic_impact = np.corrcoef(innovation_state, economic_state)[0, 1]
        environmental_impact = np.corrcoef(innovation_state, environmental_state)[0, 1]
        
        return {
            'economic_impact': float(economic_impact),
            'environmental_impact': float(environmental_impact),
            'overall_impact': float((economic_impact + environmental_impact) / 2)
        }

class CaliforniaDomain:
    """Enhanced base class for California-specific domains"""
    
    def __init__(self, name: str, config: CaliforniaDomainConfig):
        self.name = name
        self.config = config
        self.state = np.zeros(config.field_dimension)
        self.field_state = None
        self.connections = defaultdict(float)
        self.state_history = []
        self.metrics_history = []
        
    def compute_domain_metrics(self) -> Dict[str, float]:
        """Compute domain-specific metrics"""
        metrics = {
            'state_mean': float(np.mean(self.state)),
            'state_std': float(np.std(self.state)),
            'stability': float(self._compute_stability()),
            'trend': float(self._compute_trend()),
            'resilience': float(self._compute_resilience())
        }
        self.metrics_history.append(metrics)
        return metrics
    
    def _compute_stability(self) -> float:
        """Compute stability metric"""
        if len(self.state_history) < 2:
            return 1.0
        variations = np.diff([np.mean(state) for state in self.state_history[-10:]])
        return float(1 / (1 + np.std(variations)))
    
    def _compute_trend(self) -> float:
        """Compute trend direction and strength"""
        if len(self.state_history) < 2:
            return 0.0
        values = [np.mean(state) for state in self.state_history[-10:]]
        slope, _ = np.polyfit(range(len(values)), values, 1)
        return float(slope)
    
    def _compute_resilience(self) -> float:
        """Compute system resilience"""
        return AdvancedMetrics.compute_resilience(self.state_history[-20:])
    
    def forecast_state(self, horizon: int = 10) -> np.ndarray:
        """Forecast future state using time series analysis"""
        if len(self.state_history) < 2:
            return self.state
            
        # Use exponential smoothing for forecasting
        values = np.array([np.mean(state) for state in self.state_history])
        alpha = 0.3  # smoothing factor
        smoothed = [values[0]]
        for n in range(1, len(values)):
            smoothed.append(alpha * values[n] + (1 - alpha) * smoothed[n-1])
        
        # Project forward
        forecast = smoothed[-1] * np.ones(horizon)
        trend = (smoothed[-1] - smoothed[-2])
        for i in range(horizon):
            forecast[i] += trend * i
            
        return forecast

class NaturalSystem(CaliforniaDomain):
    """Enhanced natural systems modeling"""
    
    def __init__(self, config: CaliforniaDomainConfig):
        super().__init__("natural", config)
        
        # Enhanced state initialization
        self.climate_state = np.zeros((config.climate_resolution, 3))
        self.ecosystem_state = np.zeros((config.ecosystem_types, 3))  # health, diversity, resilience
        self.water_state = np.zeros((config.water_systems, 4))  # level, quality, flow, pressure
        
        # Initialize subsystem states
        self.wildfire_risk = np.zeros(config.region_types)
        self.drought_severity = np.zeros(config.region_types)
        self.biodiversity_indices = np.ones(config.ecosystem_types) * config.biodiversity_index
        
        # Enhanced tracking
        self.ecosystem_interactions = np.zeros((config.ecosystem_types, config.ecosystem_types))
        self.water_flow_network = np.zeros((config.water_systems, config.water_systems))
        
    def update_climate(self, season: float) -> None:
        """Enhanced climate update with feedback loops"""
        # Seasonal base patterns
        seasonal_temp = self.config.baseline_temperature + \
                       self.config.temperature_variation * np.sin(2 * np.pi * season)
        seasonal_precip = self.config.annual_precipitation * \
                         (1 + 0.5 * np.cos(2 * np.pi * season))
        
        # Add climate feedback effects
        drought_effect = np.mean(self.drought_severity) * 2.0
        seasonal_temp += drought_effect
        seasonal_precip *= (1 - 0.3 * drought_effect)
        
        # Update climate state with spatial variation
        for i in range(self.config.climate_resolution):
            spatial_variation = 0.1 * np.random.randn()
            self.climate_state[i, 0] = seasonal_temp * (1 + spatial_variation)
            self.climate_state[i, 1] = seasonal_precip * (1 + spatial_variation)
            self.climate_state[i, 2] = 5.0 * np.random.randn()  # wind
            
        # Update drought conditions
        self._update_drought_conditions()
        
        # Update wildfire risk
        self._update_wildfire_risk()
        
    def _update_drought_conditions(self) -> None:
        """Update drought conditions based on climate state"""
        for i in range(self.config.region_types):
            # Compute drought severity based on temperature and precipitation
            temp_factor = np.mean(self.climate_state[:, 0]) / self.config.baseline_temperature
            precip_factor = np.mean(self.climate_state[:, 1]) / self.config.annual_precipitation
            
            drought_index = (temp_factor - precip_factor) / 2
            self.drought_severity[i] = np.clip(
                self.drought_severity[i] + 0.1 * (drought_index - 0.5),
                0, 1
            )
            
    def _update_wildfire_risk(self) -> None:
        """Update wildfire risk based on climate and ecosystem conditions"""
        for i in range(self.config.region_types):
            # Compute risk factors
            temperature_factor = np.mean(self.climate_state[:, 0]) / 30.0  # normalized by 30°C
            wind_factor = np.mean(np.abs(self.climate_state[:, 2])) / 10.0
            drought_factor = self.drought_severity[i]
            
            # Combine risk factors
            risk = (0.4 * temperature_factor + 
                   0.3 * wind_factor +
                   0.3 * drought_factor)
            
            self.wildfire_risk[i] = np.clip(risk, 0, 1)
            
    def compute_ecosystem_health(self) -> Dict[str, float]:
        """Compute comprehensive ecosystem health metrics"""
        metrics = {}
        
        # Overall biodiversity
        metrics['biodiversity'] = float(np.mean(self.biodiversity_indices))
        
        # Ecosystem stability
        stability = np.mean([
            np.std(ecosystem) for ecosystem in self.ecosystem_state
        ])
        metrics['stability'] = float(1 / (1 + stability))
        
        # Ecosystem interactions
        interaction_strength = np.mean(np.abs(self.ecosystem_interactions))
        metrics['interaction_strength'] = float(interaction_strength)
        
        # Water system health
        water_metrics = self.compute_water_system_health()
        metrics.update(water_metrics)
        
        # Risk metrics
        metrics['wildfire_risk'] = float(np.mean(self.wildfire_risk))
        metrics['drought_severity'] = float(np.mean(self.drought_severity))
        
        return metrics
        
    def compute_water_system_health(self) -> Dict[str, float]:
        """Compute water system health metrics"""
        metrics = {}
        
        # Water availability
        metrics['water_availability'] = float(np.mean(self.water_state[:, 0]))
        
        # Water quality
        metrics['water_quality'] = float(np.mean(self.water_state[:, 1]))
        
        # System pressure
        metrics['system_pressure'] = float(np.mean(self.water_state[:, 3]))
        
        # Flow network efficiency
        flow_efficiency = np.mean(
            np.sum(self.water_flow_network, axis=1) /
            (np.sum(self.water_flow_network) + 1e-8)
        )
        metrics['flow_efficiency'] = float(flow_efficiency)
        
        return metrics

class Government(CaliforniaDomain):
    """Models California's government systems including policies and regulations"""
    
    def __init__(self, config: CaliforniaDomainConfig):
        super().__init__("government", config)
        
        # Initialize government subsystems
        self.policy_state = np.zeros(config.policy_dimensions)
        self.agency_state = np.zeros(config.agency_count)
        self.regulation_state = np.zeros(config.regulation_types)
        
        # Set initial connection strengths
        self.connections.update({
            "natural": 0.4,      # Environmental conditions influence
            "technology": 0.35,   # Tech sector influence
            "socioeconomic": 0.4  # Public needs influence
        })
        
    def update_policies(self, public_needs: np.ndarray) -> None:
        """Update policy states based on public needs"""
        # Model policy adaptation
        self.policy_state += self.config.adaptation_rate * (
            public_needs[:self.config.policy_dimensions] - 
            self.policy_state
        )
        
        # Update overall state
        self.state[:self.config.policy_dimensions] = self.policy_state
        
    def update_agencies(self, performance: np.ndarray) -> None:
        """Update agency states based on performance metrics"""
        # Model agency effectiveness
        self.agency_state = (1 - self.config.adaptation_rate) * self.agency_state + \
                           self.config.adaptation_rate * performance
                           
        # Update overall state
        idx = self.config.policy_dimensions
        self.state[idx:idx + self.config.agency_count] = self.agency_state
        
    def update_regulations(self, needs: np.ndarray) -> None:
        """Update regulation states based on identified needs"""
        # Model regulation adaptation
        target = np.clip(needs, 0, 1)  # Normalize regulatory needs
        self.regulation_state += self.config.adaptation_rate * (target - self.regulation_state)
        
        # Update overall state
        idx = self.config.policy_dimensions + self.config.agency_count
        self.state[idx:idx + self.config.regulation_types] = self.regulation_state

class Technology(CaliforniaDomain):
    """Models California's technology sector including innovation and startups"""
    
    def __init__(self, config: CaliforniaDomainConfig):
        super().__init__("technology", config)
        
        # Initialize tech subsystems
        self.sector_state = np.zeros(config.tech_sectors)
        self.innovation_state = np.zeros(config.tech_sectors)
        self.startup_state = np.zeros(config.tech_sectors)
        
        # Set initial connection strengths
        self.connections.update({
            "government": 0.3,    # Regulatory influence
            "natural": 0.2,       # Environmental constraints
            "socioeconomic": 0.4  # Market demands
        })
        
    def update_sectors(self, market_demand: np.ndarray) -> None:
        """Update tech sector states based on market demand"""
        # Model sector growth
        growth = self.config.innovation_rate * (
            market_demand * (1 - self.sector_state)  # Logistic growth
        )
        self.sector_state += growth
        
        # Update overall state
        self.state[:self.config.tech_sectors] = self.sector_state
        
    def update_innovation(self) -> None:
        """Update innovation states across sectors"""
        # Model innovation dynamics
        self.innovation_state = (1 - self.config.adaptation_rate) * self.innovation_state + \
                              self.config.adaptation_rate * (
                                  self.sector_state + 0.1 * np.random.randn(self.config.tech_sectors)
                              )
        
        # Update overall state
        idx = self.config.tech_sectors
        self.state[idx:idx + self.config.tech_sectors] = self.innovation_state
        
    def update_startups(self, funding: np.ndarray) -> None:
        """Update startup activity based on funding availability"""
        # Model startup dynamics
        self.startup_state = self.config.startup_density * funding * self.innovation_state
        
        # Update overall state
        idx = 2 * self.config.tech_sectors
        self.state[idx:idx + self.config.tech_sectors] = self.startup_state

class Socioeconomic(CaliforniaDomain):
    """Models California's socioeconomic systems including demographics and economy"""
    
    def __init__(self, config: CaliforniaDomainConfig):
        super().__init__("socioeconomic", config)
        
        # Initialize socioeconomic subsystems
        self.population_state = np.zeros(config.population_groups)
        self.economic_state = np.zeros(config.economic_sectors)
        self.urban_state = np.zeros(config.urban_regions)
        
        # Set initial connection strengths
        self.connections.update({
            "government": 0.35,  # Policy influence
            "technology": 0.4,   # Tech sector influence
            "natural": 0.25     # Environmental influence
        })
        
    def update_population(self, factors: np.ndarray) -> None:
        """Update population states based on various factors"""
        # Model population dynamics
        growth = 0.01 * (factors + 0.05 * np.random.randn(self.config.population_groups))
        self.population_state *= (1 + growth)
        
        # Update overall state
        self.state[:self.config.population_groups] = self.population_state
        
    def update_economy(self, indicators: np.ndarray) -> None:
        """Update economic states based on indicators"""
        # Model economic dynamics
        self.economic_state += self.config.adaptation_rate * (
            indicators - self.economic_state
        )
        
        # Update overall state
        idx = self.config.population_groups
        self.state[idx:idx + self.config.economic_sectors] = self.economic_state
        
    def update_urban(self, development: np.ndarray) -> None:
        """Update urban region states based on development patterns"""
        # Model urban dynamics
        self.urban_state = np.clip(
            self.urban_state + self.config.adaptation_rate * development,
            0, 1
        )
        
        # Update overall state
        idx = self.config.population_groups + self.config.economic_sectors
        self.state[idx:idx + self.config.urban_regions] = self.urban_state

class CaliforniaSimulation:
    """Enhanced simulation class with advanced analytics"""
    
    def __init__(self, config: Optional[CaliforniaDomainConfig] = None):
        self.config = config or CaliforniaDomainConfig()
        
        # Initialize domains
        self.natural = NaturalSystem(self.config)
        self.government = Government(self.config)
        self.technology = Technology(self.config)
        self.socioeconomic = Socioeconomic(self.config)
        
        # Initialize quantum field configurations
        self.field_config = FieldMorphismConfig(
            field_dimension=self.config.field_dimension,
            coupling_strength=self.config.coupling_strength
        )
        
        # Initialize domain fields and analysis tools
        self._initialize_domains()
        self._initialize_analytics()
        
        # Tracking and analysis
        self.simulation_history = []
        self.intervention_history = []
        self.anomaly_history = []
        
    def _initialize_analytics(self) -> None:
        """Initialize analytics tools"""
        self.metrics = AdvancedMetrics()
        self.forecasting_models = {}
        self.optimization_objectives = {}
        self.risk_models = {}
        
        # Initialize forecasting models for each domain
        for domain in [self.natural, self.government, 
                      self.technology, self.socioeconomic]:
            self.forecasting_models[domain.name] = {
                'state': stats.LinearRegression(),
                'metrics': defaultdict(stats.LinearRegression)
            }
            
    def analyze_simulation(self) -> Dict[str, Any]:
        """Perform comprehensive simulation analysis"""
        analysis = {
            'system_stability': self._analyze_stability(),
            'cross_domain_impacts': self._analyze_cross_domain_impacts(),
            'emerging_patterns': self._analyze_emerging_patterns(),
            'risk_assessment': self._analyze_risks(),
            'optimization_opportunities': self._identify_optimization_opportunities()
        }
        
        # Add forecasts
        analysis['forecasts'] = self._generate_forecasts()
        
        # Add anomaly detection
        analysis['anomalies'] = self._detect_anomalies()
        
        return analysis
        
    def _analyze_stability(self) -> Dict[str, float]:
        """Analyze system stability across domains"""
        stability_metrics = {}
        
        for domain in [self.natural, self.government, 
                      self.technology, self.socioeconomic]:
            metrics = domain.compute_domain_metrics()
            stability_metrics[domain.name] = {
                'state_stability': metrics['stability'],
                'trend_strength': abs(metrics['trend']),
                'resilience': metrics['resilience']
            }
            
        # Compute cross-domain stability
        cross_domain = np.mean([
            metrics['stability'] 
            for metrics in stability_metrics.values()
        ])
        stability_metrics['cross_domain'] = float(cross_domain)
        
        return stability_metrics
        
    def _analyze_cross_domain_impacts(self) -> Dict[str, float]:
        """Analyze cross-domain impacts and interactions"""
        impacts = {}
        
        # Compute impact matrices
        for source in [self.natural, self.government, 
                      self.technology, self.socioeconomic]:
            impacts[source.name] = {}
            for target in [self.natural, self.government, 
                         self.technology, self.socioeconomic]:
                if source != target:
                    correlation = np.corrcoef(
                        source.state, target.state
                    )[0, 1]
                    impacts[source.name][target.name] = float(correlation)
                    
        return impacts
        
    def _analyze_emerging_patterns(self) -> Dict[str, Any]:
        """Analyze emerging patterns and trends"""
        patterns = {}
        
        # Analyze each domain
        for domain in [self.natural, self.government, 
                      self.technology, self.socioeconomic]:
            # Compute trend analysis
            trend = domain.forecast_state(horizon=self.config.forecast_horizon)
            
            # Detect cycles
            if len(domain.state_history) > 1:
                fft = np.fft.fft(
                    [np.mean(state) for state in domain.state_history]
                )
                frequencies = np.fft.fftfreq(len(fft))
                dominant_freq = frequencies[np.argmax(np.abs(fft[1:])) + 1]
                
                patterns[domain.name] = {
                    'trend': trend.tolist(),
                    'cycle_length': float(1 / dominant_freq) if dominant_freq != 0 else np.inf,
                    'trend_strength': float(np.std(trend))
                }
                
        return patterns
        
    def _analyze_risks(self) -> Dict[str, Any]:
        """Analyze system risks and vulnerabilities"""
        risks = {}
        
        # Natural system risks
        risks['natural'] = {
            'wildfire_risk': float(np.mean(self.natural.wildfire_risk)),
            'drought_risk': float(np.mean(self.natural.drought_severity)),
            'ecosystem_vulnerability': 1 - float(np.mean(self.natural.biodiversity_indices))
        }
        
        # Government risks
        risks['government'] = {
            'policy_failure_risk': 1 - float(np.mean(self.government.policy_state)),
            'regulatory_gap_risk': 1 - float(np.mean(self.government.regulation_state))
        }
        
        # Technology risks
        risks['technology'] = {
            'innovation_stagnation_risk': 1 - float(np.mean(self.technology.innovation_state)),
            'startup_failure_risk': 1 - float(np.mean(self.technology.startup_state))
        }
        
        # Socioeconomic risks
        risks['socioeconomic'] = {
            'inequality_risk': AdvancedMetrics.compute_inequality(self.socioeconomic.economic_state),
            'urban_stress_risk': float(np.mean(self.socioeconomic.urban_state))
        }
        
        return risks
        
    def _identify_optimization_opportunities(self) -> Dict[str, Any]:
        """Identify optimization opportunities"""
        opportunities = {}
        
        # Analyze each domain for optimization potential
        for domain in [self.natural, self.government, 
                      self.technology, self.socioeconomic]:
            metrics = domain.compute_domain_metrics()
            
            # Identify areas below target thresholds
            below_threshold = {
                key: value for key, value in metrics.items()
                if value < self.config.policy_effectiveness_threshold
            }
            
            if below_threshold:
                opportunities[domain.name] = {
                    'metrics_to_improve': list(below_threshold.keys()),
                    'current_values': list(below_threshold.values()),
                    'improvement_potential': {
                        key: self.config.policy_effectiveness_threshold - value
                        for key, value in below_threshold.items()
                    }
                }
                
        return opportunities
        
    def _generate_forecasts(self) -> Dict[str, Any]:
        """Generate forecasts for key metrics"""
        forecasts = {}
        
        # Generate domain-specific forecasts
        for domain in [self.natural, self.government, 
                      self.technology, self.socioeconomic]:
            # State forecast
            state_forecast = domain.forecast_state(
                horizon=self.config.forecast_horizon
            )
            
            # Metric forecasts
            metric_forecasts = {}
            for metric, values in domain.metrics_history:
                if len(values) > 1:
                    model = self.forecasting_models[domain.name]['metrics'][metric]
                    forecast = model.predict(
                        np.arange(self.config.forecast_horizon).reshape(-1, 1)
                    )
                    metric_forecasts[metric] = forecast.tolist()
                    
            forecasts[domain.name] = {
                'state': state_forecast.tolist(),
                'metrics': metric_forecasts
            }
            
        return forecasts
        
    def _detect_anomalies(self) -> Dict[str, Any]:
        """Detect anomalies in system behavior"""
        anomalies = {}
        
        # Check each domain for anomalies
        for domain in [self.natural, self.government, 
                      self.technology, self.socioeconomic]:
            domain_anomalies = []
            
            # Check state variations
            if len(domain.state_history) > 1:
                state_mean = np.mean([np.mean(state) for state in domain.state_history])
                state_std = np.std([np.mean(state) for state in domain.state_history])
                
                current_state_mean = np.mean(domain.state)
                if abs(current_state_mean - state_mean) > 2 * state_std:
                    domain_anomalies.append({
                        'type': 'state_deviation',
                        'severity': float(abs(current_state_mean - state_mean) / state_std),
                        'timestamp': len(domain.state_history)
                    })
                    
            # Check metric anomalies
            metrics = domain.compute_domain_metrics()
            for metric, value in metrics.items():
                if len(domain.metrics_history) > 1:
                    metric_values = [m[metric] for m in domain.metrics_history]
                    metric_mean = np.mean(metric_values)
                    metric_std = np.std(metric_values)
                    
                    if abs(value - metric_mean) > 2 * metric_std:
                        domain_anomalies.append({
                            'type': f'metric_anomaly_{metric}',
                            'severity': float(abs(value - metric_mean) / metric_std),
                            'timestamp': len(domain.metrics_history)
                        })
                        
            if domain_anomalies:
                anomalies[domain.name] = domain_anomalies
                
        return anomalies

def create_simulation() -> CaliforniaSimulation:
    """Create and initialize an enhanced California simulation"""
    return CaliforniaSimulation()

if __name__ == "__main__":
    # Create and run enhanced simulation
    simulation = create_simulation()
    results = simulation.run_simulation(steps=100)
    
    # Perform comprehensive analysis
    analysis = simulation.analyze_simulation()
    
    # Log results
    logger.info("Simulation completed. Final analysis:")
    logger.info(f"System Stability: {analysis['system_stability']}")
    logger.info(f"Cross-domain Impacts: {analysis['cross_domain_impacts']}")
    logger.info(f"Emerging Patterns: {analysis['emerging_patterns']}")
    logger.info(f"Risk Assessment: {analysis['risk_assessment']}")
    logger.info(f"Optimization Opportunities: {analysis['optimization_opportunities']}")
