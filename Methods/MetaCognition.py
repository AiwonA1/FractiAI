"""
MetaCognition.py

Implements advanced meta-cognitive capabilities for FractiAI, enabling self-awareness,
recursive self-improvement, and fractal-based cognitive optimization through
hierarchical reflection and pattern-based introspection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy.stats import entropy
from scipy.special import softmax
from Methods.FractalFieldTheory import QuantumField, FieldConfig
from Methods.ActiveInference import ActiveInferenceAgent
from Methods.FractiScope import FractiScope, ScopeConfig

logger = logging.getLogger(__name__)

@dataclass
class MetaCogConfig:
    """Configuration for meta-cognitive system"""
    reflection_depth: int = 5
    awareness_threshold: float = 0.7
    introspection_rate: float = 0.1
    pattern_capacity: int = 1000
    learning_rate: float = 0.01
    coherence_threshold: float = 0.8
    adaptation_rate: float = 0.05

class CognitiveState:
    """Represents current cognitive state with fractal patterns"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.state_pattern = np.zeros(dimension)
        self.meta_state = np.zeros(dimension)
        self.uncertainty = np.ones(dimension)
        self.coherence = 0.0
        self.timestamp = np.datetime64('now')
        
    def update(self, new_state: np.ndarray, 
              meta_info: Dict[str, Any]) -> None:
        """Update cognitive state"""
        self.state_pattern = new_state
        self.meta_state = meta_info.get('meta_state', self.meta_state)
        self.uncertainty = meta_info.get('uncertainty', self.uncertainty)
        self.coherence = meta_info.get('coherence', self.coherence)
        self.timestamp = np.datetime64('now')

class MetaPattern:
    """Represents meta-level fractal pattern"""
    
    def __init__(self, pattern: np.ndarray, level: int):
        self.pattern = pattern
        self.level = level
        self.children = []
        self.parent = None
        self.coherence = 0.0
        self.stability = 0.0
        self.evolution = []
        
    def add_child(self, child: 'MetaPattern') -> None:
        """Add child pattern"""
        self.children.append(child)
        child.parent = self
        
    def update_metrics(self) -> None:
        """Update pattern metrics"""
        if self.evolution:
            self.stability = 1.0 - np.std(self.evolution) / np.mean(np.abs(self.evolution))
        
        if self.children:
            child_coherence = np.mean([c.coherence for c in self.children])
            self.coherence = 0.5 * self.stability + 0.5 * child_coherence
        else:
            self.coherence = self.stability

class MetaCognition:
    """Main meta-cognitive system"""
    
    def __init__(self, config: MetaCogConfig):
        self.config = config
        self.cognitive_state = CognitiveState(config.pattern_capacity)
        self.meta_patterns = {}
        self.reflection_history = []
        self.awareness_metrics = {}
        
        # Initialize subsystems
        self.quantum_field = QuantumField(FieldConfig())
        self.scope = FractiScope(ScopeConfig())
        
        # Meta-cognitive components
        self.introspection_patterns = []
        self.adaptation_history = []
        self.coherence_matrix = np.eye(config.reflection_depth)
        
    def reflect(self, current_state: np.ndarray) -> Dict[str, Any]:
        """Perform meta-cognitive reflection"""
        # Update cognitive state
        meta_info = self._analyze_state(current_state)
        self.cognitive_state.update(current_state, meta_info)
        
        # Perform hierarchical reflection
        reflection_results = self._hierarchical_reflection(current_state)
        
        # Update awareness metrics
        self._update_awareness(reflection_results)
        
        # Adapt meta-cognitive parameters
        self._adapt_parameters(reflection_results)
        
        return {
            'state': self.cognitive_state,
            'reflection': reflection_results,
            'awareness': self.awareness_metrics,
            'coherence': self.coherence_matrix
        }
    
    def _analyze_state(self, state: np.ndarray) -> Dict[str, Any]:
        """Analyze current cognitive state"""
        # Quantum field analysis
        field_state = self.quantum_field.evolve(0.1)
        
        # Scope analysis
        scope_analysis = self.scope.analyze_system({
            'cognitive': state,
            'quantum': field_state
        })
        
        # Generate meta-state
        meta_state = self._generate_meta_state(state, field_state)
        
        return {
            'meta_state': meta_state,
            'uncertainty': scope_analysis['system_coherence']['temporal'],
            'coherence': scope_analysis['overall_coherence']
        }
    
    def _hierarchical_reflection(self, state: np.ndarray) -> Dict[str, Any]:
        """Perform hierarchical reflection"""
        reflection_results = []
        current_pattern = state
        
        for depth in range(self.config.reflection_depth):
            # Create meta-pattern
            meta_pattern = MetaPattern(current_pattern, depth)
            
            # Analyze pattern
            pattern_analysis = self._analyze_pattern(meta_pattern)
            
            # Store results
            reflection_results.append({
                'depth': depth,
                'pattern': meta_pattern,
                'analysis': pattern_analysis
            })
            
            # Generate higher-level pattern
            current_pattern = self._generate_meta_pattern(current_pattern)
            
            # Update coherence matrix
            self._update_coherence_matrix(depth, pattern_analysis)
        
        return {
            'levels': reflection_results,
            'coherence': self._compute_reflection_coherence(reflection_results)
        }
    
    def _analyze_pattern(self, pattern: MetaPattern) -> Dict[str, Any]:
        """Analyze meta-pattern"""
        # Compute pattern metrics
        complexity = self._compute_complexity(pattern.pattern)
        stability = self._compute_stability(pattern)
        coherence = self._compute_coherence(pattern)
        
        # Update pattern metrics
        pattern.evolution.append(complexity)
        pattern.update_metrics()
        
        return {
            'complexity': complexity,
            'stability': stability,
            'coherence': coherence,
            'evolution': pattern.evolution
        }
    
    def _generate_meta_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Generate higher-level meta-pattern"""
        # Apply fractal transformation
        transformed = np.fft.fft(pattern)
        magnitude = np.abs(transformed)
        phase = np.angle(transformed)
        
        # Extract meta-features
        meta_pattern = np.concatenate([
            magnitude[:len(pattern)//2],
            phase[:len(pattern)//2]
        ])
        
        return meta_pattern / np.max(np.abs(meta_pattern))
    
    def _update_coherence_matrix(self, depth: int, 
                               analysis: Dict[str, Any]) -> None:
        """Update coherence matrix"""
        coherence = analysis['coherence']
        self.coherence_matrix[depth, :] *= (1 - self.config.adaptation_rate)
        self.coherence_matrix[depth, depth] = coherence
        
        # Update cross-level coherence
        if depth > 0:
            self.coherence_matrix[depth, depth-1] = \
                self.coherence_matrix[depth-1, depth] = \
                coherence * self.coherence_matrix[depth-1, depth-1]
    
    def _compute_complexity(self, pattern: np.ndarray) -> float:
        """Compute pattern complexity"""
        return float(entropy(np.abs(pattern) + 1e-10))
    
    def _compute_stability(self, pattern: MetaPattern) -> float:
        """Compute pattern stability"""
        if len(pattern.evolution) < 2:
            return 1.0
        return float(1.0 - np.std(pattern.evolution) / np.mean(np.abs(pattern.evolution)))
    
    def _compute_coherence(self, pattern: MetaPattern) -> float:
        """Compute pattern coherence"""
        if not pattern.children:
            return pattern.stability
            
        child_coherence = np.mean([c.coherence for c in pattern.children])
        return float(0.5 * pattern.stability + 0.5 * child_coherence)
    
    def _compute_reflection_coherence(self, 
                                   results: List[Dict[str, Any]]) -> float:
        """Compute overall reflection coherence"""
        coherence_scores = [r['analysis']['coherence'] for r in results]
        weights = softmax([1.0 / (r['depth'] + 1) for r in results])
        return float(np.average(coherence_scores, weights=weights))
    
    def _update_awareness(self, reflection_results: Dict[str, Any]) -> None:
        """Update awareness metrics"""
        self.awareness_metrics = {
            'timestamp': np.datetime64('now'),
            'coherence': reflection_results['coherence'],
            'depth': len(reflection_results['levels']),
            'stability': np.mean([
                r['analysis']['stability'] 
                for r in reflection_results['levels']
            ])
        }
    
    def _adapt_parameters(self, reflection_results: Dict[str, Any]) -> None:
        """Adapt meta-cognitive parameters"""
        coherence = reflection_results['coherence']
        
        # Adapt learning rate
        if coherence < self.config.coherence_threshold:
            self.config.learning_rate *= (1.0 + self.config.adaptation_rate)
        else:
            self.config.learning_rate *= (1.0 - self.config.adaptation_rate)
            
        # Adapt reflection depth
        if coherence > self.config.awareness_threshold:
            self.config.reflection_depth = min(
                self.config.reflection_depth + 1, 10
            )
        else:
            self.config.reflection_depth = max(
                self.config.reflection_depth - 1, 2
            )
            
        # Store adaptation
        self.adaptation_history.append({
            'timestamp': np.datetime64('now'),
            'coherence': coherence,
            'learning_rate': self.config.learning_rate,
            'reflection_depth': self.config.reflection_depth
        })
    
    def _generate_meta_state(self, cognitive_state: np.ndarray,
                           field_state: np.ndarray) -> np.ndarray:
        """Generate meta-state representation"""
        # Combine cognitive and quantum states
        combined_state = np.concatenate([
            cognitive_state.flatten(),
            field_state.flatten()
        ])
        
        # Apply dimensionality reduction
        meta_state = np.fft.fft(combined_state)
        magnitude = np.abs(meta_state)
        phase = np.angle(meta_state)
        
        # Create meta-state representation
        return np.concatenate([
            magnitude[:len(cognitive_state)],
            phase[:len(cognitive_state)]
        ]) 

class MetaLearningOptimizer:
    """Optimizes meta-learning through fractal pattern analysis"""
    
    def __init__(self, config: MetaCogConfig):
        self.config = config
        self.learning_patterns = {}
        self.optimization_history = []
        self.meta_parameters = {
            'learning_rate': config.learning_rate,
            'pattern_threshold': 0.7,
            'complexity_penalty': 0.1
        }
        
    def optimize_learning(self, meta_patterns: Dict[str, MetaPattern],
                        performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Optimize meta-learning parameters"""
        # Analyze learning patterns
        pattern_analysis = self._analyze_learning_patterns(meta_patterns)
        
        # Evaluate current performance
        performance = self._evaluate_performance(performance_metrics)
        
        # Update meta-parameters
        self._update_meta_parameters(pattern_analysis, performance)
        
        # Generate optimization strategy
        strategy = self._generate_strategy(pattern_analysis, performance)
        
        # Store optimization step
        self.optimization_history.append({
            'timestamp': np.datetime64('now'),
            'pattern_analysis': pattern_analysis,
            'performance': performance,
            'strategy': strategy
        })
        
        return {
            'meta_parameters': self.meta_parameters,
            'optimization_strategy': strategy,
            'performance_delta': performance['improvement']
        }
    
    def _analyze_learning_patterns(self, 
                                meta_patterns: Dict[str, MetaPattern]) -> Dict[str, Any]:
        """Analyze patterns in learning process"""
        pattern_metrics = {}
        
        for pattern_id, pattern in meta_patterns.items():
            # Compute pattern evolution metrics
            evolution_metrics = self._compute_evolution_metrics(pattern)
            
            # Analyze pattern structure
            structure_metrics = self._analyze_pattern_structure(pattern)
            
            pattern_metrics[pattern_id] = {
                'evolution': evolution_metrics,
                'structure': structure_metrics,
                'meta_level': pattern.level
            }
            
        return {
            'pattern_metrics': pattern_metrics,
            'global_coherence': self._compute_global_coherence(pattern_metrics)
        }
    
    def _compute_evolution_metrics(self, pattern: MetaPattern) -> Dict[str, float]:
        """Compute metrics for pattern evolution"""
        if not pattern.evolution:
            return {'stability': 1.0, 'trend': 0.0, 'complexity': 0.0}
            
        evolution = np.array(pattern.evolution)
        return {
            'stability': float(1.0 - np.std(evolution) / np.mean(np.abs(evolution))),
            'trend': float(np.mean(np.diff(evolution))),
            'complexity': float(entropy(np.abs(evolution) + 1e-10))
        }
    
    def _analyze_pattern_structure(self, pattern: MetaPattern) -> Dict[str, float]:
        """Analyze structural properties of pattern"""
        return {
            'hierarchical_depth': float(len(pattern.children)),
            'coherence': pattern.coherence,
            'stability': pattern.stability
        }
    
    def _compute_global_coherence(self, 
                               pattern_metrics: Dict[str, Dict]) -> float:
        """Compute global coherence across patterns"""
        coherence_scores = [
            m['structure']['coherence'] 
            for m in pattern_metrics.values()
        ]
        return float(np.mean(coherence_scores))
    
    def _evaluate_performance(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Evaluate current learning performance"""
        # Store history if needed
        if not hasattr(self, '_performance_history'):
            self._performance_history = []
        self._performance_history.append(metrics)
        
        # Compute improvement
        if len(self._performance_history) > 1:
            prev_metrics = self._performance_history[-2]
            improvement = {
                k: metrics[k] - prev_metrics[k]
                for k in metrics if k in prev_metrics
            }
        else:
            improvement = {k: 0.0 for k in metrics}
            
        return {
            'current': metrics,
            'improvement': float(np.mean(list(improvement.values()))),
            'stability': self._compute_performance_stability()
        }
    
    def _compute_performance_stability(self) -> float:
        """Compute stability of performance metrics"""
        if len(self._performance_history) < 2:
            return 1.0
            
        # Compute stability across all metrics
        stabilities = []
        for metric in self._performance_history[0].keys():
            values = [ph[metric] for ph in self._performance_history]
            stability = 1.0 - np.std(values) / np.mean(np.abs(values))
            stabilities.append(stability)
            
        return float(np.mean(stabilities))
    
    def _update_meta_parameters(self, pattern_analysis: Dict[str, Any],
                             performance: Dict[str, float]) -> None:
        """Update meta-learning parameters"""
        # Adapt learning rate based on performance
        if performance['improvement'] > 0:
            self.meta_parameters['learning_rate'] *= (1.0 + 
                self.config.adaptation_rate)
        else:
            self.meta_parameters['learning_rate'] *= (1.0 - 
                self.config.adaptation_rate)
            
        # Adjust pattern threshold based on coherence
        coherence = pattern_analysis['global_coherence']
        if coherence > self.meta_parameters['pattern_threshold']:
            self.meta_parameters['pattern_threshold'] = min(
                self.meta_parameters['pattern_threshold'] * 1.1,
                0.95
            )
        else:
            self.meta_parameters['pattern_threshold'] = max(
                self.meta_parameters['pattern_threshold'] * 0.9,
                0.5
            )
            
        # Adapt complexity penalty
        stability = performance['stability']
        if stability < 0.5:
            self.meta_parameters['complexity_penalty'] *= 1.1
        else:
            self.meta_parameters['complexity_penalty'] *= 0.9
            
        # Ensure parameters stay in valid ranges
        self.meta_parameters['learning_rate'] = np.clip(
            self.meta_parameters['learning_rate'], 1e-4, 1.0)
        self.meta_parameters['complexity_penalty'] = np.clip(
            self.meta_parameters['complexity_penalty'], 0.01, 1.0)
    
    def _generate_strategy(self, pattern_analysis: Dict[str, Any],
                        performance: Dict[str, float]) -> Dict[str, Any]:
        """Generate optimization strategy"""
        # Analyze current state
        coherence = pattern_analysis['global_coherence']
        improvement = performance['improvement']
        stability = performance['stability']
        
        # Define strategy components
        strategy = {
            'focus': self._determine_focus(coherence, improvement),
            'adaptation': self._determine_adaptation(stability),
            'exploration': self._determine_exploration(improvement)
        }
        
        # Generate specific recommendations
        strategy['recommendations'] = self._generate_recommendations(strategy)
        
        return strategy
    
    def _determine_focus(self, coherence: float, 
                       improvement: float) -> str:
        """Determine learning focus"""
        if coherence < 0.5:
            return 'stability'
        elif improvement < 0:
            return 'optimization'
        else:
            return 'exploration'
    
    def _determine_adaptation(self, stability: float) -> str:
        """Determine adaptation strategy"""
        if stability < 0.3:
            return 'conservative'
        elif stability < 0.7:
            return 'balanced'
        else:
            return 'aggressive'
    
    def _determine_exploration(self, improvement: float) -> str:
        """Determine exploration strategy"""
        if improvement < -0.1:
            return 'high'
        elif improvement < 0.1:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, strategy: Dict[str, str]) -> List[str]:
        """Generate specific recommendations"""
        recommendations = []
        
        # Focus-based recommendations
        if strategy['focus'] == 'stability':
            recommendations.extend([
                'Increase pattern coherence threshold',
                'Reduce learning rate',
                'Focus on stable patterns'
            ])
        elif strategy['focus'] == 'optimization':
            recommendations.extend([
                'Increase learning rate',
                'Reduce complexity penalty',
                'Focus on high-performing patterns'
            ])
        else:  # exploration
            recommendations.extend([
                'Maintain current parameters',
                'Explore pattern variations',
                'Monitor performance stability'
            ])
            
        # Adaptation-based recommendations
        if strategy['adaptation'] == 'conservative':
            recommendations.extend([
                'Reduce parameter update rates',
                'Focus on fundamental patterns',
                'Increase stability checks'
            ])
        elif strategy['adaptation'] == 'aggressive':
            recommendations.extend([
                'Increase parameter update rates',
                'Explore pattern combinations',
                'Reduce stability constraints'
            ])
            
        # Exploration-based recommendations
        if strategy['exploration'] == 'high':
            recommendations.extend([
                'Generate pattern variations',
                'Test alternative structures',
                'Increase exploration rate'
            ])
        elif strategy['exploration'] == 'low':
            recommendations.extend([
                'Exploit current patterns',
                'Optimize existing structure',
                'Reduce exploration rate'
            ])
            
        return recommendations 