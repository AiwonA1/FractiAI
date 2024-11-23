"""
FractiValidator.py

Implements comprehensive validation framework for FractiAI, enabling validation
across patterns, scales, and dimensions through fractal-based testing.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging
import time
from collections import deque
import pandas as pd
from scipy.stats import entropy
import networkx as nx

logger = logging.getLogger(__name__)

@dataclass
class ValidatorConfig:
    """Configuration for validation framework"""
    test_dimensions: List[str] = None  # Dimensions to test
    test_scales: List[int] = None  # Scales to test
    pattern_complexity: int = 3
    num_test_cases: int = 100
    timeout_seconds: int = 300
    parallel_tests: int = 4

    def __post_init__(self):
        if self.test_dimensions is None:
            self.test_dimensions = ['organic', 'inorganic', 'abstract']
        if self.test_scales is None:
            self.test_scales = [1, 2, 4, 8, 16]

class PatternValidator:
    """Validates fractal patterns"""
    
    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.pattern_history = []
        self.validation_history = []
        
    def validate_pattern(self, pattern: np.ndarray, 
                        expected: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Validate pattern properties and structure"""
        # Store pattern
        self.pattern_history.append(pattern)
        
        # Validate properties
        properties = self._validate_properties(pattern)
        
        # Validate against expected if provided
        if expected is not None:
            comparison = self._validate_against_expected(pattern, expected)
            properties.update(comparison)
            
        # Store validation
        self.validation_history.append({
            'pattern': pattern,
            'expected': expected,
            'properties': properties
        })
        
        return properties
        
    def _validate_properties(self, pattern: np.ndarray) -> Dict[str, float]:
        """Validate pattern properties"""
        return {
            'complexity': self._validate_complexity(pattern),
            'stability': self._validate_stability(pattern),
            'fractal_dimension': self._validate_fractal_dimension(pattern),
            'coherence': self._validate_coherence(pattern)
        }
        
    def _validate_complexity(self, pattern: np.ndarray) -> float:
        """Validate pattern complexity"""
        return float(entropy(pattern / np.sum(pattern)))
        
    def _validate_stability(self, pattern: np.ndarray) -> float:
        """Validate pattern stability"""
        if len(self.pattern_history) < 2:
            return 1.0
            
        similarities = []
        for hist_pattern in self.pattern_history[-10:]:
            similarity = np.corrcoef(pattern, hist_pattern)[0,1]
            similarities.append(abs(similarity))
            
        return float(np.mean(similarities))
        
    def _validate_fractal_dimension(self, pattern: np.ndarray) -> float:
        """Validate fractal dimension"""
        scales = np.logspace(0, 2, num=10, base=2, dtype=int)
        counts = []
        
        for scale in scales:
            scaled = self._rescale_pattern(pattern, scale)
            counts.append(len(np.unique(scaled)))
            
        if len(counts) > 1:
            coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
            return float(-coeffs[0])
        return 1.0
        
    def _validate_coherence(self, pattern: np.ndarray) -> float:
        """Validate pattern coherence"""
        if len(pattern) < 4:
            return 1.0
            
        # Check self-similarity at different scales
        coherence_scores = []
        for scale in [2, 4]:
            scaled = self._rescale_pattern(pattern, len(pattern) // scale)
            similarity = np.corrcoef(
                pattern[:len(scaled)], 
                np.repeat(scaled, scale)[:len(pattern)]
            )[0,1]
            coherence_scores.append(abs(similarity))
            
        return float(np.mean(coherence_scores))
        
    def _validate_against_expected(self, pattern: np.ndarray,
                                 expected: np.ndarray) -> Dict[str, float]:
        """Validate pattern against expected pattern"""
        if pattern.shape != expected.shape:
            pattern = self._rescale_pattern(pattern, len(expected))
            
        return {
            'accuracy': float(1.0 - np.mean(np.abs(pattern - expected))),
            'correlation': float(np.abs(np.corrcoef(pattern, expected)[0,1])),
            'mse': float(np.mean((pattern - expected) ** 2))
        }
        
    def _rescale_pattern(self, pattern: np.ndarray, target_size: int) -> np.ndarray:
        """Rescale pattern to target size"""
        if len(pattern) < target_size:
            return np.interp(
                np.linspace(0, 1, target_size),
                np.linspace(0, 1, len(pattern)),
                pattern
            )
        elif len(pattern) > target_size:
            return np.array([
                np.mean(chunk) 
                for chunk in np.array_split(pattern, target_size)
            ])
        return pattern

class ComponentValidator:
    """Validates system components"""
    
    def __init__(self, component_id: str, config: ValidatorConfig):
        self.component_id = component_id
        self.config = config
        self.test_history = []
        
    def validate_component(self, component: Any, 
                         test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate component functionality"""
        results = []
        
        for test_case in test_cases:
            # Run test case
            result = self._run_test_case(component, test_case)
            results.append(result)
            
        # Store validation
        validation = self._analyze_results(results)
        self.test_history.append({
            'timestamp': time.time(),
            'results': results,
            'validation': validation
        })
        
        return validation
        
    def _run_test_case(self, component: Any, 
                       test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run single test case"""
        start_time = time.time()
        
        try:
            # Get test parameters
            input_data = test_case['input']
            expected = test_case.get('expected')
            
            # Run test
            output = component.process(input_data)
            
            # Validate output
            validation = self._validate_output(output, expected)
            
            return {
                'success': True,
                'duration': time.time() - start_time,
                'validation': validation
            }
            
        except Exception as e:
            return {
                'success': False,
                'duration': time.time() - start_time,
                'error': str(e)
            }
            
    def _validate_output(self, output: Any, 
                        expected: Optional[Any]) -> Dict[str, float]:
        """Validate component output"""
        if expected is None:
            return {}
            
        if isinstance(output, np.ndarray) and isinstance(expected, np.ndarray):
            return {
                'accuracy': float(1.0 - np.mean(np.abs(output - expected))),
                'correlation': float(np.abs(np.corrcoef(output.flatten(), 
                                                      expected.flatten())[0,1])),
                'mse': float(np.mean((output - expected) ** 2))
            }
            
        return {
            'match': float(output == expected)
        }
        
    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze test results"""
        successes = [r['success'] for r in results]
        durations = [r['duration'] for r in results]
        
        # Compute metrics
        metrics = {
            'success_rate': float(np.mean(successes)),
            'mean_duration': float(np.mean(durations)),
            'total_duration': float(np.sum(durations))
        }
        
        # Get validation metrics
        validations = [r['validation'] for r in results if r['success']]
        if validations:
            metrics['validation_metrics'] = self._aggregate_validations(validations)
            
        return metrics
        
    def _aggregate_validations(self, 
                             validations: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate validation metrics"""
        aggregated = {}
        
        for metric in validations[0].keys():
            values = [v[metric] for v in validations]
            aggregated[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
            
        return aggregated

class FractiValidator:
    """Main validation system with fractal awareness"""
    
    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.pattern_validator = PatternValidator(config)
        self.component_validators: Dict[str, ComponentValidator] = {}
        self.validation_history = []
        
    def add_component(self, component_id: str) -> None:
        """Add component to validation system"""
        self.component_validators[component_id] = ComponentValidator(
            component_id,
            self.config
        )
        
    def validate_system(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate system state"""
        # Validate patterns
        patterns = self._extract_patterns(state)
        pattern_results = {
            pattern_id: self.pattern_validator.validate_pattern(pattern)
            for pattern_id, pattern in patterns.items()
        }
        
        # Validate components
        component_results = {}
        for component_id, data in state.items():
            if component_id in self.component_validators:
                test_cases = self._generate_test_cases(data)
                result = self.component_validators[component_id].validate_component(
                    data,
                    test_cases
                )
                component_results[component_id] = result
                
        # Store validation
        validation = {
            'patterns': pattern_results,
            'components': component_results,
            'system': self._validate_system_properties(pattern_results, 
                                                     component_results)
        }
        self.validation_history.append({
            'timestamp': time.time(),
            'validation': validation
        })
        
        return validation
        
    def _extract_patterns(self, state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract patterns from state"""
        patterns = {}
        
        def extract_recursive(data: Any, path: str = '') -> None:
            if isinstance(data, np.ndarray):
                patterns[path] = data
            elif isinstance(data, dict):
                for key, value in data.items():
                    new_path = f"{path}_{key}" if path else key
                    extract_recursive(value, new_path)
                    
        extract_recursive(state)
        return patterns
        
    def _generate_test_cases(self, component: Any) -> List[Dict[str, Any]]:
        """Generate test cases for component"""
        test_cases = []
        
        for _ in range(self.config.num_test_cases):
            # Generate test input
            input_data = self._generate_test_input(component)
            
            # Generate expected output if possible
            expected = self._generate_expected_output(component, input_data)
            
            test_cases.append({
                'input': input_data,
                'expected': expected
            })
            
        return test_cases
        
    def _generate_test_input(self, component: Any) -> np.ndarray:
        """Generate test input data"""
        # Implementation depends on component type
        return np.random.randn(10)
        
    def _generate_expected_output(self, component: Any, 
                                input_data: np.ndarray) -> Optional[np.ndarray]:
        """Generate expected output for test input"""
        # Implementation depends on component type
        return None
        
    def _validate_system_properties(self,
                                  pattern_results: Dict[str, Dict[str, float]],
                                  component_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Validate system-wide properties"""
        # Compute pattern-based metrics
        pattern_metrics = []
        for results in pattern_results.values():
            pattern_metrics.append({
                'complexity': results['complexity'],
                'stability': results['stability'],
                'coherence': results['coherence']
            })
            
        # Compute component-based metrics
        component_metrics = []
        for results in component_results.values():
            if 'success_rate' in results:
                component_metrics.append({
                    'reliability': results['success_rate'],
                    'performance': 1.0 / (results['mean_duration'] + 1e-7)
                })
                
        return {
            'pattern_quality': float(np.mean([m['coherence'] * m['stability'] 
                                            for m in pattern_metrics])),
            'system_reliability': float(np.mean([m['reliability'] 
                                               for m in component_metrics])),
            'system_performance': float(np.mean([m['performance'] 
                                               for m in component_metrics]))
        } 