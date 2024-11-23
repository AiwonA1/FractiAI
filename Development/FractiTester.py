"""
FractiTester.py

Implements comprehensive testing framework for FractiAI, enabling automated testing
across patterns, scales, and dimensions through fractal-based test generation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging
import time
from collections import deque
import pytest
from hypothesis import given, strategies as st
import networkx as nx

logger = logging.getLogger(__name__)

@dataclass
class TestConfig:
    """Configuration for testing framework"""
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

class PatternTestGenerator:
    """Generates test patterns with fractal properties"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.pattern_history = []
        
    def generate_pattern(self, scale: int) -> np.ndarray:
        """Generate test pattern at given scale"""
        # Generate base pattern
        pattern = self._generate_base_pattern(scale)
        
        # Add fractal components
        for i in range(self.config.pattern_complexity):
            fractal = self._generate_fractal_component(scale, i)
            pattern += fractal
            
        # Normalize pattern
        pattern = pattern / np.max(np.abs(pattern))
        
        # Store pattern
        self.pattern_history.append(pattern)
        
        return pattern
        
    def _generate_base_pattern(self, scale: int) -> np.ndarray:
        """Generate base pattern"""
        return np.sin(np.linspace(0, 2*np.pi, scale))
        
    def _generate_fractal_component(self, scale: int, level: int) -> np.ndarray:
        """Generate fractal component"""
        freq = 2 ** level
        return 0.5 ** level * np.sin(freq * np.linspace(0, 2*np.pi, scale))
        
    @given(st.integers(min_value=8, max_value=128))
    def test_pattern_properties(self, scale: int) -> None:
        """Property-based tests for generated patterns"""
        pattern = self.generate_pattern(scale)
        
        # Test pattern properties
        assert len(pattern) == scale
        assert np.all(np.abs(pattern) <= 1.0)
        assert np.abs(np.mean(pattern)) < 0.5  # Approximately centered

class ComponentTestGenerator:
    """Generates test cases for system components"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.test_history = []
        
    def generate_test_suite(self, component_type: str) -> List[Dict[str, Any]]:
        """Generate test suite for component"""
        test_cases = []
        
        for scale in self.config.test_scales:
            for _ in range(self.config.num_test_cases):
                test_case = self._generate_test_case(component_type, scale)
                test_cases.append(test_case)
                
        # Store test suite
        self.test_history.append({
            'component_type': component_type,
            'test_cases': test_cases,
            'timestamp': time.time()
        })
        
        return test_cases
        
    def _generate_test_case(self, component_type: str, 
                           scale: int) -> Dict[str, Any]:
        """Generate single test case"""
        if component_type == 'pattern':
            return self._generate_pattern_test(scale)
        elif component_type == 'network':
            return self._generate_network_test(scale)
        else:
            return self._generate_default_test(scale)
            
    def _generate_pattern_test(self, scale: int) -> Dict[str, Any]:
        """Generate pattern-based test"""
        pattern_gen = PatternTestGenerator(self.config)
        input_pattern = pattern_gen.generate_pattern(scale)
        expected = np.roll(input_pattern, 1)  # Simple transformation
        
        return {
            'input': input_pattern,
            'expected': expected,
            'scale': scale,
            'properties': {
                'symmetry': self._check_symmetry(input_pattern),
                'complexity': self._compute_complexity(input_pattern)
            }
        }
        
    def _generate_network_test(self, scale: int) -> Dict[str, Any]:
        """Generate network-based test"""
        # Generate random graph
        G = nx.random_geometric_graph(scale, 0.3)
        adjacency = nx.adjacency_matrix(G).todense()
        
        return {
            'input': adjacency,
            'expected': None,  # Network tests often don't have expected output
            'scale': scale,
            'properties': {
                'density': nx.density(G),
                'diameter': nx.diameter(G) if nx.is_connected(G) else -1
            }
        }
        
    def _generate_default_test(self, scale: int) -> Dict[str, Any]:
        """Generate default test case"""
        return {
            'input': np.random.randn(scale),
            'expected': None,
            'scale': scale,
            'properties': {}
        }
        
    def _check_symmetry(self, pattern: np.ndarray) -> float:
        """Check pattern symmetry"""
        return float(np.corrcoef(pattern, pattern[::-1])[0,1])
        
    def _compute_complexity(self, pattern: np.ndarray) -> float:
        """Compute pattern complexity"""
        return float(np.std(pattern))

class TestExecutor:
    """Executes and validates tests"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results_history = []
        
    def execute_test_suite(self, component: Any, 
                          test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute test suite on component"""
        results = []
        start_time = time.time()
        
        for test_case in test_cases:
            if time.time() - start_time > self.config.timeout_seconds:
                logger.warning("Test suite timeout reached")
                break
                
            result = self._execute_test(component, test_case)
            results.append(result)
            
        # Store results
        execution_results = self._analyze_results(results)
        self.results_history.append({
            'timestamp': time.time(),
            'results': results,
            'analysis': execution_results
        })
        
        return execution_results
        
    def _execute_test(self, component: Any, 
                     test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Execute single test case"""
        try:
            # Run test
            start_time = time.time()
            output = component.process(test_case['input'])
            duration = time.time() - start_time
            
            # Validate output
            validation = self._validate_output(
                output,
                test_case.get('expected'),
                test_case['properties']
            )
            
            return {
                'success': True,
                'output': output,
                'duration': duration,
                'validation': validation
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'duration': time.time() - start_time
            }
            
    def _validate_output(self, output: Any, expected: Optional[Any],
                        properties: Dict[str, float]) -> Dict[str, float]:
        """Validate test output"""
        validation = {}
        
        # Compare with expected output if available
        if expected is not None:
            if isinstance(output, np.ndarray) and isinstance(expected, np.ndarray):
                validation.update({
                    'accuracy': float(1.0 - np.mean(np.abs(output - expected))),
                    'correlation': float(np.abs(np.corrcoef(output.flatten(),
                                                          expected.flatten())[0,1]))
                })
                
        # Validate properties
        for prop, value in properties.items():
            if prop == 'symmetry':
                validation[f'{prop}_preserved'] = float(
                    abs(self._check_symmetry(output) - value)
                )
            elif prop == 'complexity':
                validation[f'{prop}_preserved'] = float(
                    abs(self._compute_complexity(output) - value)
                )
                
        return validation
        
    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze test results"""
        successes = [r['success'] for r in results]
        durations = [r['duration'] for r in results]
        
        analysis = {
            'success_rate': float(np.mean(successes)),
            'mean_duration': float(np.mean(durations)),
            'total_duration': float(np.sum(durations))
        }
        
        # Analyze validations
        validations = [r['validation'] for r in results if r['success']]
        if validations:
            analysis['validation_metrics'] = self._aggregate_validations(validations)
            
        return analysis
        
    def _aggregate_validations(self, 
                             validations: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
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
        
    def _check_symmetry(self, pattern: np.ndarray) -> float:
        """Check pattern symmetry"""
        return float(np.corrcoef(pattern, pattern[::-1])[0,1])
        
    def _compute_complexity(self, pattern: np.ndarray) -> float:
        """Compute pattern complexity"""
        return float(np.std(pattern))

class FractiTester:
    """Main testing system with fractal awareness"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.pattern_generator = PatternTestGenerator(config)
        self.component_generator = ComponentTestGenerator(config)
        self.executor = TestExecutor(config)
        
    def test_component(self, component: Any, 
                      component_type: str) -> Dict[str, Any]:
        """Test component comprehensively"""
        # Generate test suite
        test_cases = self.component_generator.generate_test_suite(component_type)
        
        # Execute tests
        results = self.executor.execute_test_suite(component, test_cases)
        
        return {
            'results': results,
            'test_cases': test_cases,
            'recommendations': self._generate_recommendations(results)
        }
        
    def _generate_recommendations(self, 
                                results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate testing recommendations"""
        recommendations = []
        
        # Check success rate
        if results['success_rate'] < 0.95:
            recommendations.append({
                'type': 'reliability',
                'description': "Low test success rate",
                'suggestion': "Review component stability"
            })
            
        # Check performance
        if results['mean_duration'] > 1.0:  # 1 second threshold
            recommendations.append({
                'type': 'performance',
                'description': "High test execution time",
                'suggestion': "Optimize component performance"
            })
            
        # Check validation metrics
        if 'validation_metrics' in results:
            for metric, stats in results['validation_metrics'].items():
                if stats['mean'] < 0.8:  # 80% threshold
                    recommendations.append({
                        'type': 'validation',
                        'description': f"Low {metric} score",
                        'suggestion': f"Improve {metric} preservation"
                    })
                    
        return recommendations 