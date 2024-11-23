"""
FractiTest.py

Implements comprehensive testing framework for FractiAI, enabling testing across
dimensions, scales, and patterns through fractal-based validation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging
import time
from collections import deque
import json

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

class TestCase:
    """Individual test case with fractal validation"""
    
    def __init__(self, test_id: str, input_data: Any, expected_output: Any):
        self.test_id = test_id
        self.input_data = input_data
        self.expected_output = expected_output
        self.results = {}
        self.metrics = {}
        self.start_time = None
        self.end_time = None
        
    def run(self, test_fn: Callable) -> bool:
        """Run test case"""
        self.start_time = time.time()
        
        try:
            # Execute test
            actual_output = test_fn(self.input_data)
            
            # Validate output
            success = self._validate_output(actual_output)
            
            # Store results
            self.results = {
                'actual_output': actual_output,
                'success': success,
                'error': None
            }
            
        except Exception as e:
            self.results = {
                'actual_output': None,
                'success': False,
                'error': str(e)
            }
            
        self.end_time = time.time()
        return self.results['success']
        
    def _validate_output(self, actual_output: Any) -> bool:
        """Validate test output"""
        if isinstance(self.expected_output, np.ndarray):
            return np.allclose(actual_output, self.expected_output)
        return actual_output == self.expected_output
        
    def get_metrics(self) -> Dict[str, float]:
        """Get test metrics"""
        if not self.start_time or not self.end_time:
            return {}
            
        return {
            'duration': self.end_time - self.start_time,
            'success': float(self.results.get('success', False)),
            'error_rate': float(self.results.get('error') is not None)
        }

class TestSuite:
    """Collection of related test cases"""
    
    def __init__(self, suite_id: str, dimension: str):
        self.suite_id = suite_id
        self.dimension = dimension
        self.test_cases: Dict[str, TestCase] = {}
        self.results_history = []
        
    def add_test(self, test_case: TestCase) -> None:
        """Add test case to suite"""
        self.test_cases[test_case.test_id] = test_case
        
    def run_suite(self, test_fn: Callable) -> Dict[str, Any]:
        """Run all test cases in suite"""
        results = {}
        
        for test_id, test_case in self.test_cases.items():
            success = test_case.run(test_fn)
            results[test_id] = {
                'success': success,
                'metrics': test_case.get_metrics()
            }
            
        # Store results
        self.results_history.append({
            'timestamp': time.time(),
            'results': results
        })
        
        return {
            'suite_id': self.suite_id,
            'dimension': self.dimension,
            'results': results,
            'metrics': self._compute_suite_metrics(results)
        }
        
    def _compute_suite_metrics(self, 
                             results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Compute suite-level metrics"""
        successes = [r['success'] for r in results.values()]
        durations = [r['metrics']['duration'] for r in results.values()]
        
        return {
            'success_rate': float(np.mean(successes)),
            'mean_duration': float(np.mean(durations)),
            'total_duration': float(np.sum(durations)),
            'error_rate': float(1.0 - np.mean(successes))
        }

class FractiTest:
    """Main testing framework with fractal validation"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.test_suites: Dict[str, TestSuite] = {}
        self.test_history = []
        self.initialize_suites()
        
    def initialize_suites(self) -> None:
        """Initialize test suites for each dimension"""
        for dimension in self.config.test_dimensions:
            suite_id = f"SUITE_{dimension.upper()}"
            self.test_suites[suite_id] = TestSuite(suite_id, dimension)
            
    def generate_test_cases(self) -> None:
        """Generate test cases with fractal patterns"""
        for scale in self.config.test_scales:
            for i in range(self.config.num_test_cases):
                # Generate fractal test pattern
                test_pattern = self._generate_test_pattern(scale)
                
                # Create test case for each dimension
                for dimension in self.config.test_dimensions:
                    test_id = f"TEST_{dimension}_{scale}_{i}"
                    test_case = TestCase(
                        test_id,
                        test_pattern['input'],
                        test_pattern['expected']
                    )
                    
                    suite_id = f"SUITE_{dimension.upper()}"
                    self.test_suites[suite_id].add_test(test_case)
                    
    def _generate_test_pattern(self, scale: int) -> Dict[str, np.ndarray]:
        """Generate fractal test pattern"""
        # Create base pattern
        size = 32 * scale
        pattern = np.zeros(size)
        
        # Add fractal components
        for i in range(self.config.pattern_complexity):
            freq = 2 ** i
            pattern += np.sin(np.linspace(0, freq * np.pi, size))
            
        return {
            'input': pattern,
            'expected': np.tanh(pattern)  # Expected transformation
        }
        
    def run_tests(self, test_fn: Callable) -> Dict[str, Any]:
        """Run all test suites"""
        results = {}
        start_time = time.time()
        
        for suite_id, suite in self.test_suites.items():
            suite_results = suite.run_suite(test_fn)
            results[suite_id] = suite_results
            
        end_time = time.time()
        
        # Store test run
        test_run = {
            'timestamp': start_time,
            'duration': end_time - start_time,
            'results': results,
            'metrics': self._compute_test_metrics(results)
        }
        self.test_history.append(test_run)
        
        return test_run
        
    def _compute_test_metrics(self, 
                            results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compute comprehensive test metrics"""
        return {
            'dimension_metrics': self._compute_dimension_metrics(results),
            'scale_metrics': self._compute_scale_metrics(results),
            'overall_metrics': self._compute_overall_metrics(results)
        }
        
    def _compute_dimension_metrics(self, 
                                 results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Compute metrics for each dimension"""
        dimension_metrics = {}
        
        for suite_id, suite_results in results.items():
            dimension = suite_results['dimension']
            metrics = suite_results['metrics']
            dimension_metrics[dimension] = metrics
            
        return dimension_metrics
        
    def _compute_scale_metrics(self, 
                             results: Dict[str, Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
        """Compute metrics for each scale"""
        scale_metrics = {scale: {
            'success_rate': [],
            'mean_duration': [],
            'error_rate': []
        } for scale in self.config.test_scales}
        
        for suite_results in results.values():
            for test_results in suite_results['results'].values():
                test_id = test_results['test_id']
                scale = int(test_id.split('_')[2])
                
                scale_metrics[scale]['success_rate'].append(
                    float(test_results['success'])
                )
                scale_metrics[scale]['mean_duration'].append(
                    test_results['metrics']['duration']
                )
                scale_metrics[scale]['error_rate'].append(
                    float(not test_results['success'])
                )
                
        # Compute averages
        for scale in scale_metrics:
            for metric in scale_metrics[scale]:
                values = scale_metrics[scale][metric]
                scale_metrics[scale][metric] = float(np.mean(values))
                
        return scale_metrics
        
    def _compute_overall_metrics(self, 
                               results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Compute overall test metrics"""
        all_successes = []
        all_durations = []
        
        for suite_results in results.values():
            for test_results in suite_results['results'].values():
                all_successes.append(float(test_results['success']))
                all_durations.append(test_results['metrics']['duration'])
                
        return {
            'total_success_rate': float(np.mean(all_successes)),
            'total_duration': float(np.sum(all_durations)),
            'mean_test_duration': float(np.mean(all_durations)),
            'test_stability': float(1.0 - np.std(all_successes))
        }
        
    def save_results(self, filename: Optional[str] = None) -> None:
        """Save test results to file"""
        if filename is None:
            filename = f"fracti_test_results_{int(time.time())}.json"
            
        results = {
            'config': self.config.__dict__,
            'test_history': self.test_history
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2) 