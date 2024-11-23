"""
FractiValidator.py

Implements comprehensive validation and testing framework for FractiAI,
ensuring system integrity, coherence, and performance across all components.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import json
import time

logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """Configuration for FractiAI validation"""
    test_batch_size: int = 100
    validation_threshold: float = 0.9
    parallel_tests: int = 4
    timeout_seconds: int = 300
    error_tolerance: float = 1e-6
    save_results: bool = True

class ValidationResult:
    """Stores and analyzes validation results"""
    
    def __init__(self, component: str, test_type: str):
        self.component = component
        self.test_type = test_type
        self.timestamp = time.time()
        self.metrics = {}
        self.errors = []
        self.passed = False
        
    def add_metric(self, name: str, value: float) -> None:
        """Add validation metric"""
        self.metrics[name] = value
        
    def add_error(self, error: str) -> None:
        """Add validation error"""
        self.errors.append(error)
        
    def set_passed(self, passed: bool) -> None:
        """Set validation pass/fail status"""
        self.passed = passed
        
    def to_dict(self) -> Dict:
        """Convert result to dictionary"""
        return {
            'component': self.component,
            'test_type': self.test_type,
            'timestamp': self.timestamp,
            'metrics': self.metrics,
            'errors': self.errors,
            'passed': self.passed
        }

class ComponentValidator:
    """Base class for component-specific validation"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.results = []
        
    def validate(self, component: Any) -> ValidationResult:
        """Validate component"""
        raise NotImplementedError

class UnipixelValidator(ComponentValidator):
    """Validates Unipixel components"""
    
    def validate(self, unipixel: Any) -> ValidationResult:
        result = ValidationResult('Unipixel', 'component')
        
        # Validate state dimensions
        state_valid = self._validate_state_dimensions(unipixel, result)
        
        # Validate processing capabilities
        processing_valid = self._validate_processing(unipixel, result)
        
        # Validate connections
        connections_valid = self._validate_connections(unipixel, result)
        
        # Set overall result
        result.set_passed(all([state_valid, processing_valid, connections_valid]))
        
        return result
    
    def _validate_state_dimensions(self, unipixel: Any, 
                                 result: ValidationResult) -> bool:
        """Validate Unipixel state dimensions"""
        try:
            if not hasattr(unipixel, 'state') or unipixel.state is None:
                result.add_error("Missing state attribute")
                return False
                
            if not isinstance(unipixel.state, np.ndarray):
                result.add_error("State is not numpy array")
                return False
                
            if unipixel.state.shape[0] != 3:  # Organic, Inorganic, Abstract
                result.add_error(f"Invalid state dimensions: {unipixel.state.shape}")
                return False
                
            result.add_metric('state_dimension_score', 1.0)
            return True
            
        except Exception as e:
            result.add_error(f"State validation error: {str(e)}")
            return False
    
    def _validate_processing(self, unipixel: Any, 
                           result: ValidationResult) -> bool:
        """Validate Unipixel processing capabilities"""
        try:
            # Test data processing
            test_input = np.random.randn(3)
            processed = unipixel.process(test_input)
            
            if not isinstance(processed, np.ndarray):
                result.add_error("Invalid processing output type")
                return False
                
            if processed.shape != test_input.shape:
                result.add_error("Processing changed input dimensions")
                return False
                
            result.add_metric('processing_score', 1.0)
            return True
            
        except Exception as e:
            result.add_error(f"Processing validation error: {str(e)}")
            return False
    
    def _validate_connections(self, unipixel: Any, 
                            result: ValidationResult) -> bool:
        """Validate Unipixel connections"""
        try:
            if not hasattr(unipixel, 'connections'):
                result.add_error("Missing connections attribute")
                return False
                
            if not isinstance(unipixel.connections, list):
                result.add_error("Connections not stored as list")
                return False
                
            result.add_metric('connections_score', 1.0)
            return True
            
        except Exception as e:
            result.add_error(f"Connections validation error: {str(e)}")
            return False

class FractiValidator:
    """Main validation system for FractiAI"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.validators = {
            'unipixel': UnipixelValidator(config)
        }
        self.results_history = []
        self.executor = ThreadPoolExecutor(max_workers=config.parallel_tests)
        
    def validate_system(self, system_components: Dict[str, Any]) -> Dict[str, Any]:
        """Validate entire FractiAI system"""
        validation_results = {}
        futures = []
        
        # Submit validation tasks
        for component_type, components in system_components.items():
            if component_type in self.validators:
                validator = self.validators[component_type]
                
                if isinstance(components, list):
                    for component in components:
                        futures.append(
                            self.executor.submit(validator.validate, component)
                        )
                else:
                    futures.append(
                        self.executor.submit(validator.validate, components)
                    )
        
        # Collect results
        for future in futures:
            try:
                result = future.result(timeout=self.config.timeout_seconds)
                if result.component not in validation_results:
                    validation_results[result.component] = []
                validation_results[result.component].append(result)
            except Exception as e:
                logger.error(f"Validation error: {str(e)}")
        
        # Store results
        self.results_history.extend([
            result for results in validation_results.values()
            for result in results
        ])
        
        # Save results if configured
        if self.config.save_results:
            self._save_results(validation_results)
        
        return self._compute_validation_summary(validation_results)
    
    def _compute_validation_summary(self, 
                                  validation_results: Dict[str, List[ValidationResult]]) -> Dict:
        """Compute summary of validation results"""
        summary = {
            'timestamp': time.time(),
            'overall_status': True,
            'component_status': {},
            'metrics': {},
            'error_count': 0
        }
        
        for component, results in validation_results.items():
            component_passed = all(result.passed for result in results)
            summary['overall_status'] &= component_passed
            
            summary['component_status'][component] = {
                'passed': component_passed,
                'total_tests': len(results),
                'passed_tests': sum(1 for r in results if r.passed)
            }
            
            # Aggregate metrics
            component_metrics = {}
            for result in results:
                for metric, value in result.metrics.items():
                    if metric not in component_metrics:
                        component_metrics[metric] = []
                    component_metrics[metric].append(value)
            
            summary['metrics'][component] = {
                metric: float(np.mean(values))
                for metric, values in component_metrics.items()
            }
            
            summary['error_count'] += sum(len(result.errors) for result in results)
        
        return summary
    
    def _save_results(self, validation_results: Dict[str, List[ValidationResult]]) -> None:
        """Save validation results to file"""
        try:
            results_dict = {
                component: [r.to_dict() for r in results]
                for component, results in validation_results.items()
            }
            
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"validation_results_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(results_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving validation results: {str(e)}") 