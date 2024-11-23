"""
FractiPipeline.py

Implements comprehensive pipeline framework for FractiAI, enabling workflow
orchestration and management through fractal-based patterns and organization.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging
import time
from collections import deque
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, Future

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for pipeline framework"""
    max_workers: int = 4
    timeout_seconds: int = 300
    retry_attempts: int = 3
    buffer_size: int = 1000
    pattern_threshold: float = 0.7
    fractal_depth: int = 3

class PipelineStage:
    """Individual stage in processing pipeline"""
    
    def __init__(self, stage_id: str, processor: Callable):
        self.stage_id = stage_id
        self.processor = processor
        self.input_buffer = deque(maxlen=1000)
        self.output_buffer = deque(maxlen=1000)
        self.metrics = {}
        self.dependencies = []
        
    def process(self, data: Any) -> Any:
        """Process input data"""
        try:
            # Store input
            self.input_buffer.append(data)
            
            # Process data
            start_time = time.time()
            result = self.processor(data)
            duration = time.time() - start_time
            
            # Store output and metrics
            self.output_buffer.append(result)
            self._update_metrics(duration)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in stage {self.stage_id}: {str(e)}")
            raise
            
    def add_dependency(self, stage: 'PipelineStage') -> None:
        """Add dependency on another stage"""
        self.dependencies.append(stage)
        
    def _update_metrics(self, duration: float) -> None:
        """Update stage metrics"""
        if 'durations' not in self.metrics:
            self.metrics['durations'] = deque(maxlen=1000)
            
        self.metrics['durations'].append(duration)
        self.metrics['mean_duration'] = float(np.mean(self.metrics['durations']))
        self.metrics['throughput'] = float(len(self.metrics['durations']) / 
                                         (time.time() - self.metrics.get('start_time', time.time())))

class Pipeline:
    """Processing pipeline with fractal organization"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stages: Dict[str, PipelineStage] = {}
        self.execution_graph = nx.DiGraph()
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.pattern_buffer = deque(maxlen=config.buffer_size)
        
    def add_stage(self, stage_id: str, processor: Callable, 
                  dependencies: Optional[List[str]] = None) -> None:
        """Add processing stage to pipeline"""
        # Create stage
        stage = PipelineStage(stage_id, processor)
        
        # Add dependencies
        if dependencies:
            for dep_id in dependencies:
                if dep_id in self.stages:
                    stage.add_dependency(self.stages[dep_id])
                    
        # Update execution graph
        self.execution_graph.add_node(stage_id)
        if dependencies:
            for dep_id in dependencies:
                self.execution_graph.add_edge(dep_id, stage_id)
                
        # Store stage
        self.stages[stage_id] = stage
        
    def process(self, data: Any) -> Dict[str, Any]:
        """Process data through pipeline"""
        # Validate pipeline
        if not nx.is_directed_acyclic_graph(self.execution_graph):
            raise ValueError("Pipeline contains cycles")
            
        # Get execution order
        execution_order = list(nx.topological_sort(self.execution_graph))
        
        # Execute stages
        results = {}
        futures: Dict[str, Future] = {}
        
        try:
            # Submit initial stages
            ready_stages = [stage_id for stage_id in execution_order 
                          if not self.stages[stage_id].dependencies]
                          
            for stage_id in ready_stages:
                futures[stage_id] = self.executor.submit(
                    self.stages[stage_id].process,
                    data
                )
                
            # Process remaining stages
            while futures:
                # Get completed stages
                done = {stage_id: future for stage_id, future in futures.items()
                       if future.done()}
                
                for stage_id, future in done.items():
                    try:
                        results[stage_id] = future.result(
                            timeout=self.config.timeout_seconds
                        )
                    except Exception as e:
                        logger.error(f"Stage {stage_id} failed: {str(e)}")
                        results[stage_id] = None
                        
                    # Remove from futures
                    del futures[stage_id]
                    
                    # Submit dependent stages
                    for next_stage in self.execution_graph.successors(stage_id):
                        # Check if all dependencies are complete
                        deps = list(self.execution_graph.predecessors(next_stage))
                        if all(dep in results for dep in deps):
                            # Get inputs from dependencies
                            inputs = [results[dep] for dep in deps]
                            
                            # Submit stage
                            futures[next_stage] = self.executor.submit(
                                self.stages[next_stage].process,
                                inputs[0] if len(inputs) == 1 else inputs
                            )
                            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
            
        # Store pattern
        self._store_pattern(results)
        
        return {
            'results': results,
            'metrics': self._compute_metrics()
        }
        
    def _store_pattern(self, results: Dict[str, Any]) -> None:
        """Store processing pattern"""
        pattern = {
            'structure': {stage_id: bool(result is not None)
                         for stage_id, result in results.items()},
            'timing': {stage_id: self.stages[stage_id].metrics.get('mean_duration', 0)
                      for stage_id in results}
        }
        self.pattern_buffer.append(pattern)
        
    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute pipeline metrics"""
        return {
            'stage_metrics': {
                stage_id: stage.metrics
                for stage_id, stage in self.stages.items()
            },
            'pipeline_metrics': {
                'total_stages': len(self.stages),
                'execution_time': self._compute_execution_time(),
                'throughput': self._compute_throughput(),
                'pattern_stability': self._compute_pattern_stability()
            }
        }
        
    def _compute_execution_time(self) -> float:
        """Compute total execution time"""
        if not self.pattern_buffer:
            return 0.0
            
        stage_times = []
        for pattern in self.pattern_buffer:
            stage_times.append(sum(pattern['timing'].values()))
            
        return float(np.mean(stage_times))
        
    def _compute_throughput(self) -> float:
        """Compute pipeline throughput"""
        stage_throughputs = []
        for stage in self.stages.values():
            if 'throughput' in stage.metrics:
                stage_throughputs.append(stage.metrics['throughput'])
                
        return float(np.mean(stage_throughputs)) if stage_throughputs else 0.0
        
    def _compute_pattern_stability(self) -> float:
        """Compute stability of processing patterns"""
        if len(self.pattern_buffer) < 2:
            return 1.0
            
        # Compare consecutive patterns
        stabilities = []
        patterns = list(self.pattern_buffer)
        
        for i in range(len(patterns)-1):
            stability = self._compare_patterns(patterns[i], patterns[i+1])
            stabilities.append(stability)
            
        return float(np.mean(stabilities))
        
    def _compare_patterns(self, pattern1: Dict[str, Any],
                         pattern2: Dict[str, Any]) -> float:
        """Compare two processing patterns"""
        # Compare structure
        structure_match = np.mean([
            float(pattern1['structure'][stage_id] == pattern2['structure'][stage_id])
            for stage_id in pattern1['structure']
        ])
        
        # Compare timing
        timing_diffs = []
        for stage_id in pattern1['timing']:
            if stage_id in pattern2['timing']:
                time1 = pattern1['timing'][stage_id]
                time2 = pattern2['timing'][stage_id]
                if time1 > 0 and time2 > 0:
                    diff = abs(time1 - time2) / max(time1, time2)
                    timing_diffs.append(1.0 - diff)
                    
        timing_match = np.mean(timing_diffs) if timing_diffs else 0.0
        
        return float((structure_match + timing_match) / 2)

class FractiPipeline:
    """Main pipeline system with fractal awareness"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.pipelines: Dict[str, Pipeline] = {}
        
    def create_pipeline(self, pipeline_id: str) -> Pipeline:
        """Create new processing pipeline"""
        pipeline = Pipeline(self.config)
        self.pipelines[pipeline_id] = pipeline
        return pipeline
        
    def process(self, pipeline_id: str, data: Any) -> Dict[str, Any]:
        """Process data through pipeline"""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
            
        return self.pipelines[pipeline_id].process(data)
        
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all pipelines"""
        return {
            pipeline_id: pipeline._compute_metrics()
            for pipeline_id, pipeline in self.pipelines.items()
        } 