"""
FractiDocs.py

Implements comprehensive documentation generation for FractiAI, enabling automatic
documentation across components, patterns, and scales through fractal organization.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import inspect
import ast
import json
import re
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class DocsConfig:
    """Configuration for documentation generation"""
    output_format: str = "markdown"  # markdown, html, or rst
    code_style: str = "github"
    include_examples: bool = True
    include_tests: bool = True
    include_metrics: bool = True
    fractal_depth: int = 3

class CodeAnalyzer:
    """Analyzes code structure and patterns"""
    
    def __init__(self, config: DocsConfig):
        self.config = config
        self.patterns = {}
        self.dependencies = {}
        self.metrics = {}
        
    def analyze_file(self, filepath: str) -> Dict[str, Any]:
        """Analyze single file"""
        with open(filepath, 'r') as f:
            content = f.read()
            
        tree = ast.parse(content)
        
        analysis = {
            'classes': self._analyze_classes(tree),
            'functions': self._analyze_functions(tree),
            'patterns': self._analyze_patterns(tree),
            'metrics': self._compute_metrics(tree)
        }
        
        return analysis
        
    def _analyze_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze class definitions"""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'docstring': ast.get_docstring(node),
                    'methods': self._analyze_methods(node),
                    'attributes': self._analyze_attributes(node),
                    'bases': [base.id for base in node.bases 
                             if isinstance(base, ast.Name)]
                }
                classes.append(class_info)
                
        return classes
        
    def _analyze_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze function definitions"""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    'name': node.name,
                    'docstring': ast.get_docstring(node),
                    'arguments': self._analyze_arguments(node),
                    'returns': self._analyze_returns(node),
                    'complexity': self._compute_complexity(node)
                }
                functions.append(func_info)
                
        return functions
        
    def _analyze_methods(self, class_node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Analyze class methods"""
        methods = []
        
        for node in ast.walk(class_node):
            if isinstance(node, ast.FunctionDef):
                method_info = {
                    'name': node.name,
                    'docstring': ast.get_docstring(node),
                    'arguments': self._analyze_arguments(node),
                    'returns': self._analyze_returns(node),
                    'decorators': [d.id for d in node.decorator_list 
                                 if isinstance(d, ast.Name)]
                }
                methods.append(method_info)
                
        return methods
        
    def _analyze_attributes(self, class_node: ast.ClassDef) -> List[Dict[str, str]]:
        """Analyze class attributes"""
        attributes = []
        
        for node in ast.walk(class_node):
            if isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    attr_info = {
                        'name': node.target.id,
                        'type': self._get_type_annotation(node.annotation)
                    }
                    attributes.append(attr_info)
                    
        return attributes
        
    def _analyze_arguments(self, func_node: ast.FunctionDef) -> List[Dict[str, str]]:
        """Analyze function arguments"""
        arguments = []
        
        for arg in func_node.args.args:
            arg_info = {
                'name': arg.arg,
                'type': self._get_type_annotation(arg.annotation)
            }
            arguments.append(arg_info)
            
        return arguments
        
    def _analyze_returns(self, func_node: ast.FunctionDef) -> Optional[str]:
        """Analyze function return type"""
        if func_node.returns:
            return self._get_type_annotation(func_node.returns)
        return None
        
    def _analyze_patterns(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code patterns"""
        patterns = {
            'fractal_patterns': self._find_fractal_patterns(tree),
            'recursive_patterns': self._find_recursive_patterns(tree),
            'parallel_patterns': self._find_parallel_patterns(tree)
        }
        
        return patterns
        
    def _compute_metrics(self, tree: ast.AST) -> Dict[str, float]:
        """Compute code metrics"""
        return {
            'complexity': self._compute_total_complexity(tree),
            'cohesion': self._compute_cohesion(tree),
            'coupling': self._compute_coupling(tree)
        }
        
    def _get_type_annotation(self, node: Optional[ast.AST]) -> Optional[str]:
        """Get string representation of type annotation"""
        if node is None:
            return None
            
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Subscript):
            return f"{node.value.id}[{self._get_type_annotation(node.slice.value)}]"
        return str(node)
        
    def _compute_complexity(self, node: ast.AST) -> int:
        """Compute cyclomatic complexity"""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try,
                                ast.ExceptHandler, ast.With, ast.Assert)):
                complexity += 1
                
        return complexity
        
    def _find_fractal_patterns(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find fractal patterns in code"""
        patterns = []
        
        # Look for recursive structures with self-similar patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if self._has_fractal_structure(node):
                    patterns.append({
                        'type': 'class',
                        'name': node.name,
                        'depth': self._compute_fractal_depth(node)
                    })
                    
        return patterns
        
    def _has_fractal_structure(self, node: ast.ClassDef) -> bool:
        """Check if class has fractal structure"""
        # Look for recursive methods and self-similar patterns
        has_recursive = False
        has_self_similar = False
        
        for child in ast.walk(node):
            if isinstance(child, ast.FunctionDef):
                if self._is_recursive(child):
                    has_recursive = True
                if self._has_self_similar_pattern(child):
                    has_self_similar = True
                    
        return has_recursive and has_self_similar
        
    def _is_recursive(self, node: ast.FunctionDef) -> bool:
        """Check if function is recursive"""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id == node.name:
                    return True
        return False
        
    def _has_self_similar_pattern(self, node: ast.FunctionDef) -> bool:
        """Check for self-similar patterns in function"""
        # Look for repeated structural patterns
        structure = self._get_node_structure(node)
        return self._is_self_similar(structure)
        
    def _get_node_structure(self, node: ast.AST) -> List[str]:
        """Get structural representation of node"""
        structure = []
        for child in ast.iter_child_nodes(node):
            structure.append(type(child).__name__)
        return structure
        
    def _is_self_similar(self, structure: List[str]) -> bool:
        """Check if structure is self-similar"""
        if len(structure) < 4:
            return False
            
        # Look for repeated patterns at different scales
        for size in range(2, len(structure)//2):
            patterns = [structure[i:i+size] for i in range(0, len(structure)-size, size)]
            if len(set(map(tuple, patterns))) == 1:
                return True
                
        return False

class DocumentationGenerator:
    """Generates documentation from code analysis"""
    
    def __init__(self, config: DocsConfig):
        self.config = config
        self.analyzer = CodeAnalyzer(config)
        
    def generate_docs(self, src_dir: str, output_dir: str) -> None:
        """Generate documentation for source directory"""
        # Analyze all Python files
        analyses = {}
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    analyses[filepath] = self.analyzer.analyze_file(filepath)
                    
        # Generate documentation
        self._generate_overview(analyses, output_dir)
        self._generate_component_docs(analyses, output_dir)
        self._generate_pattern_docs(analyses, output_dir)
        if self.config.include_metrics:
            self._generate_metrics_docs(analyses, output_dir)
            
    def _generate_overview(self, analyses: Dict[str, Dict], 
                         output_dir: str) -> None:
        """Generate overview documentation"""
        overview = {
            'title': 'FractiAI Documentation',
            'components': self._summarize_components(analyses),
            'patterns': self._summarize_patterns(analyses),
            'metrics': self._summarize_metrics(analyses)
        }
        
        self._write_doc(overview, os.path.join(output_dir, 'overview'))
        
    def _generate_component_docs(self, analyses: Dict[str, Dict],
                               output_dir: str) -> None:
        """Generate component-specific documentation"""
        components_dir = os.path.join(output_dir, 'components')
        os.makedirs(components_dir, exist_ok=True)
        
        for filepath, analysis in analyses.items():
            component_name = os.path.basename(filepath)[:-3]
            doc = {
                'title': f'Component: {component_name}',
                'classes': analysis['classes'],
                'functions': analysis['functions'],
                'patterns': analysis['patterns']
            }
            
            self._write_doc(doc, os.path.join(components_dir, component_name))
            
    def _generate_pattern_docs(self, analyses: Dict[str, Dict],
                             output_dir: str) -> None:
        """Generate pattern-specific documentation"""
        patterns_dir = os.path.join(output_dir, 'patterns')
        os.makedirs(patterns_dir, exist_ok=True)
        
        patterns = self._collect_patterns(analyses)
        for pattern_type, pattern_info in patterns.items():
            doc = {
                'title': f'Pattern: {pattern_type}',
                'description': pattern_info['description'],
                'examples': pattern_info['examples'],
                'metrics': pattern_info['metrics']
            }
            
            self._write_doc(doc, os.path.join(patterns_dir, pattern_type))
            
    def _generate_metrics_docs(self, analyses: Dict[str, Dict],
                             output_dir: str) -> None:
        """Generate metrics documentation"""
        metrics_dir = os.path.join(output_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        metrics = self._collect_metrics(analyses)
        for metric_type, metric_info in metrics.items():
            doc = {
                'title': f'Metric: {metric_type}',
                'description': metric_info['description'],
                'distribution': metric_info['distribution'],
                'trends': metric_info['trends']
            }
            
            self._write_doc(doc, os.path.join(metrics_dir, metric_type))
            
    def _write_doc(self, content: Dict[str, Any], filepath: str) -> None:
        """Write documentation to file"""
        if self.config.output_format == 'markdown':
            self._write_markdown(content, f"{filepath}.md")
        elif self.config.output_format == 'html':
            self._write_html(content, f"{filepath}.html")
        elif self.config.output_format == 'rst':
            self._write_rst(content, f"{filepath}.rst")
            
    def _write_markdown(self, content: Dict[str, Any], filepath: str) -> None:
        """Write content in Markdown format"""
        with open(filepath, 'w') as f:
            f.write(f"# {content['title']}\n\n")
            
            if 'components' in content:
                f.write("## Components\n\n")
                for component in content['components']:
                    f.write(f"### {component['name']}\n")
                    f.write(f"{component['description']}\n\n")
                    
            if 'patterns' in content:
                f.write("## Patterns\n\n")
                for pattern in content['patterns']:
                    f.write(f"### {pattern['type']}\n")
                    f.write(f"{pattern['description']}\n\n")
                    
            if 'metrics' in content:
                f.write("## Metrics\n\n")
                for metric in content['metrics']:
                    f.write(f"### {metric['name']}\n")
                    f.write(f"{metric['description']}\n\n")
                    
    def _write_html(self, content: Dict[str, Any], filepath: str) -> None:
        """Write content in HTML format"""
        # Similar to markdown but with HTML tags
        pass
        
    def _write_rst(self, content: Dict[str, Any], filepath: str) -> None:
        """Write content in reStructuredText format"""
        # Similar to markdown but with RST syntax
        pass
        
    def _summarize_components(self, analyses: Dict[str, Dict]) -> List[Dict[str, str]]:
        """Create component summaries"""
        summaries = []
        
        for filepath, analysis in analyses.items():
            component_name = os.path.basename(filepath)[:-3]
            summary = {
                'name': component_name,
                'description': self._extract_module_description(filepath),
                'num_classes': len(analysis['classes']),
                'num_functions': len(analysis['functions'])
            }
            summaries.append(summary)
            
        return summaries
        
    def _summarize_patterns(self, analyses: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Create pattern summaries"""
        patterns = []
        
        for analysis in analyses.values():
            for pattern_type, pattern_info in analysis['patterns'].items():
                if pattern_info:  # If patterns were found
                    patterns.append({
                        'type': pattern_type,
                        'count': len(pattern_info),
                        'examples': pattern_info[:3]  # First 3 examples
                    })
                    
        return patterns
        
    def _summarize_metrics(self, analyses: Dict[str, Dict]) -> List[Dict[str, float]]:
        """Create metrics summaries"""
        metrics = []
        
        for analysis in analyses.values():
            for metric_name, value in analysis['metrics'].items():
                metrics.append({
                    'name': metric_name,
                    'value': value,
                    'description': self._get_metric_description(metric_name)
                })
                
        return metrics
        
    def _extract_module_description(self, filepath: str) -> str:
        """Extract module docstring"""
        with open(filepath, 'r') as f:
            content = f.read()
            
        tree = ast.parse(content)
        return ast.get_docstring(tree) or "No description available."
        
    def _get_metric_description(self, metric_name: str) -> str:
        """Get description for metric"""
        descriptions = {
            'complexity': 'Cyclomatic complexity measure',
            'cohesion': 'Measure of component cohesion',
            'coupling': 'Measure of component coupling'
        }
        return descriptions.get(metric_name, "No description available.")
        
    def _collect_patterns(self, analyses: Dict[str, Dict]) -> Dict[str, Dict[str, Any]]:
        """Collect and organize patterns"""
        patterns = {}
        
        for analysis in analyses.values():
            for pattern_type, pattern_info in analysis['patterns'].items():
                if pattern_type not in patterns:
                    patterns[pattern_type] = {
                        'description': self._get_pattern_description(pattern_type),
                        'examples': [],
                        'metrics': {}
                    }
                    
                patterns[pattern_type]['examples'].extend(pattern_info)
                
        return patterns
        
    def _collect_metrics(self, analyses: Dict[str, Dict]) -> Dict[str, Dict[str, Any]]:
        """Collect and organize metrics"""
        metrics = {}
        
        for analysis in analyses.values():
            for metric_name, value in analysis['metrics'].items():
                if metric_name not in metrics:
                    metrics[metric_name] = {
                        'description': self._get_metric_description(metric_name),
                        'distribution': [],
                        'trends': []
                    }
                    
                metrics[metric_name]['distribution'].append(value)
                
        # Compute trends
        for metric_info in metrics.values():
            values = np.array(metric_info['distribution'])
            metric_info['trends'] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'trend': float(np.mean(np.diff(values))) if len(values) > 1 else 0.0
            }
            
        return metrics
        
    def _get_pattern_description(self, pattern_type: str) -> str:
        """Get description for pattern type"""
        descriptions = {
            'fractal_patterns': 'Self-similar structural patterns',
            'recursive_patterns': 'Recursive implementation patterns',
            'parallel_patterns': 'Parallel processing patterns'
        }
        return descriptions.get(pattern_type, "No description available.")
</rewritten_file> 