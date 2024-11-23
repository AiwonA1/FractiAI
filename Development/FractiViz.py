"""
FractiViz.py

Implements comprehensive visualization framework for FractiAI, enabling visualization
and analysis of patterns, scales, and system behavior through fractal-based visualization.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import entropy
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)

@dataclass
class VizConfig:
    """Configuration for visualization framework"""
    plot_dimensions: Tuple[int, int] = (1200, 800)
    color_scheme: str = 'viridis'
    interactive: bool = True
    save_plots: bool = True
    animation_fps: int = 30
    detail_level: int = 3

class PatternVisualizer:
    """Visualizes fractal patterns and their evolution"""
    
    def __init__(self, config: VizConfig):
        self.config = config
        self.pattern_history = []
        plt.style.use('seaborn')
        
    def visualize_pattern(self, pattern: np.ndarray, 
                         title: str = "Pattern Visualization") -> go.Figure:
        """Create interactive pattern visualization"""
        fig = go.Figure()
        
        # Add pattern trace
        fig.add_trace(go.Scatter(
            y=pattern,
            mode='lines',
            name='Pattern',
            line=dict(color='blue', width=2)
        ))
        
        # Add fractal decomposition
        scales = self._compute_fractal_scales(pattern)
        for scale, decomp in scales.items():
            fig.add_trace(go.Scatter(
                y=decomp,
                mode='lines',
                name=f'Scale {scale}',
                opacity=0.5
            ))
            
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Index",
            yaxis_title="Value",
            width=self.config.plot_dimensions[0],
            height=self.config.plot_dimensions[1],
            showlegend=True
        )
        
        self.pattern_history.append({
            'pattern': pattern,
            'scales': scales,
            'timestamp': time.time()
        })
        
        return fig
        
    def _compute_fractal_scales(self, pattern: np.ndarray) -> Dict[int, np.ndarray]:
        """Compute pattern at different scales"""
        scales = {}
        for i in range(self.config.detail_level):
            scale_factor = 2 ** i
            scaled = self._rescale_pattern(pattern, len(pattern) // scale_factor)
            scales[scale_factor] = scaled
        return scales
        
    def _rescale_pattern(self, pattern: np.ndarray, target_size: int) -> np.ndarray:
        """Rescale pattern to target size"""
        if len(pattern) == target_size:
            return pattern
            
        # Simple averaging for downscaling
        if len(pattern) > target_size:
            ratio = len(pattern) // target_size
            return np.mean(pattern.reshape(-1, ratio), axis=1)
            
        # Simple interpolation for upscaling
        ratio = target_size // len(pattern)
        return np.repeat(pattern, ratio)

class NetworkVisualizer:
    """Visualizes network structures and relationships"""
    
    def __init__(self, config: VizConfig):
        self.config = config
        self.network_history = []
        
    def visualize_network(self, adjacency: np.ndarray, 
                         node_attrs: Optional[Dict[int, Dict]] = None,
                         title: str = "Network Visualization") -> go.Figure:
        """Create interactive network visualization"""
        G = nx.from_numpy_array(adjacency)
        
        # Compute layout
        pos = nx.spring_layout(G)
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        ))
        
        # Add nodes
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
        # Node colors based on attributes
        node_color = []
        node_size = []
        if node_attrs:
            for node in G.nodes():
                attrs = node_attrs.get(node, {})
                node_color.append(attrs.get('color', '#000'))
                node_size.append(attrs.get('size', 10))
        else:
            node_color = ['#000'] * len(G.nodes())
            node_size = [10] * len(G.nodes())
            
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                color=node_color,
                size=node_size,
                line_width=2
            )
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            showlegend=False,
            width=self.config.plot_dimensions[0],
            height=self.config.plot_dimensions[1],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        self.network_history.append({
            'adjacency': adjacency,
            'node_attrs': node_attrs,
            'timestamp': time.time()
        })
        
        return fig

class MetricsVisualizer:
    """Visualizes system metrics and performance"""
    
    def __init__(self, config: VizConfig):
        self.config = config
        self.metrics_history = []
        
    def visualize_metrics(self, metrics: Dict[str, float], 
                         categories: Optional[List[str]] = None,
                         title: str = "System Metrics") -> go.Figure:
        """Create interactive metrics visualization"""
        if categories is None:
            categories = list(metrics.keys())
            
        values = [metrics[cat] for cat in categories]
        
        # Create figure
        fig = go.Figure()
        
        # Add bar chart
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=px.colors.qualitative.Set3
        ))
        
        # Add trend line if historical data available
        if self.metrics_history:
            historical_values = []
            for history in self.metrics_history:
                hist_metrics = history['metrics']
                historical_values.append([hist_metrics.get(cat, 0) for cat in categories])
                
            trends = np.mean(historical_values, axis=0)
            
            fig.add_trace(go.Scatter(
                x=categories,
                y=trends,
                mode='lines',
                name='Trend',
                line=dict(color='red', width=2)
            ))
            
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Metric",
            yaxis_title="Value",
            width=self.config.plot_dimensions[0],
            height=self.config.plot_dimensions[1],
            showlegend=True
        )
        
        self.metrics_history.append({
            'metrics': metrics,
            'categories': categories,
            'timestamp': time.time()
        })
        
        return fig

class FractiViz:
    """Main visualization system with fractal awareness"""
    
    def __init__(self, config: VizConfig):
        self.config = config
        self.pattern_viz = PatternVisualizer(config)
        self.network_viz = NetworkVisualizer(config)
        self.metrics_viz = MetricsVisualizer(config)
        self.visualization_history = []
        
    def visualize_system(self, system_state: Dict[str, Any]) -> Dict[str, go.Figure]:
        """Create comprehensive system visualization"""
        visualizations = {}
        
        # Visualize patterns
        if 'patterns' in system_state:
            for pattern_id, pattern in system_state['patterns'].items():
                viz = self.pattern_viz.visualize_pattern(
                    pattern,
                    title=f"Pattern {pattern_id}"
                )
                visualizations[f"pattern_{pattern_id}"] = viz
                
        # Visualize network
        if 'network' in system_state:
            viz = self.network_viz.visualize_network(
                system_state['network'],
                system_state.get('node_attributes'),
                title="System Network"
            )
            visualizations['network'] = viz
            
        # Visualize metrics
        if 'metrics' in system_state:
            viz = self.metrics_viz.visualize_metrics(
                system_state['metrics'],
                title="System Metrics"
            )
            visualizations['metrics'] = viz
            
        # Store visualization
        self.visualization_history.append({
            'system_state': system_state,
            'visualizations': visualizations,
            'timestamp': time.time()
        })
        
        return visualizations
        
    def create_animation(self, pattern_id: str) -> go.Figure:
        """Create animation of pattern evolution"""
        frames = []
        for history in self.visualization_history:
            if f"pattern_{pattern_id}" in history['visualizations']:
                frames.append(go.Frame(
                    data=history['visualizations'][f"pattern_{pattern_id}"].data
                ))
                
        if not frames:
            return None
            
        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {'label': 'Play',
                     'method': 'animate',
                     'args': [None, {'frame': {'duration': 1000/self.config.animation_fps}}]},
                    {'label': 'Pause',
                     'method': 'animate',
                     'args': [[None], {'frame': {'duration': 0}}]}
                ]
            }]
        )
        
        return fig
        
    def save_visualization(self, viz_id: str, 
                         filename: Optional[str] = None) -> None:
        """Save visualization to file"""
        if not self.visualization_history:
            return
            
        latest = self.visualization_history[-1]
        if viz_id not in latest['visualizations']:
            return
            
        if filename is None:
            filename = f"fracti_viz_{viz_id}_{int(time.time())}.html"
            
        fig = latest['visualizations'][viz_id]
        fig.write_html(filename)
        
    def get_visualization_metrics(self) -> Dict[str, float]:
        """Get visualization system metrics"""
        if not self.visualization_history:
            return {}
            
        return {
            'num_visualizations': len(self.visualization_history),
            'pattern_complexity': self._compute_pattern_complexity(),
            'network_complexity': self._compute_network_complexity(),
            'metrics_stability': self._compute_metrics_stability()
        }
        
    def _compute_pattern_complexity(self) -> float:
        """Compute complexity of visualized patterns"""
        if not self.pattern_viz.pattern_history:
            return 0.0
            
        complexities = []
        for history in self.pattern_viz.pattern_history:
            pattern = history['pattern']
            complexity = entropy(pattern / np.sum(pattern))
            complexities.append(complexity)
            
        return float(np.mean(complexities))
        
    def _compute_network_complexity(self) -> float:
        """Compute complexity of visualized networks"""
        if not self.network_viz.network_history:
            return 0.0
            
        complexities = []
        for history in self.network_viz.network_history:
            adjacency = history['adjacency']
            G = nx.from_numpy_array(adjacency)
            complexity = nx.density(G)
            complexities.append(complexity)
            
        return float(np.mean(complexities))
        
    def _compute_metrics_stability(self) -> float:
        """Compute stability of visualized metrics"""
        if not self.metrics_viz.metrics_history:
            return 0.0
            
        values = []
        for history in self.metrics_viz.metrics_history:
            metrics = history['metrics']
            values.append(list(metrics.values()))
            
        if not values:
            return 0.0
            
        values = np.array(values)
        return float(1.0 - np.std(values) / (np.mean(values) + 1e-7)) 