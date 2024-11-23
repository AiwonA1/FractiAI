"""
main.py

FastAPI implementation of the FractiAI API endpoints.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import numpy as np
import logging
from pydantic import BaseModel

from Methods.Method_Integrator import MethodIntegrator, UnipixelConfig
from Methods.FractiAnalytics import FractiAnalytics, AnalyticsConfig
from Methods.FractiMetrics import FractiMetrics, MetricsConfig

logger = logging.getLogger(__name__)

app = FastAPI(
    title="FractiAI API",
    description="API for FractiAI's fractal intelligence system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core components
integrator = MethodIntegrator()
analytics = FractiAnalytics(AnalyticsConfig())
metrics = FractiMetrics(MetricsConfig())

class ProcessRequest(BaseModel):
    data: List[float]
    config: Dict[str, any]

class AnalyzeRequest(BaseModel):
    patterns: List[List[float]]
    config: Optional[Dict[str, any]]

class OptimizeRequest(BaseModel):
    parameters: Dict[str, any]
    objective: str
    constraints: Optional[List[Dict]]

@app.post("/process")
async def process_data(request: ProcessRequest, background_tasks: BackgroundTasks):
    """Process input through FractiAI system"""
    try:
        # Convert input to numpy array
        input_data = np.array(request.data)
        
        # Process through integrator
        output = integrator.process_input(input_data)
        
        # Update metrics in background
        background_tasks.add_task(
            metrics.update_component_metrics,
            "processor",
            {"efficiency": float(np.mean(output))}
        )
        
        return {
            "result": output.tolist(),
            "metrics": {
                "coherence": float(np.mean(np.abs(np.corrcoef(input_data, output)))),
                "efficiency": float(np.mean(output))
            }
        }
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_patterns(request: AnalyzeRequest):
    """Analyze patterns using FractiScope"""
    try:
        # Convert patterns to numpy arrays
        patterns = [np.array(p) for p in request.patterns]
        
        # Analyze patterns
        analysis = analytics.analyze_patterns({
            f"pattern_{i}": p for i, p in enumerate(patterns)
        })
        
        return {
            "fractal_metrics": analysis['global_patterns'],
            "patterns": analysis['dimensional_analysis']
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize")
async def optimize_system(request: OptimizeRequest):
    """Optimize system parameters"""
    try:
        # Create objective function
        def objective_fn(x):
            return -np.sum(x**2)  # Example objective
            
        # Convert parameters to numpy array
        initial_state = np.array(list(request.parameters.values()))
        
        # Optimize
        optimized_state, score = integrator.optimize_system(
            {"main": initial_state},
            {"main": objective_fn}
        )
        
        return {
            "optimal_parameters": {
                k: float(v) for k, v in zip(request.parameters.keys(), 
                                          optimized_state['main'])
            },
            "objective_value": float(score),
            "convergence": {
                "iterations": 100,
                "final_error": float(abs(score))
            }
        }
        
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics(component: Optional[str] = None):
    """Get system metrics"""
    try:
        if component:
            return metrics.get_component_metrics(component)
        else:
            return metrics.get_system_metrics()
            
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    config = UnipixelConfig()
    integrator.initialize_system(config)
    logger.info("FractiAI API initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("FractiAI API shutting down") 