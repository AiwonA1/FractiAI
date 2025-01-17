"""
Main FastAPI application for FractiAI

Implements comprehensive REST and WebSocket API endpoints for
interacting with the FractiAI system.
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any
import uuid
import logging
from datetime import datetime

from .auth import AuthManager, User
from .rate_limiter import RateLimiter
from .websocket import ws_manager, MessageType
from .monitoring import MetricsCollector

# Initialize FastAPI app
app = FastAPI(
    title="FractiAI API",
    description="API for interacting with FractiAI system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
auth_manager = AuthManager()
rate_limiter = RateLimiter()
metrics_collector = MetricsCollector()

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Dependency for getting current user
async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get current authenticated user"""
    user = await auth_manager.get_current_user(token)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials"
        )
    return user

# Rate limiting dependency
async def check_rate_limit(user: User = Depends(get_current_user)):
    """Check rate limit for current user"""
    if not await rate_limiter.check_limit(user.username):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded"
        )

@app.post("/simulations")
async def create_simulation(
    config: Dict[str, Any],
    user: User = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Create new simulation"""
    try:
        # Generate simulation ID
        simulation_id = str(uuid.uuid4())
        
        # Initialize simulation
        # TODO: Implement simulation initialization
        
        return {
            "id": simulation_id,
            "status": "initializing",
            "created_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@app.get("/simulations")
async def list_simulations(
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    user: User = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """List simulations"""
    try:
        # Get simulations
        # TODO: Implement simulation listing
        
        return {
            "items": [],
            "total": 0
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@app.get("/simulations/{simulation_id}")
async def get_simulation(
    simulation_id: str,
    user: User = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Get simulation details"""
    try:
        # Get simulation
        # TODO: Implement simulation retrieval
        
        return {
            "id": simulation_id,
            "status": "running",
            "current_step": 0
        }
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

@app.delete("/simulations/{simulation_id}")
async def delete_simulation(
    simulation_id: str,
    user: User = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Delete simulation"""
    try:
        # Delete simulation
        # TODO: Implement simulation deletion
        
        return None
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

@app.post("/simulations/{simulation_id}/control")
async def control_simulation(
    simulation_id: str,
    action: Dict[str, Any],
    user: User = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Control simulation"""
    try:
        # Control simulation
        # TODO: Implement simulation control
        
        return {
            "id": simulation_id,
            "status": action["action"],
            "current_step": 0
        }
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

@app.post("/analysis")
async def request_analysis(
    request: Dict[str, Any],
    user: User = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Request analysis"""
    try:
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Start analysis
        # TODO: Implement analysis
        
        return {
            "id": analysis_id,
            "simulation_id": request["simulation_id"],
            "analysis_type": request["analysis_type"],
            "created_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@app.get("/analysis/{analysis_id}")
async def get_analysis(
    analysis_id: str,
    user: User = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Get analysis results"""
    try:
        # Get analysis results
        # TODO: Implement analysis retrieval
        
        return {
            "id": analysis_id,
            "results": {}
        }
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

@app.post("/optimize")
async def request_optimization(
    request: Dict[str, Any],
    user: User = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Request optimization"""
    try:
        # Generate optimization ID
        optimization_id = str(uuid.uuid4())
        
        # Start optimization
        # TODO: Implement optimization
        
        return {
            "id": optimization_id,
            "simulation_id": request["simulation_id"],
            "status": "running",
            "created_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@app.get("/optimize/{optimization_id}")
async def get_optimization(
    optimization_id: str,
    user: User = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Get optimization results"""
    try:
        # Get optimization results
        # TODO: Implement optimization retrieval
        
        return {
            "id": optimization_id,
            "status": "completed",
            "parameters": {},
            "objective_values": {}
        }
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

@app.websocket("/ws/{simulation_id}")
async def websocket_endpoint(websocket: WebSocket, simulation_id: str):
    """WebSocket endpoint for real-time updates"""
    try:
        # Connect to WebSocket
        await ws_manager.connect(websocket, simulation_id)
        
        try:
            while True:
                # Receive message
                data = await websocket.receive_text()
                
                # Handle message
                await ws_manager.handle_client_message(websocket, data)
                
        except WebSocketDisconnect:
            # Disconnect on WebSocket disconnect
            await ws_manager.disconnect(websocket, simulation_id)
            
    except Exception as e:
        logging.error(f"WebSocket error: {str(e)}")
        if websocket.client_state.connected:
            await websocket.close()

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    # Start metrics collection
    await metrics_collector.start()

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    # Stop metrics collection
    await metrics_collector.stop() 