"""
websocket.py

WebSocket implementation for real-time FractiAI data streaming.
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set, Any, Optional
import json
import asyncio
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class StreamConfig:
    """Configuration for data streaming"""
    batch_size: int = 100
    update_interval: float = 0.1
    compression_level: int = 1

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """Accept new connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = set()
        
    def disconnect(self, client_id: str) -> None:
        """Handle client disconnection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
            
    async def broadcast(self, message: Dict) -> None:
        """Broadcast message to all connections"""
        disconnected = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except WebSocketDisconnect:
                disconnected.append(client_id)
                
        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)
            
    async def send_personal_message(self, message: Dict, client_id: str) -> None:
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except WebSocketDisconnect:
                self.disconnect(client_id)
                
    def subscribe(self, client_id: str, channel: str) -> None:
        """Subscribe client to channel"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].add(channel)
            
    def unsubscribe(self, client_id: str, channel: str) -> None:
        """Unsubscribe client from channel"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].discard(channel)
            
    async def broadcast_to_channel(self, channel: str, message: Dict) -> None:
        """Broadcast message to channel subscribers"""
        for client_id, subs in self.subscriptions.items():
            if channel in subs:
                await self.send_personal_message(message, client_id)

class DataStreamer:
    """Handles real-time data streaming"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.manager = ConnectionManager()
        self.data_processors: Dict[str, Any] = {}
        self._running = False
        
    async def start(self) -> None:
        """Start data streaming"""
        self._running = True
        asyncio.create_task(self._stream_loop())
        
    async def stop(self) -> None:
        """Stop data streaming"""
        self._running = False
        
    async def _stream_loop(self) -> None:
        """Main streaming loop"""
        while self._running:
            try:
                # Process and stream data
                for processor_id, processor in self.data_processors.items():
                    data = await processor.get_data()
                    if data:
                        await self.manager.broadcast_to_channel(
                            processor_id,
                            self._format_data(data)
                        )
                        
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                logger.error(f"Streaming error: {str(e)}")
                await asyncio.sleep(1)  # Error cooldown
                
    def _format_data(self, data: Any) -> Dict:
        """Format data for streaming"""
        if hasattr(data, '__dict__'):
            data = asdict(data)
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'data': data
        }
        
    def add_processor(self, processor_id: str, processor: Any) -> None:
        """Add data processor"""
        self.data_processors[processor_id] = processor
        
    def remove_processor(self, processor_id: str) -> None:
        """Remove data processor"""
        if processor_id in self.data_processors:
            del self.data_processors[processor_id]

# Initialize streamer
streamer = DataStreamer(StreamConfig()) 