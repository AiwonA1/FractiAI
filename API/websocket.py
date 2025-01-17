"""
WebSocket manager for FractiAI API

Implements sophisticated real-time communication with support for
channels, pub/sub patterns, and message queuing.
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set, Optional, Any, List
import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

class MessageType(str, Enum):
    """Types of WebSocket messages"""
    STATE = "state"
    METRIC = "metric"
    ALERT = "alert"
    CONTROL = "control"
    SYSTEM = "system"

@dataclass
class Message:
    """WebSocket message structure"""
    type: MessageType
    data: Any
    timestamp: datetime = None
    source: str = None
    target: Optional[str] = None
    
    def to_json(self) -> str:
        """Convert message to JSON string"""
        return json.dumps({
            'type': self.type,
            'data': self.data,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'source': self.source,
            'target': self.target
        })
        
    @classmethod
    def from_json(cls, data: str) -> 'Message':
        """Create message from JSON string"""
        parsed = json.loads(data)
        return cls(
            type=MessageType(parsed['type']),
            data=parsed['data'],
            timestamp=datetime.fromisoformat(parsed['timestamp'])
            if parsed.get('timestamp') else None,
            source=parsed.get('source'),
            target=parsed.get('target')
        )

class Channel:
    """Represents a communication channel"""
    
    def __init__(self, name: str):
        self.name = name
        self.connections: Set[WebSocket] = set()
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.last_message: Optional[Message] = None
        
    async def connect(self, websocket: WebSocket) -> None:
        """Add connection to channel"""
        await websocket.accept()
        self.connections.add(websocket)
        
        # Send last message if available
        if self.last_message:
            await websocket.send_text(self.last_message.to_json())
            
    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove connection from channel"""
        self.connections.remove(websocket)
        
    async def broadcast(self, message: Message) -> None:
        """Broadcast message to all connections"""
        self.last_message = message
        
        dead_connections = set()
        for connection in self.connections:
            try:
                await connection.send_text(message.to_json())
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {str(e)}")
                dead_connections.add(connection)
                
        # Clean up dead connections
        for connection in dead_connections:
            await self.disconnect(connection)
            
    async def queue_message(self, message: Message) -> None:
        """Queue message for processing"""
        await self.message_queue.put(message)

class WebSocketManager:
    """Manages WebSocket connections and communication"""
    
    def __init__(self):
        self._channels: Dict[str, Channel] = {}
        self._simulation_channels: Dict[str, Set[str]] = defaultdict(set)
        self._connection_channels: Dict[WebSocket, Set[str]] = defaultdict(set)
        self._message_handlers: Dict[MessageType, callable] = {}
        
        # Start message processing
        self._start_message_processor()
        
    async def connect(self, websocket: WebSocket, simulation_id: str) -> None:
        """Connect websocket to simulation channels"""
        # Create default channels if they don't exist
        default_channels = [
            f"{simulation_id}:state",
            f"{simulation_id}:metrics",
            f"{simulation_id}:alerts"
        ]
        
        for channel_name in default_channels:
            if channel_name not in self._channels:
                self._channels[channel_name] = Channel(channel_name)
            self._simulation_channels[simulation_id].add(channel_name)
            
            # Connect to channel
            channel = self._channels[channel_name]
            await channel.connect(websocket)
            self._connection_channels[websocket].add(channel_name)
            
    async def disconnect(self, websocket: WebSocket, simulation_id: str) -> None:
        """Disconnect websocket from all channels"""
        # Get channels for this connection
        channels = self._connection_channels.get(websocket, set())
        
        # Disconnect from each channel
        for channel_name in channels:
            if channel_name in self._channels:
                await self._channels[channel_name].disconnect(websocket)
                
        # Clean up
        if websocket in self._connection_channels:
            del self._connection_channels[websocket]
            
    async def broadcast_to_simulation(
        self,
        simulation_id: str,
        data: Any,
        message_type: MessageType = MessageType.STATE
    ) -> None:
        """Broadcast message to all simulation channels"""
        channels = self._simulation_channels.get(simulation_id, set())
        
        message = Message(
            type=message_type,
            data=data,
            timestamp=datetime.utcnow(),
            source=simulation_id
        )
        
        for channel_name in channels:
            if channel_name in self._channels:
                await self._channels[channel_name].broadcast(message)
                
    async def send_to_channel(
        self,
        channel_name: str,
        data: Any,
        message_type: MessageType = MessageType.STATE
    ) -> None:
        """Send message to specific channel"""
        if channel_name not in self._channels:
            return
            
        message = Message(
            type=message_type,
            data=data,
            timestamp=datetime.utcnow()
        )
        
        await self._channels[channel_name].broadcast(message)
        
    def register_handler(
        self,
        message_type: MessageType,
        handler: callable
    ) -> None:
        """Register message handler for type"""
        self._message_handlers[message_type] = handler
        
    def _start_message_processor(self) -> None:
        """Start background task for processing messages"""
        async def process_messages():
            while True:
                try:
                    # Process messages from all channels
                    for channel in self._channels.values():
                        while not channel.message_queue.empty():
                            message = await channel.message_queue.get()
                            
                            # Handle message
                            if message.type in self._message_handlers:
                                try:
                                    await self._message_handlers[message.type](message)
                                except Exception as e:
                                    logger.error(
                                        f"Error handling message: {str(e)}"
                                    )
                                    
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error in message processor: {str(e)}")
                    await asyncio.sleep(1)
                    
        asyncio.create_task(process_messages())
        
    async def handle_client_message(
        self,
        websocket: WebSocket,
        message: str
    ) -> None:
        """Handle message from client"""
        try:
            # Parse message
            parsed_message = Message.from_json(message)
            
            # Queue for processing if handler exists
            if parsed_message.type in self._message_handlers:
                channels = self._connection_channels.get(websocket, set())
                if channels:
                    # Use first channel (could be more sophisticated)
                    channel_name = next(iter(channels))
                    await self._channels[channel_name].queue_message(parsed_message)
                    
        except Exception as e:
            logger.error(f"Error handling client message: {str(e)}")
            
    async def create_channel(self, name: str) -> Channel:
        """Create new communication channel"""
        if name not in self._channels:
            self._channels[name] = Channel(name)
        return self._channels[name]
        
    async def delete_channel(self, name: str) -> None:
        """Delete communication channel"""
        if name in self._channels:
            channel = self._channels[name]
            
            # Disconnect all connections
            for connection in list(channel.connections):
                await channel.disconnect(connection)
                
            # Remove from simulations
            for simulation_channels in self._simulation_channels.values():
                simulation_channels.discard(name)
                
            # Remove channel
            del self._channels[name]
            
    async def get_channel_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all channels"""
        return {
            name: {
                'connections': len(channel.connections),
                'queue_size': channel.message_queue.qsize(),
                'last_message_time': channel.last_message.timestamp.isoformat()
                if channel.last_message else None
            }
            for name, channel in self._channels.items()
        }

# Initialize WebSocket manager
ws_manager = WebSocketManager() 