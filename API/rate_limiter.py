"""
rate_limiter.py

Rate limiting system for FractiAI API to control request rates and ensure fair usage.
"""

from fastapi import HTTPException, Request
from typing import Dict, Optional, Tuple
import time
import logging
from dataclasses import dataclass
from collections import defaultdict
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_minute: int = 60
    burst_size: int = 10
    cleanup_interval: int = 300  # 5 minutes
    penalty_duration: int = 3600  # 1 hour

class TokenBucket:
    """Token bucket rate limiter implementation"""
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from bucket"""
        now = time.time()
        
        # Add new tokens based on time passed
        time_passed = now - self.last_update
        self.tokens = min(
            self.capacity,
            self.tokens + time_passed * self.rate
        )
        self.last_update = now
        
        # Try to consume tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

class RateLimiter:
    """Rate limiting system with penalties and quotas"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.limiters: Dict[str, TokenBucket] = {}
        self.penalties: Dict[str, float] = {}
        self.usage_stats: Dict[str, Dict] = defaultdict(lambda: {
            'total_requests': 0,
            'blocked_requests': 0,
            'last_request': 0.0
        })
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_loop())
        
    async def check_rate_limit(self, request: Request) -> bool:
        """Check if request should be rate limited"""
        client_id = self._get_client_id(request)
        
        # Check for penalties
        if self._is_penalized(client_id):
            self.usage_stats[client_id]['blocked_requests'] += 1
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Penalty in effect."
            )
            
        # Get or create limiter
        limiter = self._get_limiter(client_id)
        
        # Try to consume token
        if not limiter.consume():
            self._apply_penalty(client_id)
            self.usage_stats[client_id]['blocked_requests'] += 1
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )
            
        # Update stats
        self.usage_stats[client_id]['total_requests'] += 1
        self.usage_stats[client_id]['last_request'] = time.time()
        
        return True
        
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request"""
        # Try to get API key first
        api_key = request.headers.get('X-API-Key')
        if api_key:
            return f"api_{api_key}"
            
        # Fall back to IP address
        forwarded = request.headers.get('X-Forwarded-For')
        if forwarded:
            return f"ip_{forwarded.split(',')[0]}"
        return f"ip_{request.client.host}"
        
    def _get_limiter(self, client_id: str) -> TokenBucket:
        """Get or create token bucket for client"""
        if client_id not in self.limiters:
            self.limiters[client_id] = TokenBucket(
                self.config.requests_per_minute / 60.0,
                self.config.burst_size
            )
        return self.limiters[client_id]
        
    def _is_penalized(self, client_id: str) -> bool:
        """Check if client is currently penalized"""
        if client_id in self.penalties:
            if time.time() < self.penalties[client_id]:
                return True
            del self.penalties[client_id]
        return False
        
    def _apply_penalty(self, client_id: str) -> None:
        """Apply rate limit penalty to client"""
        self.penalties[client_id] = time.time() + self.config.penalty_duration
        logger.warning(f"Applied rate limit penalty to {client_id}")
        
    async def _cleanup_loop(self) -> None:
        """Periodically clean up old entries"""
        while True:
            await asyncio.sleep(self.config.cleanup_interval)
            self._cleanup()
            
    def _cleanup(self) -> None:
        """Clean up old entries"""
        now = time.time()
        
        # Clean up penalties
        expired_penalties = [
            client_id for client_id, expire_time in self.penalties.items()
            if now >= expire_time
        ]
        for client_id in expired_penalties:
            del self.penalties[client_id]
            
        # Clean up inactive limiters
        inactive_threshold = now - self.config.cleanup_interval
        inactive_limiters = [
            client_id for client_id in self.limiters
            if self.usage_stats[client_id]['last_request'] < inactive_threshold
        ]
        for client_id in inactive_limiters:
            del self.limiters[client_id]
            del self.usage_stats[client_id]
            
    def get_usage_stats(self, client_id: Optional[str] = None) -> Dict:
        """Get usage statistics"""
        if client_id:
            return self.usage_stats[client_id]
        return {
            client_id: stats.copy()
            for client_id, stats in self.usage_stats.items()
        }

# Initialize rate limiter
rate_limiter = RateLimiter(RateLimitConfig()) 