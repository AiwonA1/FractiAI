"""
Rate limiting module for FractiAI API

Implements sophisticated rate limiting with token bucket algorithm,
adaptive limits, and distributed rate limiting capabilities.
"""

from typing import Dict, Optional, Any
import asyncio
import time
import logging
from dataclasses import dataclass
from enum import Enum
import redis.asyncio as redis
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class LimitType(str, Enum):
    """Types of rate limits"""
    REQUESTS = "requests"
    BANDWIDTH = "bandwidth"
    COMPUTE = "compute"
    CONCURRENT = "concurrent"

@dataclass
class RateLimit:
    """Rate limit configuration"""
    limit: float
    window: float
    cost: float = 1.0
    burst_size: Optional[float] = None
    
    def __post_init__(self):
        """Set default burst size if not provided"""
        if self.burst_size is None:
            self.burst_size = self.limit

class TokenBucket:
    """Token bucket for rate limiting"""
    
    def __init__(
        self,
        rate_limit: RateLimit,
        initial_tokens: Optional[float] = None
    ):
        self.rate = rate_limit.limit / rate_limit.window
        self.capacity = rate_limit.burst_size
        self.tokens = initial_tokens if initial_tokens is not None else self.capacity
        self.last_update = time.time()
        
    def update(self) -> None:
        """Update token count based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.rate
        )
        self.last_update = now
        
    def consume(self, tokens: float = 1.0) -> bool:
        """Attempt to consume tokens"""
        self.update()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

class RateLimiter:
    """Rate limiter with distributed capabilities"""
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_window: float = 60.0,
        default_limit: float = 100.0
    ):
        self._redis = redis.from_url(redis_url) if redis_url else None
        self._local_buckets: Dict[str, Dict[LimitType, TokenBucket]] = {}
        self._default_limits = {
            LimitType.REQUESTS: RateLimit(
                limit=default_limit,
                window=default_window
            ),
            LimitType.BANDWIDTH: RateLimit(
                limit=1024 * 1024,  # 1MB/s
                window=1.0
            ),
            LimitType.COMPUTE: RateLimit(
                limit=1000,  # Compute units
                window=60.0
            ),
            LimitType.CONCURRENT: RateLimit(
                limit=10,
                window=1.0
            )
        }
        
    async def check_limit(
        self,
        key: str,
        limit_type: LimitType = LimitType.REQUESTS,
        cost: float = 1.0
    ) -> bool:
        """Check if operation is within rate limit"""
        if self._redis:
            return await self._check_distributed_limit(key, limit_type, cost)
        return self._check_local_limit(key, limit_type, cost)
        
    def _check_local_limit(
        self,
        key: str,
        limit_type: LimitType,
        cost: float
    ) -> bool:
        """Check local rate limit"""
        # Initialize buckets for key if not exists
        if key not in self._local_buckets:
            self._local_buckets[key] = {}
            
        # Initialize bucket for limit type if not exists
        if limit_type not in self._local_buckets[key]:
            self._local_buckets[key][limit_type] = TokenBucket(
                self._default_limits[limit_type]
            )
            
        # Check limit
        return self._local_buckets[key][limit_type].consume(cost)
        
    async def _check_distributed_limit(
        self,
        key: str,
        limit_type: LimitType,
        cost: float
    ) -> bool:
        """Check distributed rate limit using Redis"""
        try:
            redis_key = f"ratelimit:{key}:{limit_type.value}"
            window = self._default_limits[limit_type].window
            limit = self._default_limits[limit_type].limit
            
            # Get current window and count
            now = time.time()
            current_window = int(now / window)
            
            # Update window and count atomically
            async with self._redis.pipeline() as pipe:
                # Watch the key for changes
                await pipe.watch(redis_key)
                
                # Get current data
                data = await pipe.hgetall(redis_key)
                stored_window = int(data.get(b'window', 0))
                count = float(data.get(b'count', 0))
                
                # Reset if in new window
                if current_window > stored_window:
                    count = 0
                    stored_window = current_window
                    
                # Check if would exceed limit
                if count + cost > limit:
                    return False
                    
                # Update values
                pipe.multi()
                pipe.hset(
                    redis_key,
                    mapping={
                        'window': stored_window,
                        'count': count + cost
                    }
                )
                pipe.expire(redis_key, int(window * 2))
                
                # Execute transaction
                await pipe.execute()
                return True
                
        except redis.WatchError:
            # Key modified, retry
            return await self._check_distributed_limit(key, limit_type, cost)
        except Exception as e:
            logger.error(f"Redis error: {str(e)}")
            # Fallback to local limit
            return self._check_local_limit(key, limit_type, cost)
            
    def set_limit(
        self,
        key: str,
        limit_type: LimitType,
        rate_limit: RateLimit
    ) -> None:
        """Set custom rate limit for key and type"""
        if key not in self._local_buckets:
            self._local_buckets[key] = {}
            
        self._local_buckets[key][limit_type] = TokenBucket(rate_limit)
        
    async def reset_limit(
        self,
        key: str,
        limit_type: Optional[LimitType] = None
    ) -> None:
        """Reset rate limit for key"""
        if limit_type:
            # Reset specific limit type
            if key in self._local_buckets and limit_type in self._local_buckets[key]:
                self._local_buckets[key][limit_type] = TokenBucket(
                    self._default_limits[limit_type]
                )
                
            if self._redis:
                await self._redis.delete(f"ratelimit:{key}:{limit_type.value}")
        else:
            # Reset all limit types
            if key in self._local_buckets:
                del self._local_buckets[key]
                
            if self._redis:
                pattern = f"ratelimit:{key}:*"
                keys = await self._redis.keys(pattern)
                if keys:
                    await self._redis.delete(*keys)

class AdaptiveRateLimiter(RateLimiter):
    """Rate limiter with adaptive limits based on system load"""
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_window: float = 60.0,
        default_limit: float = 100.0,
        adaptation_interval: float = 5.0,
        load_threshold: float = 0.8
    ):
        super().__init__(redis_url, default_window, default_limit)
        self.adaptation_interval = adaptation_interval
        self.load_threshold = load_threshold
        self._load_history: Dict[str, list] = {}
        self._start_load_monitoring()
        
    def _start_load_monitoring(self) -> None:
        """Start background task for load monitoring"""
        async def monitor_load():
            while True:
                try:
                    await self._adapt_limits()
                    await asyncio.sleep(self.adaptation_interval)
                except Exception as e:
                    logger.error(f"Load monitoring error: {str(e)}")
                    await asyncio.sleep(1)
                    
        asyncio.create_task(monitor_load())
        
    async def _adapt_limits(self) -> None:
        """Adapt rate limits based on system load"""
        try:
            # Get current system load
            load = await self._get_system_load()
            
            # Store load history
            now = time.time()
            self._load_history[str(now)] = load
            
            # Remove old history
            cutoff = now - 300  # Keep 5 minutes of history
            self._load_history = {
                k: v for k, v in self._load_history.items()
                if float(k) > cutoff
            }
            
            # Compute average load
            avg_load = sum(self._load_history.values()) / len(self._load_history)
            
            # Adjust limits based on load
            if avg_load > self.load_threshold:
                # Reduce limits
                self._adjust_limits(0.8)
            elif avg_load < self.load_threshold * 0.5:
                # Increase limits
                self._adjust_limits(1.2)
                
        except Exception as e:
            logger.error(f"Error adapting limits: {str(e)}")
            
    def _adjust_limits(self, factor: float) -> None:
        """Adjust all rate limits by factor"""
        for buckets in self._local_buckets.values():
            for bucket in buckets.values():
                bucket.rate *= factor
                bucket.capacity *= factor
                
    async def _get_system_load(self) -> float:
        """Get current system load"""
        # TODO: Implement system load monitoring
        return 0.5  # Placeholder

# Initialize rate limiter
rate_limiter = AdaptiveRateLimiter() 