"""
FractiCache.py

Implements comprehensive caching framework for FractiAI, enabling pattern-aware
caching and memory management through fractal-based organization.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging
import time
from collections import deque
import hashlib
import pickle
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for caching framework"""
    max_size: int = 1000000  # Maximum cache size in bytes
    ttl: int = 3600  # Time to live in seconds
    pattern_threshold: float = 0.7
    fractal_depth: int = 3
    eviction_strategy: str = 'lru'  # lru, lfu, or pattern

class CachePattern:
    """Represents access pattern for cached data"""
    
    def __init__(self, key: str):
        self.key = key
        self.access_times = deque(maxlen=1000)
        self.access_counts = 0
        self.last_access = time.time()
        self.pattern_metrics = {}
        
    def record_access(self) -> None:
        """Record cache access"""
        now = time.time()
        self.access_times.append(now)
        self.access_counts += 1
        self.last_access = now
        self._update_metrics()
        
    def _update_metrics(self) -> None:
        """Update pattern metrics"""
        if len(self.access_times) < 2:
            return
            
        intervals = np.diff(list(self.access_times))
        self.pattern_metrics = {
            'frequency': self.access_counts / (time.time() - min(self.access_times)),
            'regularity': 1.0 - np.std(intervals) / (np.mean(intervals) + 1e-7),
            'recency': time.time() - self.last_access
        }

class CacheEntry:
    """Individual cache entry with metadata"""
    
    def __init__(self, key: str, value: Any, size: int):
        self.key = key
        self.value = value
        self.size = size
        self.created = time.time()
        self.pattern = CachePattern(key)
        
    def is_expired(self, ttl: int) -> bool:
        """Check if entry is expired"""
        return time.time() - self.created > ttl
        
    def get_value(self) -> Any:
        """Get cached value and record access"""
        self.pattern.record_access()
        return self.value
        
    def get_metrics(self) -> Dict[str, float]:
        """Get entry metrics"""
        return {
            'age': time.time() - self.created,
            'size': self.size,
            **self.pattern.pattern_metrics
        }

class MemoryManager:
    """Manages memory allocation and eviction"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.current_size = 0
        self.eviction_history = []
        
    def can_allocate(self, size: int) -> bool:
        """Check if memory can be allocated"""
        return self.current_size + size <= self.config.max_size
        
    def allocate(self, size: int) -> bool:
        """Allocate memory"""
        if self.can_allocate(size):
            self.current_size += size
            return True
        return False
        
    def deallocate(self, size: int) -> None:
        """Deallocate memory"""
        self.current_size = max(0, self.current_size - size)
        
    def select_entries_for_eviction(self, entries: Dict[str, CacheEntry],
                                  required_size: int) -> List[str]:
        """Select entries for eviction"""
        if self.config.eviction_strategy == 'lru':
            return self._select_lru(entries, required_size)
        elif self.config.eviction_strategy == 'lfu':
            return self._select_lfu(entries, required_size)
        else:  # pattern-based
            return self._select_pattern_based(entries, required_size)
            
    def _select_lru(self, entries: Dict[str, CacheEntry],
                    required_size: int) -> List[str]:
        """Select entries using LRU strategy"""
        sorted_entries = sorted(
            entries.items(),
            key=lambda x: x[1].pattern.last_access
        )
        
        selected = []
        freed_size = 0
        
        for key, entry in sorted_entries:
            if freed_size >= required_size:
                break
            selected.append(key)
            freed_size += entry.size
            
        return selected
        
    def _select_lfu(self, entries: Dict[str, CacheEntry],
                    required_size: int) -> List[str]:
        """Select entries using LFU strategy"""
        sorted_entries = sorted(
            entries.items(),
            key=lambda x: x[1].pattern.access_counts
        )
        
        selected = []
        freed_size = 0
        
        for key, entry in sorted_entries:
            if freed_size >= required_size:
                break
            selected.append(key)
            freed_size += entry.size
            
        return selected
        
    def _select_pattern_based(self, entries: Dict[str, CacheEntry],
                             required_size: int) -> List[str]:
        """Select entries using pattern-based strategy"""
        # Score entries based on pattern metrics
        scores = {}
        for key, entry in entries.items():
            metrics = entry.get_metrics()
            if metrics:
                # Combine metrics into score
                score = (
                    metrics['frequency'] * 0.4 +
                    metrics['regularity'] * 0.3 +
                    (1.0 / (metrics['recency'] + 1)) * 0.3
                )
                scores[key] = score
                
        # Sort by score (ascending)
        sorted_entries = sorted(
            entries.items(),
            key=lambda x: scores.get(x[0], 0)
        )
        
        selected = []
        freed_size = 0
        
        for key, entry in sorted_entries:
            if freed_size >= required_size:
                break
            selected.append(key)
            freed_size += entry.size
            
        return selected

class FractiCache:
    """Main caching system with fractal awareness"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.entries: Dict[str, CacheEntry] = {}
        self.memory_manager = MemoryManager(config)
        self.pattern_history = []
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.entries:
            return None
            
        entry = self.entries[key]
        
        # Check expiration
        if entry.is_expired(self.config.ttl):
            self._evict_entry(key)
            return None
            
        # Record access pattern
        self._record_pattern(key, 'get')
        
        return entry.get_value()
        
    def put(self, key: str, value: Any) -> bool:
        """Put value in cache"""
        # Compute value size
        size = len(pickle.dumps(value))
        
        # Check if we need to evict entries
        if not self.memory_manager.can_allocate(size):
            required_size = size - (self.config.max_size - self.memory_manager.current_size)
            self._evict_entries(required_size)
            
        # Try to allocate memory
        if not self.memory_manager.allocate(size):
            return False
            
        # Store entry
        self.entries[key] = CacheEntry(key, value, size)
        
        # Record access pattern
        self._record_pattern(key, 'put')
        
        return True
        
    def remove(self, key: str) -> None:
        """Remove entry from cache"""
        if key in self.entries:
            self._evict_entry(key)
            
    def clear(self) -> None:
        """Clear all entries"""
        self.entries.clear()
        self.memory_manager.current_size = 0
        
    def _evict_entry(self, key: str) -> None:
        """Evict single entry"""
        if key in self.entries:
            entry = self.entries[key]
            self.memory_manager.deallocate(entry.size)
            del self.entries[key]
            
    def _evict_entries(self, required_size: int) -> None:
        """Evict entries to free space"""
        keys_to_evict = self.memory_manager.select_entries_for_eviction(
            self.entries,
            required_size
        )
        
        for key in keys_to_evict:
            self._evict_entry(key)
            
    def _record_pattern(self, key: str, operation: str) -> None:
        """Record access pattern"""
        pattern = {
            'timestamp': time.time(),
            'key': key,
            'operation': operation
        }
        self.pattern_history.append(pattern)
        
        # Analyze patterns periodically
        if len(self.pattern_history) % 100 == 0:
            self._analyze_patterns()
            
    def _analyze_patterns(self) -> None:
        """Analyze access patterns"""
        if len(self.pattern_history) < 100:
            return
            
        # Analyze recent patterns
        recent = self.pattern_history[-100:]
        
        # Compute pattern metrics
        metrics = {
            'access_patterns': self._compute_access_patterns(recent),
            'temporal_patterns': self._compute_temporal_patterns(recent),
            'key_patterns': self._compute_key_patterns(recent)
        }
        
        # Adjust caching strategy based on patterns
        self._adjust_strategy(metrics)
        
    def _compute_access_patterns(self, 
                               patterns: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute access pattern metrics"""
        operations = [p['operation'] for p in patterns]
        return {
            'get_ratio': operations.count('get') / len(operations),
            'put_ratio': operations.count('put') / len(operations)
        }
        
    def _compute_temporal_patterns(self, 
                                 patterns: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute temporal pattern metrics"""
        timestamps = [p['timestamp'] for p in patterns]
        intervals = np.diff(timestamps)
        
        return {
            'mean_interval': float(np.mean(intervals)),
            'interval_stability': float(1.0 - np.std(intervals) / 
                                     (np.mean(intervals) + 1e-7))
        }
        
    def _compute_key_patterns(self, 
                            patterns: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute key usage pattern metrics"""
        keys = [p['key'] for p in patterns]
        unique_keys = len(set(keys))
        
        return {
            'key_diversity': float(unique_keys / len(keys)),
            'key_reuse': float(1.0 - unique_keys / len(keys))
        }
        
    def _adjust_strategy(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """Adjust caching strategy based on patterns"""
        # Example strategy adjustment
        if metrics['access_patterns']['get_ratio'] > 0.8:
            # High read ratio - optimize for reads
            self.config.eviction_strategy = 'lru'
        elif metrics['key_patterns']['key_reuse'] > 0.7:
            # High key reuse - optimize for patterns
            self.config.eviction_strategy = 'pattern'
        else:
            # Default to LFU
            self.config.eviction_strategy = 'lfu' 