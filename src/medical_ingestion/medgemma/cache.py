# ============================================================================
# src/medgemma/cache.py
# ============================================================================
"""
Advanced Caching System for MedGemma Responses

Features:
- TTL (time-to-live) support for cache expiration
- LRU eviction when size limits are reached
- Thread-safe operations
- Multiple backend support (in-memory, file-based)
- Cache statistics and monitoring
- Configurable serialization
- Cache warming and preloading

This provides a production-ready caching layer for expensive LLM inference operations.
"""

import json
import pickle
import hashlib
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import OrderedDict
from dataclasses import dataclass
import logging


@dataclass
class CacheEntry:
    """
    Single cache entry with metadata.

    Attributes:
        key: Cache key (hash)
        value: Cached response data
        created_at: When the entry was created
        last_accessed: When the entry was last read
        access_count: Number of times accessed
        size_bytes: Approximate size in bytes
        ttl_seconds: Time-to-live (None = no expiration)
    """
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None

    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL"""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def mark_accessed(self):
        """Update access metadata"""
        self.last_accessed = datetime.now()
        self.access_count += 1


class CacheStatistics:
    """Track cache performance metrics"""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0
        self.writes = 0
        self.size_bytes = 0
        self.entry_count = 0

    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "writes": self.writes,
            "hit_rate": self.hit_rate(),
            "size_bytes": self.size_bytes,
            "entry_count": self.entry_count,
        }


class MedGemmaCache:
    """
    High-performance cache for MedGemma inference results.

    Features:
    - LRU eviction when max_size is reached
    - TTL-based expiration
    - Thread-safe operations
    - Persistent storage (optional)
    - Cache warming
    - Detailed statistics

    Example:
        cache = MedGemmaCache(
            max_size=1000,
            default_ttl=3600,  # 1 hour
            cache_dir=Path("./cache")
        )

        # Store result
        cache.set("prompt_key", {"text": "response", "tokens": 100})

        # Retrieve result
        result = cache.get("prompt_key")

        # Get statistics
        stats = cache.get_statistics()
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_size_bytes: Optional[int] = None,
        default_ttl: Optional[int] = None,
        cache_dir: Optional[Path] = None,
        auto_persist: bool = True,
        persist_interval: int = 10,  # Persist every N writes
    ):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of entries (LRU eviction when exceeded)
            max_size_bytes: Maximum total size in bytes (optional)
            default_ttl: Default TTL in seconds (None = no expiration)
            cache_dir: Directory for persistent storage (None = memory only)
            auto_persist: Automatically save to disk periodically
            persist_interval: Number of writes between auto-persist
        """
        self.max_size = max_size
        self.max_size_bytes = max_size_bytes
        self.default_ttl = default_ttl
        self.cache_dir = cache_dir
        self.auto_persist = auto_persist
        self.persist_interval = persist_interval

        # Thread-safe storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self._stats = CacheStatistics()

        # Write counter for auto-persist
        self._write_count = 0

        # Logger
        self.logger = logging.getLogger(__name__)

        # Load from disk if cache_dir exists
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.

        Args:
            key: Cache key (will be hashed if not already)
            default: Default value if not found

        Returns:
            Cached value or default
        """
        with self._lock:
            cache_key = self._hash_key(key)

            # Check if exists
            if cache_key not in self._cache:
                self._stats.misses += 1
                return default

            entry = self._cache[cache_key]

            # Check if expired
            if entry.is_expired():
                self.logger.debug(f"Cache entry expired: {cache_key}")
                self._remove_entry(cache_key)
                self._stats.misses += 1
                self._stats.expirations += 1
                return default

            # Update access metadata
            entry.mark_accessed()

            # Move to end (LRU)
            self._cache.move_to_end(cache_key)

            # Update statistics
            self._stats.hits += 1

            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store value in cache.

        Args:
            key: Cache key (will be hashed if not already)
            value: Value to cache
            ttl: TTL in seconds (overrides default_ttl)
        """
        with self._lock:
            cache_key = self._hash_key(key)

            # Calculate size
            size_bytes = self._estimate_size(value)

            # Check if we need to evict
            if cache_key not in self._cache:
                self._ensure_space(size_bytes)

            # Create entry
            entry = CacheEntry(
                key=cache_key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=0,
                size_bytes=size_bytes,
                ttl_seconds=ttl if ttl is not None else self.default_ttl,
            )

            # Update or add
            if cache_key in self._cache:
                old_size = self._cache[cache_key].size_bytes
                self._stats.size_bytes -= old_size

            self._cache[cache_key] = entry
            self._cache.move_to_end(cache_key)

            # Update statistics
            self._stats.writes += 1
            self._stats.size_bytes += size_bytes
            self._stats.entry_count = len(self._cache)

            # Auto-persist
            self._write_count += 1
            if self.auto_persist and self._write_count >= self.persist_interval:
                self._save_to_disk()
                self._write_count = 0

    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            cache_key = self._hash_key(key)
            if cache_key in self._cache:
                self._remove_entry(cache_key)
                return True
            return False

    def clear(self) -> None:
        """Clear all entries from cache"""
        with self._lock:
            self._cache.clear()
            self._stats.entry_count = 0
            self._stats.size_bytes = 0
            self.logger.info("Cache cleared")

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]

            for key in expired_keys:
                self._remove_entry(key)
                self._stats.expirations += 1

            if expired_keys:
                self.logger.info(f"Cleaned up {len(expired_keys)} expired entries")

            return len(expired_keys)

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            stats = self._stats.to_dict()
            stats["max_size"] = self.max_size
            stats["max_size_bytes"] = self.max_size_bytes
            stats["default_ttl"] = self.default_ttl
            return stats

    def get_entries_by_access(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most frequently accessed entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of entry metadata sorted by access count
        """
        with self._lock:
            sorted_entries = sorted(
                self._cache.values(),
                key=lambda e: e.access_count,
                reverse=True
            )[:limit]

            return [
                {
                    "key": entry.key,
                    "access_count": entry.access_count,
                    "created_at": entry.created_at.isoformat(),
                    "last_accessed": entry.last_accessed.isoformat(),
                    "size_bytes": entry.size_bytes,
                }
                for entry in sorted_entries
            ]

    def warm_cache(self, entries: Dict[str, Any]) -> int:
        """
        Preload cache with entries.

        Args:
            entries: Dictionary of key-value pairs to cache

        Returns:
            Number of entries added
        """
        count = 0
        for key, value in entries.items():
            self.set(key, value)
            count += 1

        self.logger.info(f"Warmed cache with {count} entries")
        return count

    def _ensure_space(self, required_bytes: int):
        """
        Ensure enough space for new entry by evicting old entries if needed.

        Args:
            required_bytes: Size of new entry
        """
        # Check entry count limit
        while len(self._cache) >= self.max_size:
            self._evict_oldest()

        # Check size limit
        if self.max_size_bytes:
            while (
                self._stats.size_bytes + required_bytes > self.max_size_bytes
                and len(self._cache) > 0
            ):
                self._evict_oldest()

    def _evict_oldest(self):
        """Evict least recently used entry"""
        if not self._cache:
            return

        # OrderedDict maintains insertion order, oldest is first
        oldest_key = next(iter(self._cache))
        self._remove_entry(oldest_key)
        self._stats.evictions += 1

        self.logger.debug(f"Evicted entry: {oldest_key}")

    def _remove_entry(self, key: str):
        """Remove entry and update statistics"""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._stats.size_bytes -= entry.size_bytes
            self._stats.entry_count = len(self._cache)

    def _hash_key(self, key: str) -> str:
        """
        Generate consistent hash for cache key.

        Args:
            key: Input key (can be prompt text, params, etc.)

        Returns:
            MD5 hash string
        """
        if isinstance(key, str):
            return hashlib.md5(key.encode()).hexdigest()
        else:
            # Handle non-string keys
            return hashlib.md5(str(key).encode()).hexdigest()

    def _estimate_size(self, value: Any) -> int:
        """
        Estimate size of value in bytes.

        This is approximate and used for cache size limits.
        """
        try:
            # Try pickle for accurate size
            return len(pickle.dumps(value))
        except:
            # Fallback to JSON
            try:
                return len(json.dumps(value).encode())
            except:
                # Rough estimate
                return 1000

    def _save_to_disk(self):
        """Persist cache to disk"""
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / "medgemma_cache.pkl"
        stats_file = self.cache_dir / "cache_stats.json"

        try:
            # Save cache entries
            with open(cache_file, 'wb') as f:
                pickle.dump(dict(self._cache), f)

            # Save statistics
            with open(stats_file, 'w') as f:
                json.dump(self._stats.to_dict(), f, indent=2)

            self.logger.debug(f"Cache persisted to {cache_file}")

        except Exception as e:
            self.logger.warning(f"Failed to persist cache: {e}")

    def _load_from_disk(self):
        """Load cache from disk"""
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / "medgemma_cache.pkl"

        if not cache_file.exists():
            return

        try:
            with open(cache_file, 'rb') as f:
                loaded_cache = pickle.load(f)

            # Filter out expired entries
            valid_entries = 0

            for key, entry in loaded_cache.items():
                if not entry.is_expired():
                    self._cache[key] = entry
                    self._stats.size_bytes += entry.size_bytes
                    valid_entries += 1

            self._stats.entry_count = len(self._cache)

            self.logger.info(
                f"Loaded {valid_entries} valid entries from cache "
                f"({len(loaded_cache) - valid_entries} expired)"
            )

        except Exception as e:
            self.logger.warning(f"Failed to load cache from disk: {e}")


class PromptCache(MedGemmaCache):
    """
    Specialized cache for MedGemma prompts.

    Automatically generates cache keys from prompt + parameters.
    """

    def get_response(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.1,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response for prompt + parameters.

        Args:
            prompt: Input prompt
            max_tokens: Max tokens parameter
            temperature: Temperature parameter

        Returns:
            Cached response or None
        """
        key = self._make_prompt_key(prompt, max_tokens, temperature)
        return self.get(key)

    def set_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        response: Dict[str, Any],
        ttl: Optional[int] = None,
    ):
        """
        Cache response for prompt + parameters.

        Args:
            prompt: Input prompt
            max_tokens: Max tokens parameter
            temperature: Temperature parameter
            response: Response to cache
            ttl: TTL in seconds
        """
        key = self._make_prompt_key(prompt, max_tokens, temperature)
        self.set(key, response, ttl=ttl)

    def _make_prompt_key(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Create a safe, fixed-length cache key from prompt + parameters.

        This avoids:
        - Very long keys (from long prompts)
        - Unsafe characters for disk filenames
        - Key collisions are minimized via hashing
        """
        # Combine prompt and parameters into a single string
        key_str = f"PROMPT:{prompt}|MAX:{max_tokens}|TEMP:{temperature}"

        # Return MD5 hash as fixed-length key
        return hashlib.md5(key_str.encode("utf-8")).hexdigest()


# Singleton instance for application-wide use
_default_cache: Optional[PromptCache] = None


def get_default_cache(
    max_size: int = 1000,
    default_ttl: int = 3600,
    cache_dir: Optional[Path] = None,
) -> PromptCache:
    """
    Get or create default application cache.

    Args:
        max_size: Maximum cache entries
        default_ttl: Default TTL in seconds
        cache_dir: Cache directory

    Returns:
        Shared PromptCache instance
    """
    global _default_cache

    if _default_cache is None:
        _default_cache = PromptCache(
            max_size=max_size,
            default_ttl=default_ttl,
            cache_dir=cache_dir,
        )

    return _default_cache
