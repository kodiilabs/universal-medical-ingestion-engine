# ============================================================================
# tests/unit/test_medgemma_cache.py
# ============================================================================
"""
Tests for MedGemma cache module
"""

import pytest
import time
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

from src.medical_ingestion.medgemma.cache import (
    CacheEntry,
    CacheStatistics,
    MedGemmaCache,
    PromptCache,
    get_default_cache,
)


class TestCacheEntry:
    """Test CacheEntry dataclass"""

    def test_create_entry(self):
        """Test creating cache entry"""
        entry = CacheEntry(
            key="test_key",
            value={"data": "test"},
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            ttl_seconds=60,
        )

        assert entry.key == "test_key"
        assert entry.value == {"data": "test"}
        assert entry.access_count == 0

    def test_expiration_check(self):
        """Test TTL expiration"""
        # Non-expiring entry
        entry1 = CacheEntry(
            key="key1",
            value="value1",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            ttl_seconds=None,
        )
        assert not entry1.is_expired()

        # Expired entry (simulate old creation time)
        from datetime import timedelta
        old_time = datetime.now() - timedelta(seconds=120)
        entry2 = CacheEntry(
            key="key2",
            value="value2",
            created_at=old_time,
            last_accessed=old_time,
            ttl_seconds=60,  # 1 minute TTL
        )
        assert entry2.is_expired()

    def test_mark_accessed(self):
        """Test access tracking"""
        entry = CacheEntry(
            key="test",
            value="data",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
        )

        initial_count = entry.access_count
        initial_time = entry.last_accessed

        time.sleep(0.01)  # Small delay
        entry.mark_accessed()

        assert entry.access_count == initial_count + 1
        assert entry.last_accessed > initial_time


class TestCacheStatistics:
    """Test CacheStatistics class"""

    def test_initial_state(self):
        """Test initial statistics"""
        stats = CacheStatistics()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate() == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation"""
        stats = CacheStatistics()
        stats.hits = 7
        stats.misses = 3
        assert stats.hit_rate() == 0.7

    def test_to_dict(self):
        """Test statistics serialization"""
        stats = CacheStatistics()
        stats.hits = 10
        stats.misses = 5
        stats.entry_count = 8

        result = stats.to_dict()
        assert result["hits"] == 10
        assert result["misses"] == 5
        assert result["hit_rate"] == 0.666666666666666666
        assert result["entry_count"] == 8


class TestMedGemmaCache:
    """Test MedGemmaCache class"""

    def test_cache_creation(self):
        """Test creating cache instance"""
        cache = MedGemmaCache(max_size=100)
        assert cache.max_size == 100
        stats = cache.get_statistics()
        assert stats["entry_count"] == 0

    def test_set_and_get(self):
        """Test basic set/get operations"""
        cache = MedGemmaCache(max_size=10)

        # Set value
        cache.set("key1", {"data": "value1"})

        # Get value
        result = cache.get("key1")
        assert result == {"data": "value1"}

        # Get non-existent key
        result = cache.get("nonexistent", default="default_value")
        assert result == "default_value"

    def test_cache_statistics(self):
        """Test statistics tracking"""
        cache = MedGemmaCache(max_size=10)

        # Initial state
        stats = cache.get_statistics()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

        # Set and get (hit)
        cache.set("key1", "value1")
        cache.get("key1")

        stats = cache.get_statistics()
        assert stats["hits"] == 1
        assert stats["misses"] == 0
        assert stats["writes"] == 1

        # Get non-existent (miss)
        cache.get("nonexistent")

        stats = cache.get_statistics()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_lru_eviction(self):
        """Test LRU eviction when max_size reached"""
        cache = MedGemmaCache(max_size=3)

        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        assert cache.get("key1") == "value1"

        # Add fourth item (should evict key2, the oldest unused)
        cache.set("key4", "value4")

        # key2 should be evicted
        assert cache.get("key2") is None
        assert cache.get("key1") == "value1"  # Recently accessed
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

        stats = cache.get_statistics()
        assert stats["evictions"] == 1

    def test_ttl_expiration(self):
        """Test TTL-based expiration"""
        cache = MedGemmaCache(default_ttl=1)  # 1 second TTL

        cache.set("key1", "value1")

        # Should be available immediately
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        result = cache.get("key1")
        assert result is None

        stats = cache.get_statistics()
        assert stats["expirations"] >= 1

    def test_clear(self):
        """Test clearing cache"""
        cache = MedGemmaCache(max_size=10)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert cache.get("key1") == "value1"

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

        stats = cache.get_statistics()
        assert stats["entry_count"] == 0

    def test_delete(self):
        """Test deleting single entry"""
        cache = MedGemmaCache(max_size=10)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

        # Delete non-existent
        assert cache.delete("nonexistent") is False

    def test_cleanup_expired(self):
        """Test cleanup of expired entries"""
        cache = MedGemmaCache(max_size=10)

        # Add entries with short TTL
        cache.set("key1", "value1", ttl=1)
        cache.set("key2", "value2", ttl=1)
        cache.set("key3", "value3", ttl=100)  # Long TTL

        # Wait for expiration
        time.sleep(1.1)

        # Cleanup
        removed = cache.cleanup_expired()

        assert removed == 2
        assert cache.get("key3") == "value3"
        assert cache.get("key1") is None

    def test_warm_cache(self):
        """Test cache warming"""
        cache = MedGemmaCache(max_size=10)

        entries = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
        }

        count = cache.warm_cache(entries)

        assert count == 3
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_get_entries_by_access(self):
        """Test getting most accessed entries"""
        cache = MedGemmaCache(max_size=10)

        # Add entries
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access them different amounts
        cache.get("key1")
        cache.get("key1")
        cache.get("key1")  # 3 accesses

        cache.get("key2")
        cache.get("key2")  # 2 accesses

        cache.get("key3")  # 1 access

        # Get top entries
        top_entries = cache.get_entries_by_access(limit=2)

        assert len(top_entries) == 2
        assert top_entries[0]["access_count"] == 3
        assert top_entries[1]["access_count"] == 2

    def test_persistence(self):
        """Test saving and loading cache from disk"""
        temp_dir = Path(tempfile.mkdtemp())

        try:
            # Create cache with persistence
            cache1 = MedGemmaCache(
                max_size=10,
                cache_dir=temp_dir,
                auto_persist=True,
                persist_interval=1,
            )

            # Add entries
            cache1.set("key1", {"data": "value1"})
            cache1.set("key2", {"data": "value2"})

            # Force save
            cache1._save_to_disk()

            # Create new cache instance (should load from disk)
            cache2 = MedGemmaCache(
                max_size=10,
                cache_dir=temp_dir,
            )

            # Check values loaded
            assert cache2.get("key1") == {"data": "value1"}
            assert cache2.get("key2") == {"data": "value2"}

        finally:
            # Cleanup
            shutil.rmtree(temp_dir)


class TestPromptCache:
    """Test PromptCache specialized cache"""

    def test_prompt_caching(self):
        """Test caching with prompt parameters"""
        cache = PromptCache(max_size=10)

        response = {
            "text": "Generated response",
            "tokens": 100,
            "model": "medgemma-local",
        }

        # Cache response
        cache.set_response(
            prompt="What is diabetes?",
            max_tokens=1000,
            temperature=0.1,
            response=response,
        )

        # Retrieve response
        result = cache.get_response(
            prompt="What is diabetes?",
            max_tokens=1000,
            temperature=0.1,
        )

        assert result == response

    def test_different_parameters(self):
        """Test that different parameters create different cache keys"""
        cache = PromptCache(max_size=10)

        response1 = {"text": "Response 1"}
        response2 = {"text": "Response 2"}

        # Same prompt, different temperature
        cache.set_response(
            prompt="What is diabetes?",
            max_tokens=1000,
            temperature=0.1,
            response=response1,
        )

        cache.set_response(
            prompt="What is diabetes?",
            max_tokens=1000,
            temperature=0.5,
            response=response2,
        )

        # Should get different responses
        result1 = cache.get_response(
            prompt="What is diabetes?",
            max_tokens=1000,
            temperature=0.1,
        )

        result2 = cache.get_response(
            prompt="What is diabetes?",
            max_tokens=1000,
            temperature=0.5,
        )

        assert result1 == response1
        assert result2 == response2


class TestDefaultCache:
    """Test default cache singleton"""

    def test_singleton_behavior(self):
        """Test that get_default_cache returns same instance"""
        cache1 = get_default_cache()
        cache2 = get_default_cache()

        assert cache1 is cache2

        # Set in one, get from other
        cache1.set("test_key", "test_value")
        assert cache2.get("test_key") == "test_value"
