"""Property-based tests for cache (Properties 7, 8).

**Validates: Requirements 2.4, 2.5**
"""

import re
import time
from unittest.mock import patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.services.cache import TTLCache


# =============================================================================
# Helper: normalize_query
# =============================================================================


def normalize_query(query: str) -> str:
    """Normalize query: lowercase, collapse consecutive whitespace to single space, strip.

    This function implements the normalization logic described in the design for
    the RAG client's _normalize_query method. It is tested here as a standalone
    function per the task instructions.
    """
    # Lowercase
    result = query.lower()
    # Collapse consecutive whitespace (spaces, tabs, newlines, etc.) to single space
    result = re.sub(r"\s+", " ", result)
    # Strip leading/trailing whitespace
    result = result.strip()
    return result


# =============================================================================
# Property 7: Query Normalization Equivalence
# =============================================================================


class TestQueryNormalizationEquivalence:
    """Property 7: Query Normalization Equivalence.

    For any two query strings that differ only in letter case, leading/trailing
    whitespace, or consecutive internal whitespace, the normalized form SHALL be
    identical, producing the same cache key.

    **Validates: Requirements 2.4**
    """

    @given(
        query=st.text(
            min_size=1,
            max_size=200,
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                blacklist_categories=("Cs",),
            ),
        )
    )
    @settings(max_examples=10)
    def test_case_insensitive_normalization(self, query: str):
        """Queries differing only in case SHALL produce the same normalized form."""
        assume(query.strip() != "")
        # Exclude characters where lower(upper(c)) != lower(c) due to Unicode case-folding
        # (e.g., µ micro sign -> Μ Greek capital mu -> μ Greek small mu)
        assume(query.lower() == query.upper().lower())
        normalized_lower = normalize_query(query.lower())
        normalized_upper = normalize_query(query.upper())
        normalized_mixed = normalize_query(query.swapcase())

        assert normalized_lower == normalized_upper
        assert normalized_lower == normalized_mixed

    @given(
        base_query=st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P"),
                blacklist_categories=("Cs",),
            ),
        ),
        leading_ws=st.text(
            min_size=0,
            max_size=10,
            alphabet=st.sampled_from([" ", "\t", "\n", "\r"]),
        ),
        trailing_ws=st.text(
            min_size=0,
            max_size=10,
            alphabet=st.sampled_from([" ", "\t", "\n", "\r"]),
        ),
    )
    @settings(max_examples=10)
    def test_leading_trailing_whitespace_normalization(
        self, base_query: str, leading_ws: str, trailing_ws: str
    ):
        """Queries differing only in leading/trailing whitespace SHALL produce the same normalized form."""
        assume(base_query.strip() != "")
        query_with_ws = leading_ws + base_query + trailing_ws
        assert normalize_query(query_with_ws) == normalize_query(base_query)

    @given(
        words=st.lists(
            st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(
                    whitelist_categories=("L", "N"),
                    blacklist_categories=("Cs",),
                ),
            ),
            min_size=2,
            max_size=10,
        ),
        separators=st.lists(
            st.text(
                min_size=1,
                max_size=5,
                alphabet=st.sampled_from([" ", "\t", "\n", "\r", " "]),
            ),
            min_size=1,
            max_size=9,
        ),
    )
    @settings(max_examples=10)
    def test_consecutive_whitespace_normalization(
        self, words: list, separators: list
    ):
        """Queries differing only in consecutive internal whitespace SHALL produce the same normalized form."""
        assume(all(w.strip() != "" for w in words))

        # Build query with varying whitespace between words
        parts = []
        for i, word in enumerate(words):
            parts.append(word)
            if i < len(words) - 1:
                sep = separators[i % len(separators)]
                parts.append(sep)
        query_varied_ws = "".join(parts)

        # Build query with single space between words
        query_single_space = " ".join(words)

        assert normalize_query(query_varied_ws) == normalize_query(query_single_space)

    @given(
        query=st.text(
            min_size=1,
            max_size=200,
            alphabet=st.characters(blacklist_categories=("Cs",)),
        )
    )
    @settings(max_examples=10)
    def test_normalization_is_idempotent(self, query: str):
        """Normalizing an already-normalized query SHALL produce the same result."""
        assume(query.strip() != "")
        once = normalize_query(query)
        twice = normalize_query(once)
        assert once == twice

    @given(
        query=st.text(
            min_size=1,
            max_size=200,
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                blacklist_categories=("Cs",),
            ),
        )
    )
    @settings(max_examples=10)
    def test_normalized_queries_produce_same_cache_key(self, query: str):
        """Equivalent queries after normalization SHALL produce the same cache key (hit same cache entry)."""
        assume(query.strip() != "")
        # Exclude characters where lower(upper(c)) != lower(c) due to Unicode case-folding
        assume(query.lower() == query.upper().lower())
        cache = TTLCache(ttl_seconds=300, max_size=100)

        # Store using normalized key
        key = normalize_query(query)
        cache.set(key, "result")

        # Retrieve using various equivalent forms
        key_upper = normalize_query(query.upper())
        key_with_spaces = normalize_query("  " + query + "  ")

        assert cache.get(key_upper) == "result"
        assert cache.get(key_with_spaces) == "result"


# =============================================================================
# Property 8: Cache Size Invariant and LRU Eviction
# =============================================================================


class TestCacheSizeInvariantAndLRUEviction:
    """Property 8: Cache Size Invariant and LRU Eviction.

    For any sequence of cache set operations, the cache SHALL never contain more
    than 100 entries. When a 101st entry is inserted, the least-recently-used
    entry SHALL be evicted. When an entry's TTL expires, it SHALL not be returned
    by subsequent get operations.

    **Validates: Requirements 2.5**
    """

    @given(
        num_entries=st.integers(min_value=101, max_value=300)
    )
    @settings(max_examples=10)
    def test_cache_never_exceeds_max_size(self, num_entries: int):
        """For any sequence of set operations, the cache SHALL never contain more than 100 entries."""
        cache = TTLCache(ttl_seconds=300, max_size=100)

        for i in range(num_entries):
            cache.set(f"key_{i}", f"value_{i}")
            # Invariant: cache size never exceeds max_size
            assert len(cache._store) <= 100

        # Final size should be exactly max_size
        assert len(cache._store) == 100

    @given(
        num_entries=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=10)
    def test_cache_within_capacity_retains_all_entries(self, num_entries: int):
        """When fewer than max_size entries are inserted, all SHALL be retained."""
        cache = TTLCache(ttl_seconds=300, max_size=100)

        for i in range(num_entries):
            cache.set(f"key_{i}", f"value_{i}")

        assert len(cache._store) == num_entries
        # All entries should be retrievable
        for i in range(num_entries):
            assert cache.get(f"key_{i}") == f"value_{i}"

    @given(
        access_order=st.permutations(list(range(100)))
    )
    @settings(max_examples=10)
    def test_lru_eviction_removes_least_recently_used(self, access_order: list):
        """When a 101st entry is inserted, the least-recently-used entry SHALL be evicted."""
        cache = TTLCache(ttl_seconds=300, max_size=100)

        # Fill cache to capacity
        for i in range(100):
            cache.set(f"key_{i}", f"value_{i}")

        # Access entries in a specific order to establish LRU ordering
        # The last accessed entry is most recently used
        for idx in access_order:
            cache.get(f"key_{idx}")

        # The first entry in access_order is the least recently used
        lru_key = f"key_{access_order[0]}"

        # Insert a new entry (101st)
        cache.set("new_key", "new_value")

        # The LRU entry should have been evicted
        assert cache.get(lru_key) is None
        # The new entry should be present
        assert cache.get("new_key") == "new_value"
        # Cache size should still be 100
        assert len(cache._store) == 100

    def test_ttl_expired_entries_not_returned(self):
        """When an entry's TTL expires, it SHALL not be returned by subsequent get operations."""
        cache = TTLCache(ttl_seconds=1, max_size=100)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Simulate time passing beyond TTL
        with patch("app.services.cache.time.time", return_value=time.time() + 2):
            assert cache.get("key1") is None

    @given(
        ttl=st.integers(min_value=1, max_value=60),
        elapsed=st.integers(min_value=1, max_value=120),
    )
    @settings(max_examples=15)
    def test_ttl_expiration_property(self, ttl: int, elapsed: int):
        """Entries SHALL be returned only when elapsed time <= TTL, and not returned when elapsed > TTL."""
        cache = TTLCache(ttl_seconds=ttl, max_size=100)

        base_time = 1000000.0
        with patch("app.services.cache.time.time", return_value=base_time):
            cache.set("key", "value")

        with patch("app.services.cache.time.time", return_value=base_time + elapsed):
            result = cache.get("key")
            if elapsed > ttl:
                assert result is None, f"Entry should be expired after {elapsed}s with TTL={ttl}s"
            else:
                assert result == "value", f"Entry should still be valid after {elapsed}s with TTL={ttl}s"

    @given(
        max_size=st.integers(min_value=1, max_value=50),
        num_inserts=st.integers(min_value=1, max_value=200),
    )
    @settings(max_examples=10)
    def test_cache_size_invariant_various_max_sizes(self, max_size: int, num_inserts: int):
        """The cache size invariant holds for any configured max_size."""
        cache = TTLCache(ttl_seconds=300, max_size=max_size)

        for i in range(num_inserts):
            cache.set(f"key_{i}", f"value_{i}")
            assert len(cache._store) <= max_size

    @given(
        keys_to_insert=st.lists(
            st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
            min_size=101,
            max_size=150,
            unique=True,
        )
    )
    @settings(max_examples=15)
    def test_lru_eviction_preserves_most_recent_entries(self, keys_to_insert: list):
        """After overflow, the most recently inserted entries SHALL be retained."""
        cache = TTLCache(ttl_seconds=300, max_size=100)

        for key in keys_to_insert:
            cache.set(key, f"val_{key}")

        # The last 100 keys should all be present
        last_100 = keys_to_insert[-100:]
        for key in last_100:
            assert cache.get(key) == f"val_{key}", f"Key '{key}' should be in cache"

        # Keys before the last 100 should have been evicted
        evicted = keys_to_insert[:-100]
        for key in evicted:
            assert cache.get(key) is None, f"Key '{key}' should have been evicted"

    def test_updating_existing_key_does_not_increase_size(self):
        """Updating an existing key SHALL not increase the cache size."""
        cache = TTLCache(ttl_seconds=300, max_size=100)

        # Fill to capacity
        for i in range(100):
            cache.set(f"key_{i}", f"value_{i}")

        assert len(cache._store) == 100

        # Update an existing key
        cache.set("key_50", "updated_value")

        # Size should remain the same
        assert len(cache._store) == 100
        assert cache.get("key_50") == "updated_value"
