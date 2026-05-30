"""Property-based tests for RAG client (Properties 5, 6, 9).

**Validates: Requirements 2.1, 2.2, 2.3, 2.8**
"""

import logging
from unittest.mock import patch, MagicMock

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.models.rag import ChunkMetadata
from app.models.report import SourceType
from app.services.rag_client import (
    RAGClient,
    SCORE_THRESHOLD,
    TOP_K_DEFAULT,
    TOP_K_MIN,
    TOP_K_MAX,
)


# =============================================================================
# Strategies
# =============================================================================


def chunk_metadata_strategy(
    source_type_st=None,
    score_st=None,
):
    """Strategy for generating ChunkMetadata instances."""
    if source_type_st is None:
        source_type_st = st.sampled_from([SourceType.FICO_INTERNAL, SourceType.PUBLIC])
    if score_st is None:
        score_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

    return st.builds(
        ChunkMetadata,
        source_type=source_type_st,
        document_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "P"))),
        section_id=st.one_of(st.none(), st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N")))),
        ingestion_date=st.just("2024-01-15T10:30:00Z"),
        similarity_score=score_st,
        content=st.text(min_size=1, max_size=200, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"))),
    )


# Strategy for invalid top-k values (not valid integers in range 1-20)
invalid_top_k_strings = st.one_of(
    # Non-integer strings
    st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "P"))),
    # Floats as strings
    st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)
    .filter(lambda x: x != int(x) if x == x else True)
    .map(str),
    # Empty string
    st.just(""),
    # None
    st.just(None),
)

# Strategy for out-of-range integer values
out_of_range_top_k = st.one_of(
    st.integers(min_value=-1000, max_value=0),
    st.integers(min_value=21, max_value=1000),
)

# Strategy for valid top-k values
valid_top_k = st.integers(min_value=TOP_K_MIN, max_value=TOP_K_MAX)


# =============================================================================
# Property 5: RAG Top-K Fallback
# =============================================================================


class TestRAGTopKFallback:
    """Property 5: RAG Top-K Fallback.

    For any RAG_TOP_K environment variable value that is not a valid integer
    in the range 1-20, the RAG Client SHALL use the default value of 5 for
    retrieval and log a warning.

    **Validates: Requirements 2.1**
    """

    @given(value=out_of_range_top_k)
    @settings(max_examples=15)
    def test_out_of_range_integer_falls_back_to_default(self, value: int):
        """Out-of-range integers (< 1 or > 20) SHALL fall back to default 5."""
        mock_settings = MagicMock()
        mock_settings.rag_top_k = value
        mock_settings.aws_region = "us-east-1"
        mock_settings.aws_access_key_id = "test"
        mock_settings.aws_secret_access_key = "test"
        mock_settings.bedrock_kb_id = "test-kb-id"

        with patch("app.services.rag_client.boto3.client"):
            with patch("app.services.rag_client.logger") as mock_logger:
                from app.services.cache import TTLCache

                cache = TTLCache(ttl_seconds=300, max_size=100)
                client = RAGClient(settings=mock_settings, cache=cache)

                assert client._top_k == TOP_K_DEFAULT
                mock_logger.warning.assert_called()

    @given(value=valid_top_k)
    @settings(max_examples=10)
    def test_valid_range_integer_is_accepted(self, value: int):
        """Valid integers in range 1-20 SHALL be used as-is without warning."""
        mock_settings = MagicMock()
        mock_settings.rag_top_k = value
        mock_settings.aws_region = "us-east-1"
        mock_settings.aws_access_key_id = "test"
        mock_settings.aws_secret_access_key = "test"
        mock_settings.bedrock_kb_id = "test-kb-id"

        with patch("app.services.rag_client.boto3.client"):
            with patch("app.services.rag_client.logger") as mock_logger:
                from app.services.cache import TTLCache

                cache = TTLCache(ttl_seconds=300, max_size=100)
                client = RAGClient(settings=mock_settings, cache=cache)

                assert client._top_k == value
                mock_logger.warning.assert_not_called()

    @given(value=invalid_top_k_strings)
    @settings(max_examples=15)
    def test_non_integer_values_fall_back_to_default(self, value):
        """Non-integer values (strings, None, floats) SHALL fall back to default 5 with warning."""
        mock_settings = MagicMock()
        mock_settings.rag_top_k = value
        mock_settings.aws_region = "us-east-1"
        mock_settings.aws_access_key_id = "test"
        mock_settings.aws_secret_access_key = "test"
        mock_settings.bedrock_kb_id = "test-kb-id"

        with patch("app.services.rag_client.boto3.client"):
            with patch("app.services.rag_client.logger") as mock_logger:
                from app.services.cache import TTLCache

                cache = TTLCache(ttl_seconds=300, max_size=100)
                client = RAGClient(settings=mock_settings, cache=cache)

                assert client._top_k == TOP_K_DEFAULT
                mock_logger.warning.assert_called()

    def test_boundary_values_min_max(self):
        """Boundary values 1 and 20 SHALL be accepted; 0 and 21 SHALL fall back."""
        mock_settings = MagicMock()
        mock_settings.aws_region = "us-east-1"
        mock_settings.aws_access_key_id = "test"
        mock_settings.aws_secret_access_key = "test"
        mock_settings.bedrock_kb_id = "test-kb-id"

        with patch("app.services.rag_client.boto3.client"):
            from app.services.cache import TTLCache

            cache = TTLCache(ttl_seconds=300, max_size=100)

            # Min boundary (valid)
            mock_settings.rag_top_k = 1
            client = RAGClient(settings=mock_settings, cache=cache)
            assert client._top_k == 1

            # Max boundary (valid)
            mock_settings.rag_top_k = 20
            client = RAGClient(settings=mock_settings, cache=cache)
            assert client._top_k == 20

            # Below min (invalid)
            mock_settings.rag_top_k = 0
            client = RAGClient(settings=mock_settings, cache=cache)
            assert client._top_k == TOP_K_DEFAULT

            # Above max (invalid)
            mock_settings.rag_top_k = 21
            client = RAGClient(settings=mock_settings, cache=cache)
            assert client._top_k == TOP_K_DEFAULT


# =============================================================================
# Property 6: Source Separation and Score Ordering
# =============================================================================


class TestSourceSeparationAndScoreOrdering:
    """Property 6: Source Separation and Score Ordering.

    For any set of retrieved chunks with mixed source types, the RAG Client
    SHALL partition them into exactly two collections (FICO_INTERNAL and PUBLIC)
    where: (a) every chunk in the FICO collection has source_type FICO_INTERNAL,
    (b) every chunk in the PUBLIC collection has source_type PUBLIC, and
    (c) each collection is sorted in descending order by similarity score.

    **Validates: Requirements 2.2, 2.3**
    """

    def _create_rag_client(self):
        """Create a RAGClient instance for testing."""
        mock_settings = MagicMock()
        mock_settings.rag_top_k = 5
        mock_settings.aws_region = "us-east-1"
        mock_settings.aws_access_key_id = "test"
        mock_settings.aws_secret_access_key = "test"
        mock_settings.bedrock_kb_id = "test-kb-id"

        with patch("app.services.rag_client.boto3.client"):
            from app.services.cache import TTLCache

            cache = TTLCache(ttl_seconds=300, max_size=100)
            return RAGClient(settings=mock_settings, cache=cache)

    @given(
        chunks=st.lists(
            chunk_metadata_strategy(
                score_st=st.floats(min_value=0.3, max_value=1.0, allow_nan=False, allow_infinity=False),
            ),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=10)
    def test_fico_collection_contains_only_fico_chunks(self, chunks: list):
        """Every chunk in the FICO collection SHALL have source_type FICO_INTERNAL."""
        client = self._create_rag_client()
        fico_chunks, _ = client._separate_by_source(chunks)

        for chunk in fico_chunks:
            assert chunk.source_type == SourceType.FICO_INTERNAL, (
                f"FICO collection contains chunk with source_type {chunk.source_type}"
            )

    @given(
        chunks=st.lists(
            chunk_metadata_strategy(
                score_st=st.floats(min_value=0.3, max_value=1.0, allow_nan=False, allow_infinity=False),
            ),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=10)
    def test_public_collection_contains_only_public_chunks(self, chunks: list):
        """Every chunk in the PUBLIC collection SHALL have source_type PUBLIC."""
        client = self._create_rag_client()
        _, public_chunks = client._separate_by_source(chunks)

        for chunk in public_chunks:
            assert chunk.source_type == SourceType.PUBLIC, (
                f"PUBLIC collection contains chunk with source_type {chunk.source_type}"
            )

    @given(
        chunks=st.lists(
            chunk_metadata_strategy(
                score_st=st.floats(min_value=0.3, max_value=1.0, allow_nan=False, allow_infinity=False),
            ),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=10)
    def test_fico_collection_sorted_descending_by_score(self, chunks: list):
        """FICO collection SHALL be sorted in descending order by similarity score."""
        client = self._create_rag_client()
        fico_chunks, _ = client._separate_by_source(chunks)

        if len(fico_chunks) > 1:
            scores = [c.similarity_score for c in fico_chunks]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], (
                    f"FICO chunks not sorted descending: {scores[i]} < {scores[i+1]}"
                )

    @given(
        chunks=st.lists(
            chunk_metadata_strategy(
                score_st=st.floats(min_value=0.3, max_value=1.0, allow_nan=False, allow_infinity=False),
            ),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=10)
    def test_public_collection_sorted_descending_by_score(self, chunks: list):
        """PUBLIC collection SHALL be sorted in descending order by similarity score."""
        client = self._create_rag_client()
        _, public_chunks = client._separate_by_source(chunks)

        if len(public_chunks) > 1:
            scores = [c.similarity_score for c in public_chunks]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], (
                    f"PUBLIC chunks not sorted descending: {scores[i]} < {scores[i+1]}"
                )

    @given(
        chunks=st.lists(
            chunk_metadata_strategy(
                score_st=st.floats(min_value=0.3, max_value=1.0, allow_nan=False, allow_infinity=False),
            ),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=10)
    def test_partition_preserves_all_chunks(self, chunks: list):
        """The partition SHALL preserve all input chunks (no chunks lost or duplicated)."""
        client = self._create_rag_client()
        fico_chunks, public_chunks = client._separate_by_source(chunks)

        # Total count should match
        assert len(fico_chunks) + len(public_chunks) == len(chunks), (
            f"Partition lost or duplicated chunks: "
            f"{len(fico_chunks)} + {len(public_chunks)} != {len(chunks)}"
        )

    @given(
        fico_chunks=st.lists(
            chunk_metadata_strategy(
                source_type_st=st.just(SourceType.FICO_INTERNAL),
                score_st=st.floats(min_value=0.3, max_value=1.0, allow_nan=False, allow_infinity=False),
            ),
            min_size=1,
            max_size=10,
        ),
        public_chunks=st.lists(
            chunk_metadata_strategy(
                source_type_st=st.just(SourceType.PUBLIC),
                score_st=st.floats(min_value=0.3, max_value=1.0, allow_nan=False, allow_infinity=False),
            ),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=10)
    def test_mixed_source_types_correctly_partitioned(
        self, fico_chunks: list, public_chunks: list
    ):
        """Mixed source type chunks SHALL be correctly partitioned into two collections."""
        client = self._create_rag_client()
        mixed_chunks = fico_chunks + public_chunks

        result_fico, result_public = client._separate_by_source(mixed_chunks)

        # Count should match
        assert len(result_fico) == len(fico_chunks)
        assert len(result_public) == len(public_chunks)

    def test_empty_input_produces_empty_collections(self):
        """An empty input list SHALL produce two empty collections."""
        client = self._create_rag_client()
        fico_chunks, public_chunks = client._separate_by_source([])

        assert fico_chunks == []
        assert public_chunks == []


# =============================================================================
# Property 9: Low-Score Chunk Filtering
# =============================================================================


class TestLowScoreChunkFiltering:
    """Property 9: Low-Score Chunk Filtering.

    For any set of chunks returned by the Knowledge Base, the RAG Client output
    SHALL contain only chunks with similarity_score >= 0.3. If all chunks have
    scores below 0.3, the result SHALL be an empty set with error flag false.

    **Validates: Requirements 2.8**
    """

    def _create_rag_client(self):
        """Create a RAGClient instance for testing."""
        mock_settings = MagicMock()
        mock_settings.rag_top_k = 5
        mock_settings.aws_region = "us-east-1"
        mock_settings.aws_access_key_id = "test"
        mock_settings.aws_secret_access_key = "test"
        mock_settings.bedrock_kb_id = "test-kb-id"

        with patch("app.services.rag_client.boto3.client"):
            from app.services.cache import TTLCache

            cache = TTLCache(ttl_seconds=300, max_size=100)
            return RAGClient(settings=mock_settings, cache=cache)

    @given(
        chunks=st.lists(
            chunk_metadata_strategy(),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=10)
    def test_filtered_output_contains_only_chunks_above_threshold(self, chunks: list):
        """Output SHALL contain only chunks with similarity_score >= 0.3."""
        client = self._create_rag_client()
        filtered = client._filter_low_score_chunks(chunks)

        for chunk in filtered:
            assert chunk.similarity_score >= SCORE_THRESHOLD, (
                f"Filtered output contains chunk with score {chunk.similarity_score} "
                f"below threshold {SCORE_THRESHOLD}"
            )

    @given(
        chunks=st.lists(
            chunk_metadata_strategy(
                score_st=st.floats(
                    min_value=0.0,
                    max_value=0.2999999,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            ),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=10)
    def test_all_below_threshold_produces_empty_result(self, chunks: list):
        """If all chunks have scores below 0.3, the result SHALL be an empty set."""
        client = self._create_rag_client()
        filtered = client._filter_low_score_chunks(chunks)

        assert filtered == [], (
            f"Expected empty result when all scores below threshold, got {len(filtered)} chunks"
        )

    @given(
        chunks=st.lists(
            chunk_metadata_strategy(
                score_st=st.floats(
                    min_value=0.3,
                    max_value=1.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            ),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=10)
    def test_all_above_threshold_preserves_all_chunks(self, chunks: list):
        """If all chunks have scores >= 0.3, all SHALL be preserved."""
        client = self._create_rag_client()
        filtered = client._filter_low_score_chunks(chunks)

        assert len(filtered) == len(chunks), (
            f"Expected all {len(chunks)} chunks preserved, got {len(filtered)}"
        )

    @given(
        above_chunks=st.lists(
            chunk_metadata_strategy(
                score_st=st.floats(
                    min_value=0.3,
                    max_value=1.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            ),
            min_size=1,
            max_size=10,
        ),
        below_chunks=st.lists(
            chunk_metadata_strategy(
                score_st=st.floats(
                    min_value=0.0,
                    max_value=0.2999999,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            ),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=10)
    def test_mixed_scores_filters_correctly(self, above_chunks: list, below_chunks: list):
        """Mixed score chunks SHALL retain only those >= 0.3."""
        client = self._create_rag_client()
        mixed = above_chunks + below_chunks
        filtered = client._filter_low_score_chunks(mixed)

        # Should retain exactly the above-threshold chunks
        assert len(filtered) == len(above_chunks), (
            f"Expected {len(above_chunks)} chunks above threshold, got {len(filtered)}"
        )

        # All filtered chunks should be from the above_chunks set
        for chunk in filtered:
            assert chunk.similarity_score >= SCORE_THRESHOLD

    @given(
        chunks=st.lists(
            chunk_metadata_strategy(),
            min_size=0,
            max_size=20,
        )
    )
    @settings(max_examples=10)
    def test_filtering_never_adds_chunks(self, chunks: list):
        """Filtering SHALL never produce more chunks than the input."""
        client = self._create_rag_client()
        filtered = client._filter_low_score_chunks(chunks)

        assert len(filtered) <= len(chunks), (
            f"Filtering produced more chunks ({len(filtered)}) than input ({len(chunks)})"
        )

    def test_exact_threshold_value_is_included(self):
        """A chunk with exactly score 0.3 SHALL be included in the output."""
        client = self._create_rag_client()
        chunk = ChunkMetadata(
            source_type=SourceType.FICO_INTERNAL,
            document_name="test.pdf",
            section_id=None,
            ingestion_date="2024-01-15T10:30:00Z",
            similarity_score=0.3,
            content="Test content",
        )
        filtered = client._filter_low_score_chunks([chunk])

        assert len(filtered) == 1
        assert filtered[0].similarity_score == 0.3

    def test_just_below_threshold_is_excluded(self):
        """A chunk with score just below 0.3 SHALL be excluded."""
        client = self._create_rag_client()
        chunk = ChunkMetadata(
            source_type=SourceType.FICO_INTERNAL,
            document_name="test.pdf",
            section_id=None,
            ingestion_date="2024-01-15T10:30:00Z",
            similarity_score=0.29999,
            content="Test content",
        )
        filtered = client._filter_low_score_chunks([chunk])

        assert len(filtered) == 0

    def test_empty_input_produces_empty_output(self):
        """An empty input list SHALL produce an empty output."""
        client = self._create_rag_client()
        filtered = client._filter_low_score_chunks([])

        assert filtered == []
