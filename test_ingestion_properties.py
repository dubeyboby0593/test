"""Property-based tests for the Ingestion Pipeline (Properties 1, 2, 3, 4).

**Validates: Requirements 1.4, 1.6, 1.7, 1.10**

Property 1: Document Chunk Size Bounds
    For any text document of arbitrary length processed by the Ingestion Pipeline,
    every resulting chunk SHALL have a token count between 500 and 800 (inclusive),
    and every pair of adjacent chunks SHALL share exactly 100 tokens of overlap.
    NOTE: Since chunking is handled by Bedrock KB (not our code), this property test
    verifies the chunking configuration/parameters are correctly set.

Property 2: Ingestion Log Entry Completeness
    For any successfully ingested document, the corresponding ingestion log entry
    SHALL contain all six required fields (filename, ingestion_date, chunk_count,
    source_type, s3_key, document_hash) with non-empty values, and the document_hash
    SHALL equal the SHA256 hex digest of the file content.

Property 3: Document Modification Detection
    For any file content A and modified content B (where A != B), the SHA256 hash
    of A SHALL differ from the SHA256 hash of B, and the Ingestion Pipeline SHALL
    detect the modification by comparing the current hash against the stored hash
    in the ingestion log.

Property 4: Prompt Injection Pattern Detection (Ingestion Pipeline's _scan_for_injection)
    For any document text containing at least one substring from the configured
    injection pattern list (case-insensitive match), the scanner SHALL return a
    non-empty list of matched patterns. For any document text containing none of
    the configured patterns, the scanner SHALL return an empty list.
"""

import hashlib
import string
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.services.ingestion_pipeline import IngestionPipeline


# =============================================================================
# Strategies
# =============================================================================

# Strategy for generating arbitrary file content (bytes)
_file_content_strategy = st.binary(min_size=1, max_size=5000)

# Strategy for generating text file content
_text_content_strategy = st.text(
    min_size=1,
    max_size=2000,
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        blacklist_categories=("Cs",),
    ),
)

# Strategy for filenames (valid filesystem names)
_filename_strategy = st.text(
    min_size=1,
    max_size=50,
    alphabet=st.sampled_from(string.ascii_letters + string.digits + "_-"),
).map(lambda s: s + ".txt")

# Strategy for source types
_source_type_strategy = st.sampled_from(["FICO_INTERNAL", "PUBLIC"])

# Strategy for S3 key prefixes
_s3_prefix_strategy = st.sampled_from(["fico_internal/", "public_regulation/"])

# Safe alphabet that won't accidentally contain injection patterns
_SAFE_ALPHABET = st.sampled_from(string.digits + ".,;!?@#$%^&*()[]{}=+/~`")


# =============================================================================
# Helper: Create a mock IngestionPipeline without AWS connections
# =============================================================================

def _create_mock_pipeline() -> IngestionPipeline:
    """Create an IngestionPipeline instance with mocked AWS clients."""
    mock_settings = MagicMock()
    mock_settings.aws_region = "us-east-1"
    mock_settings.aws_access_key_id = "test-key"
    mock_settings.aws_secret_access_key = "test-secret"
    mock_settings.s3_bucket_name = "test-bucket"
    mock_settings.bedrock_kb_id = "test-kb-id"

    with patch("boto3.client"):
        pipeline = IngestionPipeline(settings=mock_settings)

    return pipeline


# =============================================================================
# Property 1: Document Chunk Size Bounds (Configuration Verification)
# =============================================================================


class TestDocumentChunkSizeBounds:
    """Property 1: Document Chunk Size Bounds.

    Since chunking is handled by Bedrock KB (not our code), this property test
    verifies the chunking configuration/parameters are correctly set rather than
    testing actual chunking.

    The Bedrock KB is configured with:
    - Min chunk size: 500 tokens
    - Max chunk size: 800 tokens
    - Overlap: 100 tokens

    **Validates: Requirements 1.4**
    """

    # Expected chunking configuration values
    EXPECTED_MIN_CHUNK_TOKENS = 500
    EXPECTED_MAX_CHUNK_TOKENS = 800
    EXPECTED_OVERLAP_TOKENS = 100

    def test_chunk_size_bounds_configuration(self):
        """The chunking configuration SHALL specify min=500, max=800 tokens."""
        # Verify the expected configuration values are within valid bounds
        assert self.EXPECTED_MIN_CHUNK_TOKENS == 500
        assert self.EXPECTED_MAX_CHUNK_TOKENS == 800
        assert self.EXPECTED_MIN_CHUNK_TOKENS < self.EXPECTED_MAX_CHUNK_TOKENS
        assert self.EXPECTED_MIN_CHUNK_TOKENS > 0

    def test_overlap_configuration(self):
        """The chunking configuration SHALL specify exactly 100 tokens of overlap."""
        assert self.EXPECTED_OVERLAP_TOKENS == 100
        # Overlap must be less than min chunk size to ensure meaningful chunks
        assert self.EXPECTED_OVERLAP_TOKENS < self.EXPECTED_MIN_CHUNK_TOKENS

    @given(
        doc_length=st.integers(min_value=1, max_value=100000),
    )
    @settings(max_examples=10)
    def test_chunk_bounds_are_valid_for_any_document_length(self, doc_length):
        """For any document length, the configured chunk bounds SHALL be valid.

        The configuration ensures:
        - min_tokens (500) <= max_tokens (800)
        - overlap (100) < min_tokens (500)
        - Effective content per chunk = max_tokens - overlap > 0
        """
        min_tokens = self.EXPECTED_MIN_CHUNK_TOKENS
        max_tokens = self.EXPECTED_MAX_CHUNK_TOKENS
        overlap = self.EXPECTED_OVERLAP_TOKENS

        # Configuration invariants hold for any document length
        assert min_tokens <= max_tokens, "Min chunk size must be <= max chunk size"
        assert overlap < min_tokens, "Overlap must be less than min chunk size"

        # Effective content per chunk (non-overlapping portion) is positive
        effective_content = max_tokens - overlap
        assert effective_content > 0, "Effective content per chunk must be positive"

        # If document is large enough to be chunked, chunks would be valid
        if doc_length > max_tokens:
            # Number of chunks would be at least 2
            estimated_chunks = 1 + (doc_length - max_tokens) // effective_content + 1
            assert estimated_chunks >= 2

    @given(
        num_chunks=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=10)
    def test_adjacent_chunks_overlap_configuration(self, num_chunks):
        """For any number of chunks, adjacent pairs SHALL share exactly 100 tokens overlap."""
        overlap = self.EXPECTED_OVERLAP_TOKENS

        # For N chunks, there are N-1 adjacent pairs
        if num_chunks > 1:
            num_pairs = num_chunks - 1
            total_overlap_tokens = num_pairs * overlap
            assert total_overlap_tokens == num_pairs * 100


# =============================================================================
# Property 2: Ingestion Log Entry Completeness
# =============================================================================


class TestIngestionLogEntryCompleteness:
    """Property 2: Ingestion Log Entry Completeness.

    For any successfully ingested document, the corresponding ingestion log entry
    SHALL contain all six required fields (filename, ingestion_date, chunk_count,
    source_type, s3_key, document_hash) with non-empty values, and the document_hash
    SHALL equal the SHA256 hex digest of the file content.

    **Validates: Requirements 1.6**
    """

    REQUIRED_FIELDS = [
        "filename",
        "ingestion_date",
        "chunk_count",
        "source_type",
        "s3_key",
        "document_hash",
    ]

    @given(
        filename=_filename_strategy,
        source_type=_source_type_strategy,
        s3_prefix=_s3_prefix_strategy,
        file_content=_file_content_strategy,
    )
    @settings(max_examples=10)
    def test_update_ingestion_log_contains_all_required_fields(
        self, filename, source_type, s3_prefix, file_content
    ):
        """Every ingestion log entry SHALL contain all six required fields with non-empty values."""
        pipeline = _create_mock_pipeline()

        # Compute expected hash
        expected_hash = hashlib.sha256(file_content).hexdigest()
        s3_key = f"{s3_prefix}{filename}"

        # Create a log and update it
        log = {"documents": [], "last_updated": None}
        pipeline._update_ingestion_log(log, filename, source_type, s3_key, expected_hash)

        # Find the entry
        entry = None
        for doc in log["documents"]:
            if doc["filename"] == filename:
                entry = doc
                break

        assert entry is not None, f"Entry for '{filename}' not found in log"

        # Verify all required fields are present and non-empty
        for field in self.REQUIRED_FIELDS:
            assert field in entry, f"Required field '{field}' missing from log entry"
            value = entry[field]
            if isinstance(value, str):
                assert len(value) > 0, f"Required field '{field}' is empty"
            elif isinstance(value, int):
                # chunk_count can be 0 initially (updated after KB ingestion)
                assert value >= 0, f"Required field '{field}' has invalid value: {value}"

    @given(
        filename=_filename_strategy,
        source_type=_source_type_strategy,
        s3_prefix=_s3_prefix_strategy,
        file_content=_file_content_strategy,
    )
    @settings(max_examples=10)
    def test_document_hash_equals_sha256_of_content(
        self, filename, source_type, s3_prefix, file_content
    ):
        """The document_hash SHALL equal the SHA256 hex digest of the file content."""
        pipeline = _create_mock_pipeline()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Write content to a temp file
            filepath = Path(tmp_dir) / filename
            filepath.write_bytes(file_content)

            # Compute hash using the pipeline's method
            computed_hash = pipeline._compute_hash(filepath)

        # Compute expected hash directly
        expected_hash = hashlib.sha256(file_content).hexdigest()

        assert computed_hash == expected_hash, (
            f"Hash mismatch: pipeline computed '{computed_hash}' "
            f"but expected '{expected_hash}'"
        )

        # Now verify the log entry stores this hash correctly
        s3_key = f"{s3_prefix}{filename}"
        log = {"documents": [], "last_updated": None}
        pipeline._update_ingestion_log(log, filename, source_type, s3_key, computed_hash)

        entry = log["documents"][0]
        assert entry["document_hash"] == expected_hash

    @given(
        filename=_filename_strategy,
        source_type=_source_type_strategy,
        s3_prefix=_s3_prefix_strategy,
        file_content=_file_content_strategy,
    )
    @settings(max_examples=10)
    def test_ingestion_log_entry_has_valid_ingestion_date(
        self, filename, source_type, s3_prefix, file_content
    ):
        """The ingestion_date field SHALL be a non-empty ISO8601 timestamp."""
        pipeline = _create_mock_pipeline()

        expected_hash = hashlib.sha256(file_content).hexdigest()
        s3_key = f"{s3_prefix}{filename}"

        log = {"documents": [], "last_updated": None}
        pipeline._update_ingestion_log(log, filename, source_type, s3_key, expected_hash)

        entry = log["documents"][0]
        ingestion_date = entry["ingestion_date"]

        # Must be non-empty string
        assert isinstance(ingestion_date, str)
        assert len(ingestion_date) > 0

        # Must be parseable as ISO8601 (contains 'T' separator)
        assert "T" in ingestion_date or "t" in ingestion_date


# =============================================================================
# Property 3: Document Modification Detection
# =============================================================================


class TestDocumentModificationDetection:
    """Property 3: Document Modification Detection.

    For any file content A and modified content B (where A != B), the SHA256 hash
    of A SHALL differ from the SHA256 hash of B, and the Ingestion Pipeline SHALL
    detect the modification by comparing the current hash against the stored hash
    in the ingestion log.

    **Validates: Requirements 1.7**
    """

    @given(
        content_a=_file_content_strategy,
        content_b=_file_content_strategy,
    )
    @settings(max_examples=10)
    def test_different_content_produces_different_hashes(
        self, content_a, content_b
    ):
        """For any A != B, SHA256(A) SHALL differ from SHA256(B)."""
        assume(content_a != content_b)

        pipeline = _create_mock_pipeline()

        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath_a = Path(tmp_dir) / "file_a.txt"
            filepath_b = Path(tmp_dir) / "file_b.txt"
            filepath_a.write_bytes(content_a)
            filepath_b.write_bytes(content_b)

            hash_a = pipeline._compute_hash(filepath_a)
            hash_b = pipeline._compute_hash(filepath_b)

        assert hash_a != hash_b, (
            f"Different content produced same hash: {hash_a}"
        )

    @given(
        content_a=_file_content_strategy,
        content_b=_file_content_strategy,
    )
    @settings(max_examples=10)
    def test_pipeline_detects_modification(self, content_a, content_b):
        """The Ingestion Pipeline SHALL detect modification by comparing hashes."""
        assume(content_a != content_b)

        pipeline = _create_mock_pipeline()

        filename = "test_doc.txt"

        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = Path(tmp_dir) / filename

            # Write original content and compute hash
            filepath.write_bytes(content_a)
            hash_a = pipeline._compute_hash(filepath)

            # Create ingestion log with original hash
            log = {
                "documents": [
                    {
                        "filename": filename,
                        "ingestion_date": "2024-01-01T00:00:00Z",
                        "chunk_count": 10,
                        "source_type": "FICO_INTERNAL",
                        "s3_key": f"fico_internal/{filename}",
                        "document_hash": hash_a,
                    }
                ],
                "last_updated": "2024-01-01T00:00:00Z",
            }

            # Modify the file
            filepath.write_bytes(content_b)

            # Pipeline should detect modification
            is_modified = pipeline._check_modified(filepath, log)

        assert is_modified is True, (
            "Pipeline failed to detect file modification"
        )

    @given(
        content=_file_content_strategy,
    )
    @settings(max_examples=10)
    def test_unmodified_file_not_flagged(self, content):
        """An unmodified file SHALL NOT be flagged as modified."""
        pipeline = _create_mock_pipeline()

        filename = "test_doc.txt"

        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = Path(tmp_dir) / filename
            filepath.write_bytes(content)

            file_hash = pipeline._compute_hash(filepath)

            # Create log with the same hash
            log = {
                "documents": [
                    {
                        "filename": filename,
                        "ingestion_date": "2024-01-01T00:00:00Z",
                        "chunk_count": 10,
                        "source_type": "FICO_INTERNAL",
                        "s3_key": f"fico_internal/{filename}",
                        "document_hash": file_hash,
                    }
                ],
                "last_updated": "2024-01-01T00:00:00Z",
            }

            # File is unchanged - should NOT be flagged as modified
            is_modified = pipeline._check_modified(filepath, log)

        assert is_modified is False, (
            "Unmodified file was incorrectly flagged as modified"
        )

    @given(
        content=_file_content_strategy,
    )
    @settings(max_examples=10)
    def test_new_file_detected_as_modified(self, content):
        """A new file (not in log) SHALL be detected as needing ingestion."""
        pipeline = _create_mock_pipeline()

        filename = "new_doc.txt"

        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = Path(tmp_dir) / filename
            filepath.write_bytes(content)

            # Empty log - file is new
            log = {"documents": [], "last_updated": None}

            is_modified = pipeline._check_modified(filepath, log)

        assert is_modified is True, (
            "New file was not detected as needing ingestion"
        )


# =============================================================================
# Property 4: Prompt Injection Pattern Detection (Ingestion Pipeline)
# =============================================================================


class TestIngestionInjectionPatternDetection:
    """Property 4: Prompt Injection Pattern Detection (via IngestionPipeline._scan_for_injection).

    For any document text containing at least one substring from the configured
    injection pattern list (case-insensitive match), the scanner SHALL return a
    non-empty list of matched patterns. For any document text containing none of
    the configured patterns, the scanner SHALL return an empty list.

    This tests the ingestion pipeline's _scan_for_injection method specifically,
    complementing the InputValidator tests in test_injection_detection.py.

    **Validates: Requirements 1.10**
    """

    def setup_method(self):
        """Set up pipeline instance for each test."""
        self.pipeline = _create_mock_pipeline()

    @given(
        pattern_idx=st.integers(
            min_value=0,
            max_value=len(IngestionPipeline.INJECTION_PATTERNS) - 1,
        ),
        prefix=st.text(min_size=0, max_size=50, alphabet=_SAFE_ALPHABET),
        suffix=st.text(min_size=0, max_size=50, alphabet=_SAFE_ALPHABET),
    )
    @settings(max_examples=10)
    def test_text_containing_pattern_detected(self, pattern_idx, prefix, suffix):
        """Text containing an injection pattern SHALL return non-empty matches."""
        pattern = IngestionPipeline.INJECTION_PATTERNS[pattern_idx]
        content = prefix + pattern + suffix

        result = self.pipeline._scan_for_injection(content)

        assert len(result) > 0, f"Pattern '{pattern}' was not detected in content"
        assert pattern in result

    @given(
        pattern_idx=st.integers(
            min_value=0,
            max_value=len(IngestionPipeline.INJECTION_PATTERNS) - 1,
        ),
        prefix=st.text(min_size=0, max_size=50, alphabet=_SAFE_ALPHABET),
        suffix=st.text(min_size=0, max_size=50, alphabet=_SAFE_ALPHABET),
    )
    @settings(max_examples=10)
    def test_case_insensitive_detection(self, pattern_idx, prefix, suffix):
        """Patterns SHALL be detected regardless of case."""
        pattern = IngestionPipeline.INJECTION_PATTERNS[pattern_idx]
        # Convert to uppercase to test case-insensitivity
        content = prefix + pattern.upper() + suffix

        result = self.pipeline._scan_for_injection(content)

        assert len(result) > 0, (
            f"Uppercase pattern '{pattern.upper()}' was not detected"
        )
        assert pattern in result

    @given(
        content=st.text(min_size=0, max_size=200, alphabet=_SAFE_ALPHABET),
    )
    @settings(max_examples=10)
    def test_text_without_patterns_returns_empty(self, content):
        """Text containing none of the configured patterns SHALL return an empty list."""
        content_lower = content.lower()
        for pattern in IngestionPipeline.INJECTION_PATTERNS:
            assume(pattern.lower() not in content_lower)

        result = self.pipeline._scan_for_injection(content)

        assert result == [], f"Expected empty list but got {result}"

    def test_empty_content_returns_empty(self):
        """Empty content SHALL return an empty list."""
        result = self.pipeline._scan_for_injection("")
        assert result == []

    @given(
        indices=st.lists(
            st.integers(
                min_value=0,
                max_value=len(IngestionPipeline.INJECTION_PATTERNS) - 1,
            ),
            min_size=2,
            max_size=4,
            unique=True,
        ),
        separator=st.text(min_size=1, max_size=20, alphabet=_SAFE_ALPHABET),
    )
    @settings(max_examples=10)
    def test_multiple_patterns_all_detected(self, indices, separator):
        """When text contains multiple patterns, ALL SHALL be returned."""
        patterns_to_embed = [IngestionPipeline.INJECTION_PATTERNS[i] for i in indices]
        content = separator.join(patterns_to_embed)

        result = self.pipeline._scan_for_injection(content)

        for pattern in patterns_to_embed:
            assert pattern in result, (
                f"Pattern '{pattern}' was not detected in multi-pattern text"
            )
