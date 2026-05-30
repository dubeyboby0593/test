"""Property-based tests for prompt safety structure (Property 23).

**Validates: Requirements 9.1, 9.5**

Property 23: Prompt Safety Structure
For any AlertMetadata input, the generated prompt SHALL:
(a) enclose all retrieved context within XML-style delimiters,
(b) include a system instruction stating that context is data not instructions,
(c) pass alert metadata as structured JSON parameters rather than concatenating
    raw values into instruction text.
"""

import json

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.models.alert import AlertMetadata
from app.models.rag import ChunkMetadata, RAGResult
from app.models.report import SourceType
from app.services.prompt_builder import build_alert_prompt, build_qa_prompt


# =============================================================================
# Strategies
# =============================================================================

# Strategy for generating valid AlertMetadata with diverse field values
_safe_text = st.text(
    min_size=1,
    max_size=200,
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        blacklist_categories=("Cs", "Cc"),
    ),
)

_alert_metadata_strategy = st.builds(
    AlertMetadata,
    alert_id=st.one_of(st.none(), _safe_text),
    alert_type=st.one_of(st.none(), _safe_text),
    application=_safe_text,  # Always provide application to satisfy minimum field requirement
    action=st.one_of(st.none(), _safe_text),
    destination=st.one_of(st.none(), _safe_text),
    source_ip=st.one_of(st.none(), st.from_regex(r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}", fullmatch=True)),
    user=st.one_of(st.none(), _safe_text),
    zscaler_category=st.one_of(st.none(), _safe_text),
    timestamp=st.one_of(st.none(), st.just("2024-01-15T10:30:00Z")),
    additional_context=st.one_of(
        st.none(),
        st.dictionaries(
            keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
            values=st.one_of(
                st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"), blacklist_categories=("Cs", "Cc"))),
                st.integers(min_value=-100, max_value=100),
                st.booleans(),
            ),
            min_size=0,
            max_size=5,
        ),
    ),
)

# Strategy for generating ChunkMetadata
_chunk_metadata_strategy = st.builds(
    ChunkMetadata,
    source_type=st.sampled_from([SourceType.FICO_INTERNAL, SourceType.PUBLIC]),
    document_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "P"), blacklist_categories=("Cs", "Cc"))),
    section_id=st.one_of(st.none(), st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N")))),
    ingestion_date=st.just("2024-06-15T10:00:00Z"),
    similarity_score=st.floats(min_value=0.3, max_value=1.0),
    content=st.text(min_size=1, max_size=200, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"), blacklist_categories=("Cs", "Cc"))),
)

# Strategy for generating RAGResult with mixed chunks
_rag_result_strategy = st.builds(
    RAGResult,
    fico_chunks=st.lists(_chunk_metadata_strategy.filter(lambda c: c.source_type == SourceType.FICO_INTERNAL), min_size=0, max_size=3),
    public_chunks=st.lists(_chunk_metadata_strategy.filter(lambda c: c.source_type == SourceType.PUBLIC), min_size=0, max_size=3),
    error=st.just(False),
    error_message=st.none(),
    retrieval_latency_ms=st.floats(min_value=0.0, max_value=1000.0),
)

# Simpler RAG result strategy that avoids filtering issues
_fico_chunk_strategy = st.builds(
    ChunkMetadata,
    source_type=st.just(SourceType.FICO_INTERNAL),
    document_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "P"), blacklist_categories=("Cs", "Cc"))),
    section_id=st.one_of(st.none(), st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N")))),
    ingestion_date=st.just("2024-06-15T10:00:00Z"),
    similarity_score=st.floats(min_value=0.3, max_value=1.0),
    content=st.text(min_size=1, max_size=200, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"), blacklist_categories=("Cs", "Cc"))),
)

_public_chunk_strategy = st.builds(
    ChunkMetadata,
    source_type=st.just(SourceType.PUBLIC),
    document_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "P"), blacklist_categories=("Cs", "Cc"))),
    section_id=st.one_of(st.none(), st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N")))),
    ingestion_date=st.just("2024-06-15T10:00:00Z"),
    similarity_score=st.floats(min_value=0.3, max_value=1.0),
    content=st.text(min_size=1, max_size=200, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"), blacklist_categories=("Cs", "Cc"))),
)

_simple_rag_result_strategy = st.builds(
    RAGResult,
    fico_chunks=st.lists(_fico_chunk_strategy, min_size=0, max_size=3),
    public_chunks=st.lists(_public_chunk_strategy, min_size=0, max_size=3),
    error=st.just(False),
    error_message=st.none(),
    retrieval_latency_ms=st.floats(min_value=0.0, max_value=1000.0),
)


# =============================================================================
# Property 23: Prompt Safety Structure
# =============================================================================


class TestPromptSafetyStructure:
    """Property 23: Prompt Safety Structure.

    For any AlertMetadata input, the generated prompt SHALL:
    (a) enclose all retrieved context within XML-style delimiters,
    (b) include a system instruction stating that context is data not instructions,
    (c) pass alert metadata as structured JSON parameters rather than concatenating
        raw values into instruction text.

    **Validates: Requirements 9.1, 9.5**
    """

    # --- (a) XML-style delimiters enclose all retrieved context ---

    @given(
        alert=_alert_metadata_strategy,
        rag_result=_simple_rag_result_strategy,
    )
    @settings(max_examples=10)
    def test_alert_prompt_context_enclosed_in_xml_delimiters(
        self, alert: AlertMetadata, rag_result: RAGResult
    ):
        """For any AlertMetadata, the prompt SHALL enclose retrieved context within XML-style delimiters."""
        result = build_alert_prompt(alert, rag_result)

        # The result must have the expected structure
        assert "system" in result
        assert "messages" in result
        assert "max_tokens" in result

        # Get the user message content
        user_message = result["messages"][0]["content"]

        # Context must be enclosed in <context>...</context> tags
        assert "<context>" in user_message
        assert "</context>" in user_message

        # The <context> tag must appear before </context>
        context_open_idx = user_message.index("<context>")
        context_close_idx = user_message.index("</context>")
        assert context_open_idx < context_close_idx

    @given(
        alert=_alert_metadata_strategy,
        rag_result=_simple_rag_result_strategy,
    )
    @settings(max_examples=10)
    def test_alert_prompt_metadata_enclosed_in_xml_delimiters(
        self, alert: AlertMetadata, rag_result: RAGResult
    ):
        """For any AlertMetadata, the alert metadata SHALL be enclosed within <alert_metadata> XML-style delimiters."""
        result = build_alert_prompt(alert, rag_result)

        user_message = result["messages"][0]["content"]

        # Alert metadata must be enclosed in <alert_metadata>...</alert_metadata> tags
        assert "<alert_metadata>" in user_message
        assert "</alert_metadata>" in user_message

        # The <alert_metadata> tag must appear before </alert_metadata>
        meta_open_idx = user_message.index("<alert_metadata>")
        meta_close_idx = user_message.index("</alert_metadata>")
        assert meta_open_idx < meta_close_idx

    # --- (b) System instruction states context is data not instructions ---

    @given(
        alert=_alert_metadata_strategy,
        rag_result=_simple_rag_result_strategy,
    )
    @settings(max_examples=10)
    def test_alert_prompt_system_instruction_declares_context_as_data(
        self, alert: AlertMetadata, rag_result: RAGResult
    ):
        """For any AlertMetadata, the system instruction SHALL state that context is data not instructions."""
        result = build_alert_prompt(alert, rag_result)

        system_instruction = result["system"]

        # System instruction must contain language indicating context is data
        assert "retrieved data" in system_instruction.lower() or "reference material" in system_instruction.lower()
        # System instruction must state context is not instructions
        assert "not as instructions" in system_instruction.lower() or "not instructions" in system_instruction.lower()

    @given(
        question=st.text(
            min_size=1,
            max_size=200,
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                blacklist_categories=("Cs", "Cc"),
            ),
        ),
        rag_result=_simple_rag_result_strategy,
        max_tokens=st.integers(min_value=100, max_value=1500),
    )
    @settings(max_examples=15)
    def test_qa_prompt_system_instruction_declares_context_as_data(
        self, question: str, rag_result: RAGResult, max_tokens: int
    ):
        """For any Q&A prompt, the system instruction SHALL state that context is data not instructions."""
        result = build_qa_prompt(question, rag_result, max_tokens)

        system_instruction = result["system"]

        # System instruction must contain language indicating context is data
        assert "retrieved data" in system_instruction.lower() or "reference material" in system_instruction.lower()
        # System instruction must state context is not instructions
        assert "not as instructions" in system_instruction.lower() or "not instructions" in system_instruction.lower()

    # --- (c) Alert metadata as structured JSON, not concatenated into system instruction ---

    @given(
        alert=_alert_metadata_strategy,
        rag_result=_simple_rag_result_strategy,
    )
    @settings(max_examples=10)
    def test_alert_raw_values_not_in_system_instruction(
        self, alert: AlertMetadata, rag_result: RAGResult
    ):
        """For any AlertMetadata, raw alert field values SHALL NOT appear in the system instruction text."""
        result = build_alert_prompt(alert, rag_result)

        system_instruction = result["system"]

        # Check that non-null alert field values do not appear in the system instruction
        # (they should only appear in the user message within <alert_metadata> tags as JSON)
        fields_to_check = [
            alert.application,
            alert.action,
            alert.destination,
            alert.user,
            alert.source_ip,
            alert.alert_type,
            alert.zscaler_category,
        ]

        for field_value in fields_to_check:
            if field_value is not None and len(field_value) > 3:
                # Only check values with meaningful length to avoid false positives
                # from common short words that might appear in instruction text
                assert field_value not in system_instruction, (
                    f"Raw alert value '{field_value}' found in system instruction. "
                    f"Alert metadata should be passed as structured JSON parameters, "
                    f"not concatenated into instruction text."
                )

    @given(
        alert=_alert_metadata_strategy,
        rag_result=_simple_rag_result_strategy,
    )
    @settings(max_examples=10)
    def test_alert_metadata_serialized_as_json_in_user_message(
        self, alert: AlertMetadata, rag_result: RAGResult
    ):
        """For any AlertMetadata, the alert data SHALL be serialized as JSON within the user message."""
        result = build_alert_prompt(alert, rag_result)

        user_message = result["messages"][0]["content"]

        # Extract content between <alert_metadata> tags
        meta_start = user_message.index("<alert_metadata>") + len("<alert_metadata>")
        meta_end = user_message.index("</alert_metadata>")
        alert_content = user_message[meta_start:meta_end].strip()

        # The content between tags must be valid JSON
        parsed_json = json.loads(alert_content)
        assert isinstance(parsed_json, dict)

        # The JSON must contain the non-null fields from the alert
        alert_dict = alert.model_dump(exclude_none=True)
        for key, value in alert_dict.items():
            assert key in parsed_json, (
                f"Alert field '{key}' not found in serialized JSON parameters"
            )

    @given(
        question=st.text(
            min_size=1,
            max_size=200,
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                blacklist_categories=("Cs", "Cc"),
            ),
        ),
        rag_result=_simple_rag_result_strategy,
        max_tokens=st.integers(min_value=100, max_value=1500),
    )
    @settings(max_examples=15)
    def test_qa_prompt_context_enclosed_in_xml_delimiters(
        self, question: str, rag_result: RAGResult, max_tokens: int
    ):
        """For any Q&A prompt, the retrieved context SHALL be enclosed within XML-style delimiters."""
        result = build_qa_prompt(question, rag_result, max_tokens)

        user_message = result["messages"][0]["content"]

        # Context must be enclosed in <context>...</context> tags
        assert "<context>" in user_message
        assert "</context>" in user_message

        # The <context> tag must appear before </context>
        context_open_idx = user_message.index("<context>")
        context_close_idx = user_message.index("</context>")
        assert context_open_idx < context_close_idx
