"""Property-based tests for input validation (Properties 13, 14).

**Validates: Requirements 3.8, 4.10, 7.5, 7.6, 9.2, 9.3, 9.6**
"""

import json
import string

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from pydantic import ValidationError

from app.models.query import QueryRequest
from app.models.alert import AlertMetadata


# =============================================================================
# Property 13: Query Input Validation
# =============================================================================


class TestQueryInputValidation:
    """Property 13: Query Input Validation.

    For any string of length 0 (empty) or length greater than 1000 characters,
    the system SHALL reject the request with a validation error. For any string
    of length 1 to 1000 (inclusive), the system SHALL accept the request for
    processing.

    **Validates: Requirements 3.8, 7.5**
    """

    @given(
        question=st.text(
            min_size=1,
            max_size=1000,
            alphabet=st.characters(
                blacklist_categories=("Cs",)  # Exclude surrogates
            ),
        )
    )
    @settings(max_examples=10)
    def test_valid_length_queries_are_accepted(self, question: str):
        """Any string of length 1 to 1000 (inclusive) SHALL be accepted."""
        request = QueryRequest(question=question)
        assert request.question == question
        assert 1 <= len(request.question) <= 1000

    @given(
        question=st.text(
            min_size=1001,
            max_size=2000,
            alphabet=st.characters(
                blacklist_categories=("Cs",)
            ),
        )
    )
    @settings(max_examples=15)
    def test_overlength_queries_are_rejected(self, question: str):
        """Any string of length greater than 1000 SHALL be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(question=question)
        # Verify the error mentions the question field
        errors = exc_info.value.errors()
        assert any(
            "question" in str(e.get("loc", ""))
            for e in errors
        )

    def test_empty_string_is_rejected(self):
        """An empty string (length 0) SHALL be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(question="")
        errors = exc_info.value.errors()
        assert any(
            "question" in str(e.get("loc", ""))
            for e in errors
        )

    @given(
        length=st.integers(min_value=1001, max_value=5000)
    )
    @settings(max_examples=10)
    def test_various_overlength_sizes_rejected(self, length: int):
        """Strings of various lengths > 1000 are all rejected."""
        question = "a" * length
        with pytest.raises(ValidationError):
            QueryRequest(question=question)

    @given(
        length=st.integers(min_value=1, max_value=1000)
    )
    @settings(max_examples=10)
    def test_boundary_lengths_accepted(self, length: int):
        """Strings of any length from 1 to 1000 are accepted."""
        question = "x" * length
        request = QueryRequest(question=question)
        assert len(request.question) == length


# =============================================================================
# Property 14: Alert Metadata Input Validation
# =============================================================================


# Strategies for generating test data

# ASCII control characters U+0000-U+001F excluding U+000A (newline) and U+000D (CR)
_DISALLOWED_CONTROL_CHARS = [
    chr(c) for c in range(0x00, 0x20) if c not in (0x0A, 0x0D)
]

# Valid simple values for additional_context
_simple_value_strategy = st.one_of(
    st.text(min_size=0, max_size=100, alphabet=st.characters(blacklist_categories=("Cs",))),
    st.integers(min_value=-1000, max_value=1000),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000),
    st.booleans(),
)


class TestAlertMetadataInputValidation:
    """Property 14: Alert Metadata Input Validation.

    For any AlertMetadata that:
    (a) is missing all three of application, action, and alert_type, OR
    (b) contains string fields with ASCII control characters U+0000-U+001F
        (except U+000A and U+000D), OR
    (c) has an additional_context field exceeding 3 nesting levels, 5000 total
        serialized characters, 20 key-value pairs, or values that are not
        string/number/boolean, OR
    (d) has individual additional_context values exceeding 500 characters
    — the system SHALL reject the request with a validation error identifying
    each non-conforming field.

    **Validates: Requirements 4.10, 7.6, 9.2, 9.3, 9.6**
    """

    # --- (a) Missing minimum required fields ---

    @given(
        destination=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
        source_ip=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
        user=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
    )
    @settings(max_examples=10)
    def test_missing_all_minimum_fields_rejected(
        self, destination, source_ip, user
    ):
        """AlertMetadata missing all of application, action, alert_type SHALL be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            AlertMetadata(
                application=None,
                action=None,
                alert_type=None,
                destination=destination,
                source_ip=source_ip,
                user=user,
            )
        error_str = str(exc_info.value)
        assert "application" in error_str or "action" in error_str or "alert_type" in error_str

    @given(
        app=st.one_of(
            st.just(("app_val", None, None)),
            st.just((None, "action_val", None)),
            st.just((None, None, "alert_val")),
            st.just(("app_val", "action_val", None)),
            st.just(("app_val", None, "alert_val")),
            st.just((None, "action_val", "alert_val")),
            st.just(("app_val", "action_val", "alert_val")),
        )
    )
    @settings(max_examples=10)
    def test_at_least_one_minimum_field_accepted(self, app):
        """AlertMetadata with at least one of application, action, alert_type SHALL be accepted."""
        application, action, alert_type = app
        alert = AlertMetadata(
            application=application,
            action=action,
            alert_type=alert_type,
        )
        assert any([alert.application, alert.action, alert.alert_type])

    # --- (b) Control characters in string fields ---

    @given(
        control_char=st.sampled_from(_DISALLOWED_CONTROL_CHARS),
        field_name=st.sampled_from([
            "alert_type", "application", "action", "destination",
            "source_ip", "user", "zscaler_category",
        ]),
    )
    @settings(max_examples=15)
    def test_control_characters_in_fields_rejected(self, control_char, field_name):
        """String fields with ASCII control chars U+0000-U+001F (except newline/CR) SHALL be rejected."""
        # Build kwargs with a valid minimum field plus the offending field
        kwargs = {"application": "valid_app"}  # ensure minimum field requirement met
        # Insert control char into the target field
        kwargs[field_name] = f"value{control_char}here"

        with pytest.raises(ValidationError) as exc_info:
            AlertMetadata(**kwargs)
        error_str = str(exc_info.value)
        assert "control character" in error_str.lower() or field_name in error_str

    @given(
        text_value=st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(
                blacklist_categories=("Cs", "Cc"),  # No surrogates, no control chars
            ),
        )
    )
    @settings(max_examples=10)
    def test_fields_without_control_chars_accepted(self, text_value):
        """String fields without control characters SHALL be accepted."""
        # Ensure no control chars slipped through (except allowed newline/CR)
        assume(not any(
            chr(c) in text_value
            for c in range(0x00, 0x20)
            if c not in (0x0A, 0x0D)
        ))
        alert = AlertMetadata(
            application=text_value,
        )
        assert alert.application == text_value

    # --- (c) additional_context constraints ---

    @given(
        num_pairs=st.integers(min_value=21, max_value=30)
    )
    @settings(max_examples=10)
    def test_additional_context_exceeding_20_pairs_rejected(self, num_pairs):
        """additional_context with more than 20 key-value pairs SHALL be rejected."""
        context = {f"key_{i}": "val" for i in range(num_pairs)}
        with pytest.raises(ValidationError) as exc_info:
            AlertMetadata(
                application="test_app",
                additional_context=context,
            )
        error_str = str(exc_info.value)
        assert "20" in error_str or "key-value" in error_str.lower() or "additional_context" in error_str

    @given(
        num_pairs=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=10)
    def test_additional_context_within_20_pairs_accepted(self, num_pairs):
        """additional_context with 20 or fewer key-value pairs SHALL be accepted."""
        context = {f"k{i}": "v" for i in range(num_pairs)}
        alert = AlertMetadata(
            application="test_app",
            additional_context=context,
        )
        assert len(alert.additional_context) == num_pairs

    def test_additional_context_exceeding_3_nesting_levels_rejected(self):
        """additional_context exceeding 3 nesting levels SHALL be rejected."""
        # 4 levels deep: level1 -> level2 -> level3 -> level4
        context = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": "too deep"
                    }
                }
            }
        }
        with pytest.raises(ValidationError) as exc_info:
            AlertMetadata(
                application="test_app",
                additional_context=context,
            )
        error_str = str(exc_info.value)
        assert "nesting" in error_str.lower() or "depth" in error_str.lower() or "additional_context" in error_str

    def test_additional_context_within_3_nesting_levels_accepted(self):
        """additional_context within 3 nesting levels SHALL be accepted."""
        # Exactly 3 levels: level1 -> level2 -> level3 (value)
        context = {
            "level1": {
                "level2": {
                    "level3": "ok"
                }
            }
        }
        alert = AlertMetadata(
            application="test_app",
            additional_context=context,
        )
        assert alert.additional_context == context

    @given(
        char_count=st.integers(min_value=5001, max_value=6000)
    )
    @settings(max_examples=10)
    def test_additional_context_exceeding_5000_serialized_chars_rejected(self, char_count):
        """additional_context exceeding 5000 total serialized characters SHALL be rejected."""
        # Create a context that when serialized exceeds 5000 chars
        # Account for JSON overhead: {"key": "value"} adds ~10 chars
        value_length = char_count - 20  # subtract overhead for key and JSON structure
        context = {"k": "x" * max(value_length, 4900)}
        serialized = json.dumps(context)
        assume(len(serialized) > 5000)

        with pytest.raises(ValidationError) as exc_info:
            AlertMetadata(
                application="test_app",
                additional_context=context,
            )
        error_str = str(exc_info.value)
        assert "5000" in error_str or "character" in error_str.lower() or "additional_context" in error_str

    def test_additional_context_with_invalid_value_types_rejected(self):
        """additional_context with non-string/number/boolean values SHALL be rejected."""
        # Lists are not allowed
        context = {"key": [1, 2, 3]}
        with pytest.raises(ValidationError) as exc_info:
            AlertMetadata(
                application="test_app",
                additional_context=context,
            )
        error_str = str(exc_info.value)
        assert "type" in error_str.lower() or "additional_context" in error_str

    @given(
        value=st.one_of(
            st.lists(st.integers(), min_size=1, max_size=3),
            st.tuples(st.integers()),
        )
    )
    @settings(max_examples=10)
    def test_additional_context_non_primitive_values_rejected(self, value):
        """additional_context values that are not string/number/boolean SHALL be rejected."""
        context = {"key": value}
        with pytest.raises(ValidationError) as exc_info:
            AlertMetadata(
                application="test_app",
                additional_context=context,
            )
        error_str = str(exc_info.value)
        assert "type" in error_str.lower() or "additional_context" in error_str

    # --- (d) Individual additional_context values exceeding 500 characters ---

    @given(
        value_length=st.integers(min_value=501, max_value=800)
    )
    @settings(max_examples=10)
    def test_additional_context_value_exceeding_500_chars_rejected(self, value_length):
        """Individual additional_context values exceeding 500 characters SHALL be rejected."""
        context = {"key": "a" * value_length}
        with pytest.raises(ValidationError) as exc_info:
            AlertMetadata(
                application="test_app",
                additional_context=context,
            )
        error_str = str(exc_info.value)
        assert "500" in error_str or "character" in error_str.lower() or "additional_context" in error_str

    @given(
        value_length=st.integers(min_value=1, max_value=500)
    )
    @settings(max_examples=10)
    def test_additional_context_value_within_500_chars_accepted(self, value_length):
        """Individual additional_context values within 500 characters SHALL be accepted."""
        context = {"key": "a" * value_length}
        alert = AlertMetadata(
            application="test_app",
            additional_context=context,
        )
        assert alert.additional_context["key"] == "a" * value_length

    # --- Combined valid AlertMetadata ---

    @given(
        application=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                blacklist_categories=("Cs", "Cc"),
            ),
        ),
        action=st.one_of(
            st.none(),
            st.text(
                min_size=1,
                max_size=50,
                alphabet=st.characters(
                    whitelist_categories=("L", "N", "P", "Z"),
                    blacklist_categories=("Cs", "Cc"),
                ),
            ),
        ),
    )
    @settings(max_examples=10)
    def test_valid_alert_metadata_accepted(self, application, action):
        """Valid AlertMetadata with proper fields SHALL be accepted."""
        # Filter out any control chars that might slip through
        assume(not any(
            chr(c) in application
            for c in range(0x00, 0x20)
            if c not in (0x0A, 0x0D)
        ))
        if action:
            assume(not any(
                chr(c) in action
                for c in range(0x00, 0x20)
                if c not in (0x0A, 0x0D)
            ))

        alert = AlertMetadata(
            application=application,
            action=action,
        )
        assert alert.application == application
