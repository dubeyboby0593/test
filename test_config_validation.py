"""
Property-based tests for configuration validation at startup (Property 27).

**Validates: Requirements 10.4, 10.5, 10.7**

Property 27: Configuration Validation at Startup
- For any set of environment variables where at least one required variable
  (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BEDROCK_KB_ID) is missing or empty,
  the application SHALL fail at startup with an error message naming each
  missing variable.
- For any numeric optional variable set to a non-numeric value or a value <= 0,
  the application SHALL fail at startup identifying the variable and invalid value.
"""

import os
import sys
from io import StringIO
from unittest.mock import patch

import pytest
from hypothesis import given, strategies as st, assume, settings as hyp_settings
from hypothesis import HealthCheck

from app.config import Settings, validate_settings, REQUIRED_VARIABLES


# --- Constants ---

# Required variables that the config module validates as must-not-be-empty
# These are the variables that default to "" in Settings and are checked by
# validate_settings(). They map to the REQUIRED_VARIABLES list in config.py.
VALIDATED_REQUIRED_VARS = [var.upper() for var in REQUIRED_VARIABLES]

# Optional numeric variables that must be positive integers
OPTIONAL_NUMERIC_VARS = [
    "RAG_TOP_K",
    "CACHE_TTL_SECONDS",
    "MAX_FILE_SIZE_MB",
    "MAX_RESPONSE_TOKENS",
    "MAX_PAYLOAD_SIZE_KB",
    "AUDIT_RETENTION_DAYS",
]

# A valid baseline environment for all required variables
VALID_ENV = {
    "AWS_REGION": "us-east-1",
    "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
    "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "BEDROCK_MODEL_ID": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "TITAN_EMBED_MODEL_ID": "amazon.titan-embed-text-v2:0",
    "BEDROCK_KB_ID": "KB12345678",
    "S3_BUCKET_NAME": "fico-soc-policy-kb",
}


# --- Strategies ---

# Strategy for empty or whitespace-only strings
empty_or_whitespace = st.one_of(
    st.just(""),
    st.text(alphabet=" \t", min_size=1, max_size=10),
)

# Strategy for non-numeric strings (cannot be parsed as int)
non_numeric_strings = st.text(
    alphabet=st.characters(whitelist_categories=("L", "P", "S"),
                           min_codepoint=32, max_codepoint=126),
    min_size=1,
    max_size=20,
).filter(lambda s: not _is_valid_int(s))

# Strategy for non-positive integers (zero or negative)
non_positive_integers = st.integers(max_value=0)

# Strategy for positive integers (valid numeric values)
positive_integers = st.integers(min_value=1, max_value=10000)


def _is_valid_int(s: str) -> bool:
    """Check if a string can be parsed as a valid integer."""
    try:
        int(s.strip())
        return True
    except (ValueError, TypeError):
        return False


# --- Property Tests ---

class TestProperty27MissingRequiredVariables:
    """
    Property 27 (Part 1): For any set of environment variables where at least
    one required variable is missing or empty, the application SHALL fail at
    startup with an error message naming each missing variable.

    **Validates: Requirements 10.4**
    """

    @given(
        vars_to_remove=st.lists(
            st.sampled_from(VALIDATED_REQUIRED_VARS),
            min_size=1,
            max_size=len(VALIDATED_REQUIRED_VARS),
            unique=True,
        )
    )
    @hyp_settings(max_examples=10)
    def test_missing_required_vars_cause_startup_failure(self, vars_to_remove):
        """
        For any non-empty subset of required variables that are removed from
        the environment, the application SHALL fail with SystemExit.

        **Validates: Requirements 10.4**
        """
        # Build env with some required vars removed
        env = {k: v for k, v in VALID_ENV.items() if k not in vars_to_remove}

        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)
            with pytest.raises(SystemExit):
                validate_settings(settings)

    @given(
        var_to_empty=st.sampled_from(VALIDATED_REQUIRED_VARS),
        empty_value=empty_or_whitespace,
    )
    @hyp_settings(max_examples=10)
    def test_empty_required_vars_cause_startup_failure(self, var_to_empty, empty_value):
        """
        For any required variable set to an empty or whitespace-only value,
        the application SHALL fail with SystemExit.

        **Validates: Requirements 10.4**
        """
        env = {**VALID_ENV, var_to_empty: empty_value}

        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)
            with pytest.raises(SystemExit):
                validate_settings(settings)

    @given(
        vars_to_remove=st.lists(
            st.sampled_from(VALIDATED_REQUIRED_VARS),
            min_size=1,
            max_size=len(VALIDATED_REQUIRED_VARS),
            unique=True,
        )
    )
    @hyp_settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_error_message_names_missing_variables(self, vars_to_remove, capsys):
        """
        The error message SHALL name each missing variable.

        **Validates: Requirements 10.4**
        """
        # Build env with some required vars removed
        env = {k: v for k, v in VALID_ENV.items() if k not in vars_to_remove}

        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)
            with pytest.raises(SystemExit):
                validate_settings(settings)

            captured = capsys.readouterr()
            # Each missing variable should be named in the error output
            for var in vars_to_remove:
                assert var in captured.err, (
                    f"Expected '{var}' to be named in error output, "
                    f"got: {captured.err}"
                )


class TestProperty27InvalidNumericVariables:
    """
    Property 27 (Part 2): For any numeric optional variable set to a non-numeric
    value or a value <= 0, the application SHALL fail at startup identifying
    the variable and invalid value.

    **Validates: Requirements 10.7**
    """

    @given(
        var_name=st.sampled_from(OPTIONAL_NUMERIC_VARS),
        invalid_value=non_positive_integers,
    )
    @hyp_settings(max_examples=10)
    def test_non_positive_numeric_vars_cause_startup_failure(self, var_name, invalid_value):
        """
        For any optional numeric variable set to a value <= 0,
        the application SHALL fail at startup.

        **Validates: Requirements 10.7**
        """
        env = {**VALID_ENV, var_name: str(invalid_value)}

        with patch.dict(os.environ, env, clear=True):
            with pytest.raises((SystemExit, Exception)):
                settings = Settings(_env_file=None)
                validate_settings(settings)

    @given(
        var_name=st.sampled_from(OPTIONAL_NUMERIC_VARS),
        invalid_value=non_numeric_strings,
    )
    @hyp_settings(max_examples=10)
    def test_non_numeric_string_vars_cause_startup_failure(self, var_name, invalid_value):
        """
        For any optional numeric variable set to a non-numeric string,
        the application SHALL fail at startup.

        **Validates: Requirements 10.7**
        """
        env = {**VALID_ENV, var_name: invalid_value}

        with patch.dict(os.environ, env, clear=True):
            with pytest.raises((SystemExit, Exception)):
                settings = Settings(_env_file=None)
                validate_settings(settings)


class TestProperty27ValidConfiguration:
    """
    Complementary property: For any valid configuration (all required vars
    present and non-empty, all numeric vars positive), startup SHALL succeed.

    **Validates: Requirements 10.5**
    """

    @given(
        rag_top_k=positive_integers,
        cache_ttl=positive_integers,
        max_file_size=positive_integers,
        max_tokens=positive_integers,
        max_payload=positive_integers,
        audit_days=positive_integers,
    )
    @hyp_settings(max_examples=10)
    def test_valid_config_succeeds(self, rag_top_k, cache_ttl, max_file_size,
                                   max_tokens, max_payload, audit_days):
        """
        For any valid set of positive numeric values combined with all required
        variables present, the application SHALL start successfully.

        **Validates: Requirements 10.5**
        """
        env = {
            **VALID_ENV,
            "RAG_TOP_K": str(rag_top_k),
            "CACHE_TTL_SECONDS": str(cache_ttl),
            "MAX_FILE_SIZE_MB": str(max_file_size),
            "MAX_RESPONSE_TOKENS": str(max_tokens),
            "MAX_PAYLOAD_SIZE_KB": str(max_payload),
            "AUDIT_RETENTION_DAYS": str(audit_days),
        }

        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)
            result = validate_settings(settings)
            assert result is not None
            assert result.rag_top_k == rag_top_k
            assert result.cache_ttl_seconds == cache_ttl
            assert result.max_file_size_mb == max_file_size
            assert result.max_response_tokens == max_tokens
            assert result.max_payload_size_kb == max_payload
            assert result.audit_retention_days == audit_days
