"""Property-based tests for prompt injection pattern detection (Property 4).

**Validates: Requirements 1.10, 9.4**

Property 4: Prompt Injection Pattern Detection
For any document text containing at least one substring from the configured
injection pattern list (case-insensitive match), the scanner SHALL return a
non-empty list of matched patterns. For any document text containing none of
the configured patterns, the scanner SHALL return an empty list.
"""

import string

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.security.input_validator import InputValidator, DEFAULT_INJECTION_PATTERNS


# =============================================================================
# Strategies
# =============================================================================

# Strategy for generating random case variations of a string
def random_case(s: str) -> st.SearchStrategy[str]:
    """Generate a random case variation of the given string."""
    return st.builds(
        lambda chars: "".join(chars),
        st.tuples(
            *[st.sampled_from([c.lower(), c.upper()]) if c.isalpha() else st.just(c) for c in s]
        ),
    )


# Strategy for safe text that won't accidentally contain any default patterns
_SAFE_ALPHABET = st.sampled_from(string.digits + ".,;!?@#$%^&*()[]{}=+/~`")

# Strategy for custom pattern lists (ASCII only to avoid Unicode case-folding
# edge cases like ß -> SS which don't round-trip with .lower())
_custom_pattern_alphabet = st.sampled_from(
    string.ascii_letters + string.digits + " _-:.<>|#"
)
_custom_pattern_strategy = st.lists(
    st.text(min_size=3, max_size=30, alphabet=_custom_pattern_alphabet),
    min_size=1,
    max_size=5,
)


# =============================================================================
# Property 4: Prompt Injection Pattern Detection
# =============================================================================


class TestPromptInjectionPatternDetection:
    """Property 4: Prompt Injection Pattern Detection.

    For any document text containing at least one substring from the configured
    injection pattern list (case-insensitive match), the scanner SHALL return a
    non-empty list of matched patterns. For any document text containing none of
    the configured patterns, the scanner SHALL return an empty list.

    **Validates: Requirements 1.10, 9.4**
    """

    def setup_method(self):
        """Set up InputValidator instance for each test."""
        self.validator = InputValidator()

    # --- Text containing patterns MUST be detected ---

    @given(
        pattern_idx=st.integers(min_value=0, max_value=len(DEFAULT_INJECTION_PATTERNS) - 1),
        prefix=st.text(min_size=0, max_size=50, alphabet=_SAFE_ALPHABET),
        suffix=st.text(min_size=0, max_size=50, alphabet=_SAFE_ALPHABET),
    )
    @settings(max_examples=10)
    def test_text_containing_default_pattern_detected(self, pattern_idx, prefix, suffix):
        """Any text containing a default injection pattern SHALL return non-empty matches."""
        pattern = DEFAULT_INJECTION_PATTERNS[pattern_idx]
        content = prefix + pattern + suffix

        result = self.validator.scan_injection_patterns(content)

        assert len(result) > 0, f"Pattern '{pattern}' was not detected in '{content}'"
        assert pattern in result

    @given(
        pattern_idx=st.integers(min_value=0, max_value=len(DEFAULT_INJECTION_PATTERNS) - 1),
        prefix=st.text(min_size=0, max_size=50, alphabet=_SAFE_ALPHABET),
        suffix=st.text(min_size=0, max_size=50, alphabet=_SAFE_ALPHABET),
        data=st.data(),
    )
    @settings(max_examples=10)
    def test_case_insensitive_detection(self, pattern_idx, prefix, suffix, data):
        """Patterns SHALL be detected regardless of case (case-insensitive match)."""
        pattern = DEFAULT_INJECTION_PATTERNS[pattern_idx]
        # Generate a random case variation of the pattern
        case_varied = data.draw(random_case(pattern))
        content = prefix + case_varied + suffix

        result = self.validator.scan_injection_patterns(content)

        assert len(result) > 0, (
            f"Case-varied pattern '{case_varied}' (original: '{pattern}') "
            f"was not detected in '{content}'"
        )
        assert pattern in result

    # --- Text without patterns MUST return empty list ---

    @given(
        content=st.text(min_size=0, max_size=200, alphabet=_SAFE_ALPHABET),
    )
    @settings(max_examples=10)
    def test_text_without_patterns_returns_empty(self, content):
        """Text containing none of the configured patterns SHALL return an empty list."""
        # Verify content doesn't accidentally contain any pattern
        content_lower = content.lower()
        for pattern in DEFAULT_INJECTION_PATTERNS:
            assume(pattern.lower() not in content_lower)

        result = self.validator.scan_injection_patterns(content)

        assert result == [], f"Expected empty list but got {result} for content '{content}'"

    @given(
        content=st.text(
            min_size=1,
            max_size=500,
            alphabet=st.characters(
                whitelist_categories=("L", "N", "Z"),
                blacklist_categories=("Cs",),
            ),
        ),
    )
    @settings(max_examples=10)
    def test_arbitrary_text_without_patterns_returns_empty(self, content):
        """Arbitrary text not containing any pattern SHALL return an empty list."""
        content_lower = content.lower()
        for pattern in DEFAULT_INJECTION_PATTERNS:
            assume(pattern.lower() not in content_lower)

        result = self.validator.scan_injection_patterns(content)

        assert result == [], f"Expected empty list but got {result} for content '{content}'"

    # --- Custom pattern lists ---

    @given(
        custom_patterns=_custom_pattern_strategy,
        pattern_choice=st.integers(min_value=0, max_value=100),
        prefix=st.text(min_size=0, max_size=30, alphabet=_SAFE_ALPHABET),
        suffix=st.text(min_size=0, max_size=30, alphabet=_SAFE_ALPHABET),
    )
    @settings(max_examples=10)
    def test_custom_patterns_detected_when_present(
        self, custom_patterns, pattern_choice, prefix, suffix
    ):
        """Custom pattern lists SHALL detect patterns when present in text."""
        # Pick one pattern from the custom list
        idx = pattern_choice % len(custom_patterns)
        chosen_pattern = custom_patterns[idx]
        assume(len(chosen_pattern.strip()) > 0)

        content = prefix + chosen_pattern + suffix

        result = self.validator.scan_injection_patterns(content, patterns=custom_patterns)

        assert len(result) > 0, (
            f"Custom pattern '{chosen_pattern}' was not detected in '{content}'"
        )
        assert chosen_pattern in result

    @given(
        custom_patterns=_custom_pattern_strategy,
        content=st.text(min_size=0, max_size=200, alphabet=_SAFE_ALPHABET),
    )
    @settings(max_examples=10)
    def test_custom_patterns_empty_when_absent(self, custom_patterns, content):
        """Custom pattern lists SHALL return empty when no patterns are in text."""
        content_lower = content.lower()
        for pattern in custom_patterns:
            assume(pattern.lower() not in content_lower)

        result = self.validator.scan_injection_patterns(content, patterns=custom_patterns)

        assert result == [], (
            f"Expected empty list but got {result} for content '{content}' "
            f"with patterns {custom_patterns}"
        )

    @given(
        custom_patterns=_custom_pattern_strategy,
        data=st.data(),
        prefix=st.text(min_size=0, max_size=30, alphabet=_SAFE_ALPHABET),
        suffix=st.text(min_size=0, max_size=30, alphabet=_SAFE_ALPHABET),
    )
    @settings(max_examples=10)
    def test_custom_patterns_case_insensitive(self, custom_patterns, data, prefix, suffix):
        """Custom patterns SHALL be matched case-insensitively."""
        # Pick a pattern and vary its case
        idx = data.draw(st.integers(min_value=0, max_value=len(custom_patterns) - 1))
        chosen_pattern = custom_patterns[idx]
        assume(len(chosen_pattern.strip()) > 0)
        # Only vary case for alphabetic characters
        assume(any(c.isalpha() for c in chosen_pattern))

        case_varied = data.draw(random_case(chosen_pattern))
        content = prefix + case_varied + suffix

        result = self.validator.scan_injection_patterns(content, patterns=custom_patterns)

        assert len(result) > 0, (
            f"Case-varied custom pattern '{case_varied}' (original: '{chosen_pattern}') "
            f"was not detected"
        )
        assert chosen_pattern in result

    # --- Multiple patterns in same text ---

    @given(
        indices=st.lists(
            st.integers(min_value=0, max_value=len(DEFAULT_INJECTION_PATTERNS) - 1),
            min_size=2,
            max_size=4,
            unique=True,
        ),
        separator=st.text(min_size=1, max_size=20, alphabet=_SAFE_ALPHABET),
    )
    @settings(max_examples=15)
    def test_multiple_patterns_all_detected(self, indices, separator):
        """When text contains multiple patterns, ALL matched patterns SHALL be returned."""
        patterns_to_embed = [DEFAULT_INJECTION_PATTERNS[i] for i in indices]
        content = separator.join(patterns_to_embed)

        result = self.validator.scan_injection_patterns(content)

        for pattern in patterns_to_embed:
            assert pattern in result, (
                f"Pattern '{pattern}' was not detected in multi-pattern text"
            )

    # --- Empty content ---

    def test_empty_content_returns_empty(self):
        """Empty content SHALL return an empty list."""
        result = self.validator.scan_injection_patterns("")
        assert result == []

    def test_empty_content_with_custom_patterns_returns_empty(self):
        """Empty content with custom patterns SHALL return an empty list."""
        result = self.validator.scan_injection_patterns("", patterns=["test", "pattern"])
        assert result == []

    # --- Pattern embedded in larger words ---

    @given(
        pattern_idx=st.integers(min_value=0, max_value=len(DEFAULT_INJECTION_PATTERNS) - 1),
        word_prefix=st.text(min_size=1, max_size=20, alphabet=st.sampled_from(string.ascii_letters)),
        word_suffix=st.text(min_size=1, max_size=20, alphabet=st.sampled_from(string.ascii_letters)),
    )
    @settings(max_examples=15)
    def test_pattern_as_substring_still_detected(self, pattern_idx, word_prefix, word_suffix):
        """Patterns embedded within larger text (substring match) SHALL still be detected."""
        pattern = DEFAULT_INJECTION_PATTERNS[pattern_idx]
        # Embed pattern within other text without spaces separating
        content = word_prefix + pattern + word_suffix

        result = self.validator.scan_injection_patterns(content)

        assert pattern in result, (
            f"Pattern '{pattern}' embedded in '{content}' was not detected"
        )
