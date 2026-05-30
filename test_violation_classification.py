"""Property-based tests for violation classification (Properties 18, 19, 20, 21).

**Validates: Requirements 5.2, 5.3, 5.4, 5.6**
"""

import json
from typing import Optional

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.models.alert import AlertMetadata
from app.models.rag import ChunkMetadata
from app.models.report import Severity, SourceType
from app.services.violation_classifier import ViolationClassifier, _ALERT_METADATA_FIELDS


# =============================================================================
# Strategies
# =============================================================================

# Strategy for similarity scores in [0.0, 1.0]
similarity_scores = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Strategy for confidence scores in [0, 100]
confidence_scores = st.integers(min_value=0, max_value=100)

# Strategy for number of fields present (0-10)
# We generate AlertMetadata with a controlled number of fields present
fields_present_count = st.integers(min_value=1, max_value=10)


def _build_alert_with_n_fields(n: int) -> AlertMetadata:
    """Build an AlertMetadata with exactly n fields present out of 10.

    The 10 fields are: alert_id, alert_type, application, action, destination,
    source_ip, user, zscaler_category, timestamp, additional_context.

    At least one of application, action, or alert_type must be present for
    AlertMetadata validation to pass.
    """
    all_fields = [
        ("alert_id", "ALERT-001"),
        ("alert_type", "DLP"),
        ("application", "Dropbox"),
        ("action", "upload"),
        ("destination", "https://example.com"),
        ("source_ip", "192.168.1.1"),
        ("user", "analyst@fico.com"),
        ("zscaler_category", "Cloud Storage"),
        ("timestamp", "2024-01-15T10:30:00Z"),
        ("additional_context", {"key": "value"}),
    ]

    # Always include at least one of application, action, or alert_type
    # to satisfy the model validator
    fields_to_set = {}

    # Ensure we have at least one required field
    # Pick from the first n fields, but ensure at least one of indices 1,2,3
    # (alert_type, application, action) is included
    selected_indices = list(range(min(n, 10)))

    # Check if any of indices 1, 2, 3 are in selected
    required_indices = {1, 2, 3}  # alert_type, application, action
    if not required_indices.intersection(set(selected_indices)):
        # Replace the first selected index with index 2 (application)
        if selected_indices:
            selected_indices[0] = 2
        else:
            selected_indices = [2]

    for idx in selected_indices:
        field_name, field_value = all_fields[idx]
        fields_to_set[field_name] = field_value

    return AlertMetadata(**fields_to_set)


def _make_violation_json(n_violations: int) -> str:
    """Create a valid model response JSON with n violations."""
    violations = []
    for i in range(n_violations):
        violations.append({
            "policy_name": f"Policy {i+1}",
            "clause": {
                "document_name": f"Document {i+1}",
                "section": f"Section {i+1}.1",
                "excerpt": f"This is the excerpt for violation {i+1}.",
            },
            "source_type": "FICO_INTERNAL",
            "remediation": f"Remediation action for violation {i+1}.",
        })
    return json.dumps({"violations": violations})


# =============================================================================
# Property 18: Confidence Score Computation
# =============================================================================


class TestConfidenceScoreComputation:
    """Property 18: Confidence Score Computation.

    For any similarity score S (float 0.0-1.0) and alert metadata with F fields
    present out of 10 total schema fields, the computed confidence score SHALL
    equal round(S * 100 * 0.6 + (F / 10) * 100 * 0.4), and SHALL be clamped
    to the range [0, 100].

    **Validates: Requirements 5.3**
    """

    @given(
        similarity_score=similarity_scores,
        num_fields=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=10)
    def test_confidence_score_formula(self, similarity_score: float, num_fields: int):
        """Confidence score SHALL equal the specified formula for any valid inputs."""
        # Feature: soc-policy-engine, Property 18: Confidence Score Computation
        classifier = ViolationClassifier()
        alert = _build_alert_with_n_fields(num_fields)

        # Count actual fields present (may differ slightly from num_fields
        # due to required field constraints)
        actual_fields_present = 0
        for field_name in _ALERT_METADATA_FIELDS:
            value = getattr(alert, field_name, None)
            if value is not None:
                actual_fields_present += 1

        confidence = classifier._compute_confidence(similarity_score, alert)

        # Expected formula
        expected_raw = similarity_score * 100.0 * 0.6 + (actual_fields_present / 10.0) * 100.0 * 0.4
        expected = max(0, min(100, round(expected_raw)))

        assert confidence == expected, (
            f"Expected confidence={expected} for similarity={similarity_score}, "
            f"fields_present={actual_fields_present}, got {confidence}"
        )

    @given(similarity_score=similarity_scores)
    @settings(max_examples=10)
    def test_confidence_score_clamped_to_0_100(self, similarity_score: float):
        """Confidence score SHALL always be in [0, 100] regardless of inputs."""
        # Feature: soc-policy-engine, Property 18: Confidence Score Computation
        classifier = ViolationClassifier()
        alert = _build_alert_with_n_fields(5)

        confidence = classifier._compute_confidence(similarity_score, alert)

        assert 0 <= confidence <= 100, (
            f"Confidence {confidence} is outside [0, 100] range"
        )

    @given(
        similarity_score=similarity_scores,
        num_fields=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=10)
    def test_confidence_is_integer(self, similarity_score: float, num_fields: int):
        """Confidence score SHALL be an integer (rounded to nearest)."""
        # Feature: soc-policy-engine, Property 18: Confidence Score Computation
        classifier = ViolationClassifier()
        alert = _build_alert_with_n_fields(num_fields)

        confidence = classifier._compute_confidence(similarity_score, alert)

        assert isinstance(confidence, int), (
            f"Confidence should be int, got {type(confidence)}"
        )


# =============================================================================
# Property 19: Severity Assignment from Confidence
# =============================================================================


class TestSeverityAssignmentFromConfidence:
    """Property 19: Severity Assignment from Confidence.

    For any confidence score C in [0, 100], the assigned severity SHALL be:
    CRITICAL if 90 <= C <= 100, HIGH if 70 <= C <= 89, MEDIUM if 50 <= C <= 69,
    LOW if 30 <= C <= 49, INFORMATIONAL if 0 <= C <= 29.

    **Validates: Requirements 5.2**
    """

    @given(confidence=st.integers(min_value=90, max_value=100))
    @settings(max_examples=15)
    def test_critical_severity_for_90_to_100(self, confidence: int):
        """Confidence 90-100 SHALL map to CRITICAL severity."""
        # Feature: soc-policy-engine, Property 19: Severity Assignment from Confidence
        classifier = ViolationClassifier()
        severity = classifier._assign_severity(confidence)
        assert severity == Severity.CRITICAL, (
            f"Confidence {confidence} should map to CRITICAL, got {severity}"
        )

    @given(confidence=st.integers(min_value=70, max_value=89))
    @settings(max_examples=15)
    def test_high_severity_for_70_to_89(self, confidence: int):
        """Confidence 70-89 SHALL map to HIGH severity."""
        # Feature: soc-policy-engine, Property 19: Severity Assignment from Confidence
        classifier = ViolationClassifier()
        severity = classifier._assign_severity(confidence)
        assert severity == Severity.HIGH, (
            f"Confidence {confidence} should map to HIGH, got {severity}"
        )

    @given(confidence=st.integers(min_value=50, max_value=69))
    @settings(max_examples=15)
    def test_medium_severity_for_50_to_69(self, confidence: int):
        """Confidence 50-69 SHALL map to MEDIUM severity."""
        # Feature: soc-policy-engine, Property 19: Severity Assignment from Confidence
        classifier = ViolationClassifier()
        severity = classifier._assign_severity(confidence)
        assert severity == Severity.MEDIUM, (
            f"Confidence {confidence} should map to MEDIUM, got {severity}"
        )

    @given(confidence=st.integers(min_value=30, max_value=49))
    @settings(max_examples=15)
    def test_low_severity_for_30_to_49(self, confidence: int):
        """Confidence 30-49 SHALL map to LOW severity."""
        # Feature: soc-policy-engine, Property 19: Severity Assignment from Confidence
        classifier = ViolationClassifier()
        severity = classifier._assign_severity(confidence)
        assert severity == Severity.LOW, (
            f"Confidence {confidence} should map to LOW, got {severity}"
        )

    @given(confidence=st.integers(min_value=0, max_value=29))
    @settings(max_examples=15)
    def test_informational_severity_for_0_to_29(self, confidence: int):
        """Confidence 0-29 SHALL map to INFORMATIONAL severity."""
        # Feature: soc-policy-engine, Property 19: Severity Assignment from Confidence
        classifier = ViolationClassifier()
        severity = classifier._assign_severity(confidence)
        assert severity == Severity.INFORMATIONAL, (
            f"Confidence {confidence} should map to INFORMATIONAL, got {severity}"
        )

    @given(confidence=confidence_scores)
    @settings(max_examples=10)
    def test_severity_covers_full_range(self, confidence: int):
        """Every confidence score in [0, 100] SHALL map to exactly one severity level."""
        # Feature: soc-policy-engine, Property 19: Severity Assignment from Confidence
        classifier = ViolationClassifier()
        severity = classifier._assign_severity(confidence)

        # Verify it maps to the correct severity based on thresholds
        if 90 <= confidence <= 100:
            assert severity == Severity.CRITICAL
        elif 70 <= confidence <= 89:
            assert severity == Severity.HIGH
        elif 50 <= confidence <= 69:
            assert severity == Severity.MEDIUM
        elif 30 <= confidence <= 49:
            assert severity == Severity.LOW
        elif 0 <= confidence <= 29:
            assert severity == Severity.INFORMATIONAL
        else:
            # Should never reach here for valid inputs
            assert False, f"Unexpected confidence value: {confidence}"


# =============================================================================
# Property 20: Multi-Violation Preservation with Cap
# =============================================================================


class TestMultiViolationPreservationWithCap:
    """Property 20: Multi-Violation Preservation with Cap.

    For any model response containing N identified violations (N >= 1), the
    PolicyViolationReport SHALL contain min(N, 25) violation objects, each with
    an independently computed confidence score and severity. No violations SHALL
    be dropped below the 25-violation cap.

    **Validates: Requirements 5.4**
    """

    @given(num_violations=st.integers(min_value=1, max_value=25))
    @settings(max_examples=15)
    def test_all_violations_preserved_below_cap(self, num_violations: int):
        """For N <= 25 violations, all N SHALL be preserved in the report."""
        # Feature: soc-policy-engine, Property 20: Multi-Violation Preservation with Cap
        classifier = ViolationClassifier()
        model_response = _make_violation_json(num_violations)

        alert = AlertMetadata(
            alert_type="DLP",
            application="Dropbox",
            action="upload",
        )
        chunks = [
            ChunkMetadata(
                source_type=SourceType.FICO_INTERNAL,
                document_name="Test Policy",
                section_id="1.1",
                ingestion_date="2024-01-15T10:30:00Z",
                similarity_score=0.85,
                content="Test content",
            )
        ]

        report = classifier.classify(model_response, alert, chunks)

        assert len(report.violations) == num_violations, (
            f"Expected {num_violations} violations, got {len(report.violations)}"
        )

    @given(num_violations=st.integers(min_value=26, max_value=50))
    @settings(max_examples=10)
    def test_violations_capped_at_25(self, num_violations: int):
        """For N > 25 violations, exactly 25 SHALL be in the report."""
        # Feature: soc-policy-engine, Property 20: Multi-Violation Preservation with Cap
        classifier = ViolationClassifier()
        model_response = _make_violation_json(num_violations)

        alert = AlertMetadata(
            alert_type="DLP",
            application="Dropbox",
            action="upload",
        )
        chunks = [
            ChunkMetadata(
                source_type=SourceType.FICO_INTERNAL,
                document_name="Test Policy",
                section_id="1.1",
                ingestion_date="2024-01-15T10:30:00Z",
                similarity_score=0.85,
                content="Test content",
            )
        ]

        report = classifier.classify(model_response, alert, chunks)

        assert len(report.violations) == 25, (
            f"Expected 25 violations (capped), got {len(report.violations)}"
        )

    @given(num_violations=st.integers(min_value=1, max_value=50))
    @settings(max_examples=15)
    def test_violation_count_equals_min_n_25(self, num_violations: int):
        """Report SHALL contain exactly min(N, 25) violations."""
        # Feature: soc-policy-engine, Property 20: Multi-Violation Preservation with Cap
        classifier = ViolationClassifier()
        model_response = _make_violation_json(num_violations)

        alert = AlertMetadata(
            alert_type="DLP",
            application="Dropbox",
            action="upload",
        )
        chunks = [
            ChunkMetadata(
                source_type=SourceType.FICO_INTERNAL,
                document_name="Test Policy",
                section_id="1.1",
                ingestion_date="2024-01-15T10:30:00Z",
                similarity_score=0.85,
                content="Test content",
            )
        ]

        report = classifier.classify(model_response, alert, chunks)

        expected_count = min(num_violations, 25)
        assert len(report.violations) == expected_count, (
            f"Expected min({num_violations}, 25) = {expected_count} violations, "
            f"got {len(report.violations)}"
        )

    @given(num_violations=st.integers(min_value=2, max_value=25))
    @settings(max_examples=10)
    def test_each_violation_has_independent_confidence_and_severity(
        self, num_violations: int
    ):
        """Each violation SHALL have an independently computed confidence and severity."""
        # Feature: soc-policy-engine, Property 20: Multi-Violation Preservation with Cap
        classifier = ViolationClassifier()
        model_response = _make_violation_json(num_violations)

        alert = AlertMetadata(
            alert_type="DLP",
            application="Dropbox",
            action="upload",
        )
        chunks = [
            ChunkMetadata(
                source_type=SourceType.FICO_INTERNAL,
                document_name="Test Policy",
                section_id="1.1",
                ingestion_date="2024-01-15T10:30:00Z",
                similarity_score=0.85,
                content="Test content",
            )
        ]

        report = classifier.classify(model_response, alert, chunks)

        for violation in report.violations:
            # Each violation must have a valid confidence score
            assert 0 <= violation.confidence_score <= 100
            # Each violation must have a valid severity
            assert violation.severity in [
                Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM,
                Severity.LOW, Severity.INFORMATIONAL,
            ]


# =============================================================================
# Property 21: Overall Severity is Maximum
# =============================================================================


class TestOverallSeverityIsMaximum:
    """Property 21: Overall Severity is Maximum.

    For any PolicyViolationReport containing one or more violations, the
    overall_severity SHALL equal the highest severity among all individual
    violations, using the ordering CRITICAL > HIGH > MEDIUM > LOW > INFORMATIONAL.
    If no violations exist, overall_severity SHALL be NO_VIOLATION.

    **Validates: Requirements 5.6**
    """

    @given(num_violations=st.integers(min_value=1, max_value=25))
    @settings(max_examples=15)
    def test_overall_severity_equals_max_individual(self, num_violations: int):
        """Overall severity SHALL equal the maximum severity among violations."""
        # Feature: soc-policy-engine, Property 21: Overall Severity is Maximum
        classifier = ViolationClassifier()
        model_response = _make_violation_json(num_violations)

        alert = AlertMetadata(
            alert_type="DLP",
            application="Dropbox",
            action="upload",
        )
        chunks = [
            ChunkMetadata(
                source_type=SourceType.FICO_INTERNAL,
                document_name="Test Policy",
                section_id="1.1",
                ingestion_date="2024-01-15T10:30:00Z",
                similarity_score=0.85,
                content="Test content",
            )
        ]

        report = classifier.classify(model_response, alert, chunks)

        # Determine expected max severity from individual violations
        severity_order = [
            Severity.CRITICAL,
            Severity.HIGH,
            Severity.MEDIUM,
            Severity.LOW,
            Severity.INFORMATIONAL,
            Severity.NO_VIOLATION,
        ]

        if report.violations:
            max_severity = Severity.NO_VIOLATION
            for v in report.violations:
                if severity_order.index(v.severity) < severity_order.index(max_severity):
                    max_severity = v.severity
            assert report.overall_severity == max_severity, (
                f"Overall severity {report.overall_severity} != max individual "
                f"severity {max_severity}"
            )

    def test_no_violations_yields_no_violation_severity(self):
        """If no violations exist, overall_severity SHALL be NO_VIOLATION."""
        # Feature: soc-policy-engine, Property 21: Overall Severity is Maximum
        classifier = ViolationClassifier()
        # Empty violations list
        model_response = json.dumps({"violations": []})

        alert = AlertMetadata(
            alert_type="DLP",
            application="Dropbox",
            action="upload",
        )
        chunks = [
            ChunkMetadata(
                source_type=SourceType.FICO_INTERNAL,
                document_name="Test Policy",
                section_id="1.1",
                ingestion_date="2024-01-15T10:30:00Z",
                similarity_score=0.85,
                content="Test content",
            )
        ]

        report = classifier.classify(model_response, alert, chunks)

        assert report.overall_severity == Severity.NO_VIOLATION, (
            f"Expected NO_VIOLATION for empty violations, got {report.overall_severity}"
        )

    @given(
        similarity_scores_list=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=15)
    def test_overall_severity_ordering_is_correct(self, similarity_scores_list: list):
        """Overall severity uses correct ordering: CRITICAL > HIGH > MEDIUM > LOW > INFORMATIONAL."""
        # Feature: soc-policy-engine, Property 21: Overall Severity is Maximum
        classifier = ViolationClassifier()

        num_violations = len(similarity_scores_list)
        model_response = _make_violation_json(num_violations)

        alert = AlertMetadata(
            alert_type="DLP",
            application="Dropbox",
            action="upload",
            destination="https://example.com",
            source_ip="10.0.0.1",
            user="test@fico.com",
        )

        # Use the average of provided similarity scores
        avg_score = sum(similarity_scores_list) / len(similarity_scores_list)
        chunks = [
            ChunkMetadata(
                source_type=SourceType.FICO_INTERNAL,
                document_name="Test Policy",
                section_id="1.1",
                ingestion_date="2024-01-15T10:30:00Z",
                similarity_score=avg_score,
                content="Test content",
            )
        ]

        report = classifier.classify(model_response, alert, chunks)

        if report.violations:
            # The overall severity should be >= every individual severity
            severity_order = [
                Severity.CRITICAL,
                Severity.HIGH,
                Severity.MEDIUM,
                Severity.LOW,
                Severity.INFORMATIONAL,
                Severity.NO_VIOLATION,
            ]
            overall_idx = severity_order.index(report.overall_severity)
            for v in report.violations:
                individual_idx = severity_order.index(v.severity)
                assert overall_idx <= individual_idx, (
                    f"Overall severity {report.overall_severity} is lower than "
                    f"individual severity {v.severity}"
                )
