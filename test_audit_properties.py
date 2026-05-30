"""Property-based tests for audit logging (Properties 24, 25, 26).

**Validates: Requirements 8.1, 8.5, 8.6**
"""

import hashlib
import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from app.models.audit import AuditLogEntry
from app.services.audit_logger import AuditLogger


# =============================================================================
# Strategies
# =============================================================================

# Strategy for non-empty strings suitable for alert metadata fields
_non_empty_text = st.text(
    min_size=1,
    max_size=100,
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S"),
        blacklist_categories=("Cs",),
    ),
)


def _alert_with_user_strategy():
    """Strategy generating alert dicts that always have a non-None user field."""
    return st.fixed_dictionaries(
        {
            "alert_type": _non_empty_text,
            "application": _non_empty_text,
            "action": _non_empty_text,
            "user": _non_empty_text,
        },
        optional={
            "source_ip": st.one_of(st.none(), _non_empty_text),
            "destination": st.one_of(st.none(), _non_empty_text),
        },
    )


def _alert_with_source_ip_strategy():
    """Strategy generating alert dicts that always have a non-None source_ip field."""
    return st.fixed_dictionaries(
        {
            "alert_type": _non_empty_text,
            "application": _non_empty_text,
            "action": _non_empty_text,
            "source_ip": _non_empty_text,
        },
        optional={
            "user": st.one_of(st.none(), _non_empty_text),
            "destination": st.one_of(st.none(), _non_empty_text),
        },
    )


def _alert_metadata_strategy():
    """Strategy generating alert metadata dicts with optional user and source_ip."""
    return st.fixed_dictionaries(
        {
            "alert_type": _non_empty_text,
            "application": _non_empty_text,
            "action": _non_empty_text,
        },
        optional={
            "user": st.one_of(st.none(), _non_empty_text),
            "source_ip": st.one_of(st.none(), _non_empty_text),
            "destination": st.one_of(st.none(), _non_empty_text),
            "zscaler_category": st.one_of(st.none(), _non_empty_text),
        },
    )


def _audit_log_entry_strategy():
    """Strategy generating valid AuditLogEntry instances."""
    return st.builds(
        AuditLogEntry,
        timestamp=st.just(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")),
        mode=st.sampled_from(["Q&A", "Alert Analysis"]),
        input_data=st.text(min_size=1, max_size=200, alphabet=st.characters(
            whitelist_categories=("L", "N", "P", "S", "Z"),
            blacklist_categories=("Cs",),
        )),
        chunk_count=st.integers(min_value=0, max_value=50),
        retrieval_latency_ms=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        model_response=st.text(min_size=1, max_size=200, alphabet=st.characters(
            whitelist_categories=("L", "N", "P", "S", "Z"),
            blacklist_categories=("Cs",),
        )),
        parsed_summary=st.text(min_size=1, max_size=200, alphabet=st.characters(
            whitelist_categories=("L", "N", "P", "S", "Z"),
            blacklist_categories=("Cs",),
        )),
        analyst_identifier=st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=("L", "N"),
            blacklist_categories=("Cs",),
        )),
        from_cache=st.booleans(),
    )


# =============================================================================
# Property 24: PII Redaction in Audit Logs
# =============================================================================


class TestPIIRedactionInAuditLogs:
    """Property 24: PII Redaction in Audit Logs.

    For any AlertMetadata containing user and/or source_ip fields, the
    _redact_pii method SHALL replace those values with their SHA256 hex digests
    while preserving all other fields unchanged.

    **Validates: Requirements 8.6**
    """

    @given(alert=_alert_with_user_strategy())
    @settings(max_examples=10)
    def test_user_field_replaced_with_sha256(self, alert: dict):
        """When user field is present and non-None, it SHALL be replaced with its SHA256 hex digest."""
        logger = AuditLogger(db_path=":memory:", retention_days=90)
        redacted = logger._redact_pii(alert)

        expected_hash = hashlib.sha256(alert["user"].encode("utf-8")).hexdigest()
        assert redacted["user"] == expected_hash

    @given(alert=_alert_with_source_ip_strategy())
    @settings(max_examples=10)
    def test_source_ip_field_replaced_with_sha256(self, alert: dict):
        """When source_ip field is present and non-None, it SHALL be replaced with its SHA256 hex digest."""
        logger = AuditLogger(db_path=":memory:", retention_days=90)
        redacted = logger._redact_pii(alert)

        expected_hash = hashlib.sha256(alert["source_ip"].encode("utf-8")).hexdigest()
        assert redacted["source_ip"] == expected_hash

    @given(alert=_alert_metadata_strategy())
    @settings(max_examples=10)
    def test_other_fields_preserved_unchanged(self, alert: dict):
        """All fields other than user and source_ip SHALL remain unchanged after redaction."""
        logger = AuditLogger(db_path=":memory:", retention_days=90)
        redacted = logger._redact_pii(alert)

        for key, value in alert.items():
            if key not in ("user", "source_ip"):
                assert redacted[key] == value, (
                    f"Field '{key}' was modified: expected {value!r}, got {redacted[key]!r}"
                )

    @given(alert=_alert_metadata_strategy())
    @settings(max_examples=10)
    def test_redaction_does_not_mutate_original(self, alert: dict):
        """The _redact_pii method SHALL not mutate the original alert dict."""
        logger = AuditLogger(db_path=":memory:", retention_days=90)
        original_copy = dict(alert)
        _ = logger._redact_pii(alert)

        assert alert == original_copy

    @given(
        user=_non_empty_text,
        source_ip=_non_empty_text,
    )
    @settings(max_examples=10)
    def test_both_pii_fields_redacted_simultaneously(self, user: str, source_ip: str):
        """When both user and source_ip are present, both SHALL be replaced with SHA256 digests."""
        alert = {
            "alert_type": "DLP",
            "application": "Dropbox",
            "user": user,
            "source_ip": source_ip,
        }

        logger = AuditLogger(db_path=":memory:", retention_days=90)
        redacted = logger._redact_pii(alert)

        assert redacted["user"] == hashlib.sha256(user.encode("utf-8")).hexdigest()
        assert redacted["source_ip"] == hashlib.sha256(source_ip.encode("utf-8")).hexdigest()
        # Non-PII fields preserved
        assert redacted["alert_type"] == "DLP"
        assert redacted["application"] == "Dropbox"

    @given(alert=_alert_metadata_strategy())
    @settings(max_examples=10)
    def test_none_pii_fields_not_hashed(self, alert: dict):
        """When user or source_ip is None, it SHALL remain None (not hashed)."""
        # Force both to None
        alert["user"] = None
        alert["source_ip"] = None

        logger = AuditLogger(db_path=":memory:", retention_days=90)
        redacted = logger._redact_pii(alert)

        assert redacted.get("user") is None
        assert redacted.get("source_ip") is None


# =============================================================================
# Property 25: Audit Log Entry Completeness
# =============================================================================


class TestAuditLogEntryCompleteness:
    """Property 25: Audit Log Entry Completeness.

    For any logged interaction, the audit log entry SHALL contain all required
    fields (timestamp, mode, input_data, chunk_count, retrieval_latency_ms,
    model_response, parsed_summary, analyst_identifier, from_cache) with
    non-empty/non-null values.

    **Validates: Requirements 8.1**
    """

    @given(entry=_audit_log_entry_strategy())
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @pytest.mark.asyncio
    async def test_logged_entry_contains_all_required_fields(self, entry: AuditLogEntry):
        """Every logged entry SHALL contain all required fields with non-empty/non-null values."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "test_audit.db")
            logger = AuditLogger(db_path=db_path, retention_days=90)

            await logger.log_interaction(entry)

            # Query back the entry
            logs = await logger.query_logs()
            assert len(logs) == 1

            logged = logs[0]

            # Verify all required fields are present and non-null
            assert logged.timestamp is not None and logged.timestamp != ""
            assert logged.mode is not None and logged.mode != ""
            assert logged.input_data is not None and logged.input_data != ""
            assert logged.chunk_count is not None
            assert logged.retrieval_latency_ms is not None
            assert logged.model_response is not None and logged.model_response != ""
            assert logged.parsed_summary is not None and logged.parsed_summary != ""
            assert logged.analyst_identifier is not None and logged.analyst_identifier != ""
            assert logged.from_cache is not None

    @given(entry=_audit_log_entry_strategy())
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @pytest.mark.asyncio
    async def test_logged_entry_preserves_field_values(self, entry: AuditLogEntry):
        """Logged entry field values SHALL match the original entry values."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "test_audit.db")
            logger = AuditLogger(db_path=db_path, retention_days=90)

            await logger.log_interaction(entry)

            logs = await logger.query_logs()
            assert len(logs) == 1

            logged = logs[0]

            assert logged.timestamp == entry.timestamp
            assert logged.mode == entry.mode
            assert logged.input_data == entry.input_data[:10000]
            assert logged.chunk_count == entry.chunk_count
            assert logged.retrieval_latency_ms == pytest.approx(entry.retrieval_latency_ms, rel=1e-5)
            assert logged.model_response == entry.model_response[:10000]
            assert logged.parsed_summary == entry.parsed_summary
            assert logged.analyst_identifier == entry.analyst_identifier
            assert logged.from_cache == entry.from_cache

    @given(entry=_audit_log_entry_strategy())
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @pytest.mark.asyncio
    async def test_from_cache_flag_recorded(self, entry: AuditLogEntry):
        """The from_cache flag SHALL be accurately recorded for cache hit rate monitoring."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "test_audit.db")
            logger = AuditLogger(db_path=db_path, retention_days=90)

            await logger.log_interaction(entry)

            logs = await logger.query_logs()
            assert len(logs) == 1
            assert logs[0].from_cache == entry.from_cache


# =============================================================================
# Property 26: Audit Retention Purge
# =============================================================================


class TestAuditRetentionPurge:
    """Property 26: Audit Retention Purge.

    For any set of audit entries, purge_expired SHALL delete all entries with
    timestamp older than AUDIT_RETENTION_DAYS and retain all entries within the
    retention period.

    **Validates: Requirements 8.5**
    """

    @given(
        retention_days=st.integers(min_value=1, max_value=365),
        days_old=st.integers(min_value=2, max_value=730),
    )
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @pytest.mark.asyncio
    async def test_expired_entries_deleted(self, retention_days: int, days_old: int):
        """Entries with timestamp older than AUDIT_RETENTION_DAYS SHALL be deleted by purge_expired."""
        assume(days_old > retention_days)

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "test_audit.db")
            logger = AuditLogger(db_path=db_path, retention_days=retention_days)

            # Create an entry with a timestamp older than retention period
            old_timestamp = (
                datetime.now(timezone.utc) - timedelta(days=days_old)
            ).strftime("%Y-%m-%dT%H:%M:%S")

            entry = AuditLogEntry(
                timestamp=old_timestamp,
                mode="Q&A",
                input_data="test question",
                chunk_count=3,
                retrieval_latency_ms=150.0,
                model_response="test response",
                parsed_summary="summary",
                analyst_identifier="analyst1",
                from_cache=False,
            )

            await logger.log_interaction(entry)

            # Verify entry exists
            logs = await logger.query_logs()
            assert len(logs) == 1

            # Purge expired entries
            deleted = await logger.purge_expired()

            # Entry should be deleted
            assert deleted >= 1
            logs_after = await logger.query_logs()
            assert len(logs_after) == 0

    @given(
        retention_days=st.integers(min_value=30, max_value=365),
        days_old=st.integers(min_value=0, max_value=28),
    )
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @pytest.mark.asyncio
    async def test_recent_entries_retained(self, retention_days: int, days_old: int):
        """Entries within the retention period SHALL be retained by purge_expired."""
        assume(days_old < retention_days)

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "test_audit.db")
            logger = AuditLogger(db_path=db_path, retention_days=retention_days)

            # Create an entry within the retention period
            recent_timestamp = (
                datetime.now(timezone.utc) - timedelta(days=days_old)
            ).strftime("%Y-%m-%dT%H:%M:%S")

            entry = AuditLogEntry(
                timestamp=recent_timestamp,
                mode="Alert Analysis",
                input_data="alert data",
                chunk_count=5,
                retrieval_latency_ms=200.0,
                model_response="analysis response",
                parsed_summary="violation summary",
                analyst_identifier="analyst2",
                from_cache=True,
            )

            await logger.log_interaction(entry)

            # Purge expired entries
            deleted = await logger.purge_expired()

            # Entry should be retained
            assert deleted == 0
            logs_after = await logger.query_logs()
            assert len(logs_after) == 1

    @given(
        retention_days=st.integers(min_value=10, max_value=180),
        num_expired=st.integers(min_value=1, max_value=5),
        num_recent=st.integers(min_value=1, max_value=5),
    )
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    @pytest.mark.asyncio
    async def test_mixed_entries_purge_correctly(
        self, retention_days: int, num_expired: int, num_recent: int
    ):
        """purge_expired SHALL delete all expired entries and retain all recent entries."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "test_audit.db")
            logger = AuditLogger(db_path=db_path, retention_days=retention_days)

            # Insert expired entries (older than retention period)
            for i in range(num_expired):
                old_timestamp = (
                    datetime.now(timezone.utc) - timedelta(days=retention_days + i + 1)
                ).strftime("%Y-%m-%dT%H:%M:%S")
                entry = AuditLogEntry(
                    timestamp=old_timestamp,
                    mode="Q&A",
                    input_data=f"expired question {i}",
                    chunk_count=i,
                    retrieval_latency_ms=100.0 + i,
                    model_response=f"expired response {i}",
                    parsed_summary=f"expired summary {i}",
                    analyst_identifier=f"analystexpired{i}",
                    from_cache=False,
                )
                await logger.log_interaction(entry)

            # Insert recent entries (within retention period)
            for i in range(num_recent):
                recent_timestamp = (
                    datetime.now(timezone.utc) - timedelta(days=i)
                ).strftime("%Y-%m-%dT%H:%M:%S")
                entry = AuditLogEntry(
                    timestamp=recent_timestamp,
                    mode="Alert Analysis",
                    input_data=f"recent alert {i}",
                    chunk_count=i + 1,
                    retrieval_latency_ms=50.0 + i,
                    model_response=f"recent response {i}",
                    parsed_summary=f"recent summary {i}",
                    analyst_identifier=f"analystrecent{i}",
                    from_cache=True,
                )
                await logger.log_interaction(entry)

            # Verify all entries exist
            all_logs = await logger.query_logs()
            assert len(all_logs) == num_expired + num_recent

            # Purge expired entries
            deleted = await logger.purge_expired()

            # All expired entries should be deleted
            assert deleted == num_expired

            # Only recent entries should remain
            remaining_logs = await logger.query_logs()
            assert len(remaining_logs) == num_recent
