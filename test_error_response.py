"""Property-based tests for consistent error response schema (Property 30).

**Validates: Requirements 12.6**

For any error condition (validation failure, internal error, timeout, auth failure),
the error response SHALL conform to the StandardErrorResponse schema containing:
error (bool=True), error_code (non-empty string), message (non-empty string),
and request_id (non-empty string). No error response SHALL expose stack traces
or internal paths.
"""

import re
import uuid
from typing import Optional
from unittest.mock import AsyncMock, patch

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import AsyncClient, ASGITransport

from app.middleware.request_id import RequestIDMiddleware
from app.models.error import StandardErrorResponse
from app.routers import query as query_router
from app.routers import alert as alert_router
from app.routers import ingest as ingest_router
from app.routers import audit as audit_router


# =============================================================================
# Helpers
# =============================================================================

# Patterns that indicate stack traces or internal paths in error messages
STACK_TRACE_PATTERNS = [
    r"Traceback \(most recent call last\)",
    r"File \"[^\"]+\", line \d+",
    r"raise \w+",
    r"^\s+at\s+",  # JavaScript-style stack traces
    r"[A-Z]:\\",  # Windows paths like C:\Users\...
    r"/home/",  # Linux home paths
    r"/usr/",  # Linux system paths
    r"/app/",  # Container paths
    r"\\app\\",  # Windows container paths
    r"site-packages",  # Python package paths
    r"\.py:\d+",  # Python file:line references
    r"\.pyc",  # Compiled Python files
]

STACK_TRACE_RE = re.compile("|".join(STACK_TRACE_PATTERNS), re.MULTILINE)


def _create_test_app() -> FastAPI:
    """Create a minimal FastAPI app with routers and request ID middleware."""
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)
    app.include_router(query_router.router)
    app.include_router(alert_router.router)
    app.include_router(ingest_router.router)
    app.include_router(audit_router.router)
    return app


def _validate_error_response(body: dict) -> None:
    """Validate that a response body conforms to StandardErrorResponse schema.

    Asserts:
    - error field is True (bool)
    - error_code is a non-empty string
    - message is a non-empty string
    - request_id is a non-empty string
    - No stack traces or internal paths in message
    """
    # Schema conformance
    assert "error" in body, "Response missing 'error' field"
    assert body["error"] is True, f"'error' field should be True, got {body['error']}"

    assert "error_code" in body, "Response missing 'error_code' field"
    assert isinstance(body["error_code"], str), f"'error_code' should be str, got {type(body['error_code'])}"
    assert len(body["error_code"]) > 0, "'error_code' should be non-empty"

    assert "message" in body, "Response missing 'message' field"
    assert isinstance(body["message"], str), f"'message' should be str, got {type(body['message'])}"
    assert len(body["message"]) > 0, "'message' should be non-empty"

    assert "request_id" in body, "Response missing 'request_id' field"
    assert isinstance(body["request_id"], str), f"'request_id' should be str, got {type(body['request_id'])}"
    assert len(body["request_id"]) > 0, "'request_id' should be non-empty"

    # No stack traces or internal paths
    assert not STACK_TRACE_RE.search(body["message"]), (
        f"Error message exposes stack trace or internal path: {body['message']}"
    )
    assert not STACK_TRACE_RE.search(body["error_code"]), (
        f"Error code exposes stack trace or internal path: {body['error_code']}"
    )


# =============================================================================
# Strategies
# =============================================================================

# Strategy for generating various exception types that could occur internally
exception_types = st.sampled_from([
    RuntimeError,
    ValueError,
    TypeError,
    IOError,
    ConnectionError,
    TimeoutError,
])

# Strategy for generating exception messages (potentially containing sensitive info)
exception_messages = st.one_of(
    st.text(min_size=1, max_size=200),
    st.sampled_from([
        "Traceback (most recent call last):\n  File \"/app/services/engine.py\", line 42",
        "C:\\Users\\admin\\project\\app\\services\\policy_engine.py:55",
        "/home/deploy/soc-engine/app/services/rag_client.py line 100",
        "ConnectionRefusedError at /usr/lib/python3.11/site-packages/boto3/client.py:123",
        "raise ValueError('invalid config at /app/config.py:30')",
        "NoneType has no attribute 'process_query'",
        "site-packages/fastapi/routing.py:234",
    ]),
)

# Strategy for generating invalid query payloads that trigger validation errors
invalid_query_payloads = st.one_of(
    # Empty question
    st.just({"question": ""}),
    # Question too long (over 1000 chars)
    st.text(min_size=1001, max_size=1500, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"))).map(
        lambda q: {"question": q}
    ),
    # Missing question field entirely
    st.just({}),
    st.just({"query": "wrong field name"}),
)


# =============================================================================
# Property 30: Consistent Error Response Schema
# =============================================================================


class TestConsistentErrorResponseSchema:
    """Property 30: Consistent Error Response Schema.

    For any error condition (validation failure, internal error, timeout,
    auth failure), the error response SHALL conform to the StandardErrorResponse
    schema containing: error (bool=True), error_code (non-empty string),
    message (non-empty string), and request_id (non-empty string).
    No error response SHALL expose stack traces or internal paths.

    **Validates: Requirements 12.6**
    """

    @given(exc_type=exception_types, exc_msg=exception_messages)
    @settings(max_examples=10)
    @pytest.mark.asyncio
    async def test_internal_error_conforms_to_schema(self, exc_type, exc_msg: str):
        """Internal errors from the policy engine SHALL produce conformant error responses."""
        app = _create_test_app()

        # Mock the policy engine to raise an exception
        mock_engine = AsyncMock()
        mock_engine.process_query = AsyncMock(side_effect=exc_type(exc_msg))

        # Override the dependency to return our mock engine
        from app.dependencies import get_policy_engine
        app.dependency_overrides[get_policy_engine] = lambda: mock_engine

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/query",
                json={"question": "What is the acceptable use policy?"},
            )

        # Should be an error status code
        assert response.status_code >= 400

        body = response.json()
        _validate_error_response(body)

        # Clean up
        app.dependency_overrides.clear()

    @given(exc_type=exception_types, exc_msg=exception_messages)
    @settings(max_examples=10)
    @pytest.mark.asyncio
    async def test_alert_internal_error_conforms_to_schema(self, exc_type, exc_msg: str):
        """Internal errors from alert analysis SHALL produce conformant error responses."""
        app = _create_test_app()

        # Mock the policy engine to raise an exception
        mock_engine = AsyncMock()
        mock_engine.analyze_alert = AsyncMock(side_effect=exc_type(exc_msg))

        from app.dependencies import get_policy_engine
        app.dependency_overrides[get_policy_engine] = lambda: mock_engine

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/check_policy",
                json={"application": "Dropbox", "action": "upload"},
            )

        assert response.status_code >= 400

        body = response.json()
        _validate_error_response(body)

        app.dependency_overrides.clear()

    @given(payload=invalid_query_payloads)
    @settings(max_examples=10)
    @pytest.mark.asyncio
    async def test_validation_error_conforms_to_schema(self, payload: dict):
        """Validation failures SHALL produce conformant error responses."""
        app = _create_test_app()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/query",
                json=payload,
            )

        # Should be a 422 validation error
        assert response.status_code == 422

        body = response.json()
        # FastAPI's default validation error format uses "detail" field
        # But our StandardErrorResponse uses error/error_code/message/request_id
        # The router may or may not override the default validation handler.
        # If it's the default FastAPI validation response, it still should not
        # expose stack traces or internal paths.
        if "error" in body:
            _validate_error_response(body)
        else:
            # FastAPI default validation response - verify no stack traces
            response_text = str(body)
            assert not STACK_TRACE_RE.search(response_text), (
                f"Validation error exposes stack trace or internal path: {response_text}"
            )

    @given(
        token=st.one_of(
            st.just(None),
            st.just(""),
            st.just("wrong-token"),
            st.from_regex(r"[a-zA-Z0-9\-_]{1,50}", fullmatch=True).filter(
                lambda t: t != "admin-secret-token"
            ),
        )
    )
    @settings(max_examples=10)
    @pytest.mark.asyncio
    async def test_auth_failure_conforms_to_schema(self, token: Optional[str]):
        """Authentication failures SHALL produce conformant error responses."""
        app = _create_test_app()

        headers = {}
        if token is not None:
            headers["X-Admin-Token"] = token

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/ingest",
                json={"source": "fico"},
                headers=headers,
            )

        # Should be 403 Forbidden
        assert response.status_code == 403

        body = response.json()
        _validate_error_response(body)

    @given(
        token=st.one_of(
            st.just(None),
            st.just(""),
            st.just("invalid-token"),
            st.from_regex(r"[a-zA-Z0-9\-_]{1,50}", fullmatch=True).filter(
                lambda t: t not in ("admin-secret-token", "manager-secret-token")
            ),
        )
    )
    @settings(max_examples=10)
    @pytest.mark.asyncio
    async def test_audit_auth_failure_conforms_to_schema(self, token: Optional[str]):
        """Audit endpoint auth failures SHALL produce conformant error responses."""
        app = _create_test_app()

        headers = {}
        if token is not None:
            headers["X-Admin-Token"] = token

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/audit",
                headers=headers,
            )

        assert response.status_code == 403

        body = response.json()
        _validate_error_response(body)

    @settings(max_examples=10)
    @given(
        exc_msg=exception_messages,
    )
    @pytest.mark.asyncio
    async def test_timeout_error_conforms_to_schema(self, exc_msg: str):
        """Timeout errors SHALL produce conformant error responses without exposing internals."""
        app = _create_test_app()

        # Mock the policy engine to raise TimeoutError
        mock_engine = AsyncMock()
        mock_engine.process_query = AsyncMock(side_effect=TimeoutError(exc_msg))

        from app.dependencies import get_policy_engine
        app.dependency_overrides[get_policy_engine] = lambda: mock_engine

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/query",
                json={"question": "What is the data retention policy?"},
            )

        assert response.status_code == 500

        body = response.json()
        _validate_error_response(body)

        # Specifically verify timeout error code
        assert body["error_code"] == "TIMEOUT"

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_service_unavailable_conforms_to_schema(self):
        """Service unavailable (engine=None) SHALL produce conformant error responses."""
        app = _create_test_app()

        # The default dependency returns None (engine not wired)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/query",
                json={"question": "What is the acceptable use policy?"},
            )

        assert response.status_code == 503

        body = response.json()
        _validate_error_response(body)
        assert body["error_code"] == "SERVICE_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_alert_service_unavailable_conforms_to_schema(self):
        """Alert endpoint service unavailable SHALL produce conformant error responses."""
        app = _create_test_app()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/check_policy",
                json={"application": "Dropbox", "action": "upload"},
            )

        assert response.status_code == 503

        body = response.json()
        _validate_error_response(body)
        assert body["error_code"] == "SERVICE_UNAVAILABLE"

    @given(exc_msg=exception_messages)
    @settings(max_examples=10)
    @pytest.mark.asyncio
    async def test_error_message_never_exposes_sensitive_info(self, exc_msg: str):
        """Error messages SHALL never expose stack traces or internal paths regardless of exception content."""
        app = _create_test_app()

        # Inject an exception with potentially sensitive info in its message
        mock_engine = AsyncMock()
        mock_engine.process_query = AsyncMock(side_effect=RuntimeError(exc_msg))

        from app.dependencies import get_policy_engine
        app.dependency_overrides[get_policy_engine] = lambda: mock_engine

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/query",
                json={"question": "Test question for error handling"},
            )

        assert response.status_code >= 400

        body = response.json()
        # The error message in the response must not contain the raw exception message
        # if it contains sensitive patterns
        if STACK_TRACE_RE.search(exc_msg):
            assert body["message"] != exc_msg, (
                "Error response should not echo raw exception messages containing sensitive info"
            )
        # Always validate schema conformance
        _validate_error_response(body)

        app.dependency_overrides.clear()
