"""Property-based tests for payload size enforcement (Property 31).

**Validates: Requirements 7.8**
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import AsyncClient, ASGITransport

from app.middleware.payload_limit import PayloadLimitMiddleware


# =============================================================================
# Property 31: Payload Size Enforcement
# =============================================================================


def _create_test_app(max_payload_size_kb: int = 100) -> FastAPI:
    """Create a minimal FastAPI app with PayloadLimitMiddleware for testing."""
    app = FastAPI()
    app.add_middleware(PayloadLimitMiddleware, max_payload_size_kb=max_payload_size_kb)

    @app.post("/test")
    async def test_endpoint(request: Request):
        body = await request.body()
        return JSONResponse(
            status_code=200,
            content={"received": len(body)},
        )

    return app


class TestPayloadSizeEnforcement:
    """Property 31: Payload Size Enforcement.

    For any request with Content-Length exceeding MAX_PAYLOAD_SIZE_KB * 1024
    bytes, the middleware SHALL return HTTP 413. For any request within the
    limit, the middleware SHALL pass the request through.

    **Validates: Requirements 7.8**
    """

    @given(
        content_length=st.integers(
            min_value=102401,  # 100 * 1024 + 1 = just over 100KB
            max_value=500000,
        )
    )
    @settings(max_examples=10)
    @pytest.mark.asyncio
    async def test_oversized_payload_returns_413(self, content_length: int):
        """Requests with Content-Length exceeding MAX_PAYLOAD_SIZE_KB * 1024 SHALL return HTTP 413."""
        app = _create_test_app(max_payload_size_kb=100)
        max_bytes = 100 * 1024  # 102400 bytes

        # Ensure content_length exceeds the limit
        assume(content_length > max_bytes)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/test",
                content=b"x" * 1024,  # Actual body doesn't matter; middleware checks header
                headers={"content-length": str(content_length)},
            )

        assert response.status_code == 413
        body = response.json()
        assert body["error"] is True
        assert body["error_code"] == "PAYLOAD_TOO_LARGE"

    @given(
        content_length=st.integers(
            min_value=0,
            max_value=102400,  # 100 * 1024 = exactly 100KB (within limit)
        )
    )
    @settings(max_examples=10)
    @pytest.mark.asyncio
    async def test_within_limit_payload_passes_through(self, content_length: int):
        """Requests with Content-Length within MAX_PAYLOAD_SIZE_KB * 1024 SHALL pass through."""
        app = _create_test_app(max_payload_size_kb=100)

        # Create body matching the content-length header
        body_content = b"x" * content_length

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/test",
                content=body_content,
                headers={"content-length": str(content_length)},
            )

        assert response.status_code == 200
        response_body = response.json()
        assert response_body["received"] == content_length

    @given(
        max_kb=st.integers(min_value=1, max_value=500),
        excess_bytes=st.integers(min_value=1, max_value=10000),
    )
    @settings(max_examples=10)
    @pytest.mark.asyncio
    async def test_configurable_limit_rejects_oversized(self, max_kb: int, excess_bytes: int):
        """For any configured MAX_PAYLOAD_SIZE_KB, requests exceeding that limit SHALL return 413."""
        app = _create_test_app(max_payload_size_kb=max_kb)
        max_bytes = max_kb * 1024
        content_length = max_bytes + excess_bytes

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/test",
                content=b"x" * 1024,
                headers={"content-length": str(content_length)},
            )

        assert response.status_code == 413
        body = response.json()
        assert body["error"] is True
        assert body["error_code"] == "PAYLOAD_TOO_LARGE"

    @given(
        max_kb=st.integers(min_value=1, max_value=500),
        fraction=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=10)
    @pytest.mark.asyncio
    async def test_configurable_limit_accepts_within(self, max_kb: int, fraction: float):
        """For any configured MAX_PAYLOAD_SIZE_KB, requests within that limit SHALL pass through."""
        app = _create_test_app(max_payload_size_kb=max_kb)
        max_bytes = max_kb * 1024
        content_length = int(max_bytes * fraction)

        body_content = b"x" * content_length

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/test",
                content=body_content,
                headers={"content-length": str(content_length)},
            )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_boundary_exactly_at_limit_passes(self):
        """A request with Content-Length exactly equal to MAX_PAYLOAD_SIZE_KB * 1024 SHALL pass."""
        app = _create_test_app(max_payload_size_kb=100)
        max_bytes = 100 * 1024  # 102400

        body_content = b"x" * max_bytes

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/test",
                content=body_content,
                headers={"content-length": str(max_bytes)},
            )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_boundary_one_byte_over_limit_rejected(self):
        """A request with Content-Length one byte over MAX_PAYLOAD_SIZE_KB * 1024 SHALL return 413."""
        app = _create_test_app(max_payload_size_kb=100)
        max_bytes = 100 * 1024  # 102400
        content_length = max_bytes + 1

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/test",
                content=b"x" * 1024,
                headers={"content-length": str(content_length)},
            )

        assert response.status_code == 413
