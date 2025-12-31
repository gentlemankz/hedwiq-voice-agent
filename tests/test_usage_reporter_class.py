"""
Tests for the UsageReporter class.

Tests cover:
- HTTP client lifecycle management
- Thread-safety with asyncio locks
- Fail-closed behavior
- Service token handling
- Request/response handling
- Error scenarios
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Optional
import os


# ============================================================================
# Test Fixtures and Mocks
# ============================================================================

@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx client."""
    client = AsyncMock()
    client.is_closed = False
    client.post = AsyncMock()
    client.get = AsyncMock()
    client.aclose = AsyncMock()
    return client


@pytest.fixture
def mock_response_success():
    """Create a successful mock response."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"success": True}
    response.content = b'{"success": true}'
    return response


@pytest.fixture
def mock_response_error():
    """Create an error mock response."""
    response = MagicMock()
    response.status_code = 500
    response.json.return_value = {"error": "Internal server error"}
    response.content = b'{"error": "Internal server error"}'
    return response


# ============================================================================
# UsageReporter Initialization Tests
# ============================================================================

class TestUsageReporterInit:
    """Tests for UsageReporter initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        # Patch environment variables
        with patch.dict(os.environ, {
            "FRONTEND_URL": "http://test:3000",
            "INTERNAL_SERVICE_TOKEN": "test-token",
            "NODE_ENV": "development",
            "BYPASS_USAGE_CHECKS": "",
        }):
            from usage_reporter import UsageReporter
            reporter = UsageReporter()

            assert reporter.frontend_url == "http://test:3000"
            assert reporter.service_token == "test-token"
            assert reporter.timeout == 10.0
            assert reporter._client is None

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        with patch.dict(os.environ, {
            "NODE_ENV": "development",
            "BYPASS_USAGE_CHECKS": "",
        }):
            from usage_reporter import UsageReporter
            reporter = UsageReporter(
                frontend_url="http://custom:8080",
                service_token="custom-token",
                timeout=30.0,
            )

            assert reporter.frontend_url == "http://custom:8080"
            assert reporter.service_token == "custom-token"
            assert reporter.timeout == 30.0

    def test_warning_when_no_service_token(self, caplog):
        """Test that warning is logged when service token is missing."""
        import logging

        with patch.dict(os.environ, {
            "FRONTEND_URL": "http://test:3000",
            "INTERNAL_SERVICE_TOKEN": "",
            "NODE_ENV": "development",
            "BYPASS_USAGE_CHECKS": "",
        }):
            from usage_reporter import UsageReporter

            # Set caplog to capture WARNING level
            with caplog.at_level(logging.WARNING):
                reporter = UsageReporter(service_token="")

            # Verify warning was logged in constructor
            assert any(
                "INTERNAL_SERVICE_TOKEN not configured" in record.message
                for record in caplog.records
            ), "Expected warning about missing service token"
            assert reporter.service_token == ""


# ============================================================================
# HTTP Client Lifecycle Tests
# ============================================================================

class TestHTTPClientLifecycle:
    """Tests for HTTP client lifecycle management."""

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self):
        """Test that _get_client creates a client when none exists."""
        with patch.dict(os.environ, {
            "NODE_ENV": "development",
            "BYPASS_USAGE_CHECKS": "",
        }):
            from usage_reporter import UsageReporter

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_instance = AsyncMock()
                mock_instance.is_closed = False
                mock_client_class.return_value = mock_instance

                reporter = UsageReporter(service_token="test-token")
                client = await reporter._get_client()

                mock_client_class.assert_called_once_with(timeout=10.0)
                assert client == mock_instance

    @pytest.mark.asyncio
    async def test_get_client_reuses_existing(self):
        """Test that _get_client reuses existing client."""
        with patch.dict(os.environ, {
            "NODE_ENV": "development",
            "BYPASS_USAGE_CHECKS": "",
        }):
            from usage_reporter import UsageReporter

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_instance = AsyncMock()
                mock_instance.is_closed = False
                mock_client_class.return_value = mock_instance

                reporter = UsageReporter(service_token="test-token")

                # Get client twice
                client1 = await reporter._get_client()
                client2 = await reporter._get_client()

                # Should only create once
                mock_client_class.assert_called_once()
                assert client1 == client2

    @pytest.mark.asyncio
    async def test_get_client_recreates_if_closed(self):
        """Test that _get_client recreates client if closed."""
        with patch.dict(os.environ, {
            "NODE_ENV": "development",
            "BYPASS_USAGE_CHECKS": "",
        }):
            from usage_reporter import UsageReporter

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_instance1 = AsyncMock()
                mock_instance1.is_closed = False

                mock_instance2 = AsyncMock()
                mock_instance2.is_closed = False

                mock_client_class.side_effect = [mock_instance1, mock_instance2]

                reporter = UsageReporter(service_token="test-token")

                # Get first client
                client1 = await reporter._get_client()

                # Mark as closed
                mock_instance1.is_closed = True

                # Get second client (should create new)
                client2 = await reporter._get_client()

                assert mock_client_class.call_count == 2
                assert client1 != client2

    @pytest.mark.asyncio
    async def test_close_client(self):
        """Test that close() properly closes the client."""
        with patch.dict(os.environ, {
            "NODE_ENV": "development",
            "BYPASS_USAGE_CHECKS": "",
        }):
            from usage_reporter import UsageReporter

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_instance = AsyncMock()
                mock_instance.is_closed = False
                mock_instance.aclose = AsyncMock()
                mock_client_class.return_value = mock_instance

                reporter = UsageReporter(service_token="test-token")

                # Get client first
                await reporter._get_client()

                # Close it
                await reporter.close()

                mock_instance.aclose.assert_called_once()
                assert reporter._client is None


# ============================================================================
# Thread Safety Tests
# ============================================================================

class TestThreadSafety:
    """Tests for thread safety with asyncio locks."""

    @pytest.mark.asyncio
    async def test_concurrent_get_client_creates_once(self):
        """Test that concurrent _get_client calls only create one client."""
        with patch.dict(os.environ, {
            "NODE_ENV": "development",
            "BYPASS_USAGE_CHECKS": "",
        }):
            from usage_reporter import UsageReporter

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_instance = AsyncMock()
                mock_instance.is_closed = False
                mock_client_class.return_value = mock_instance

                reporter = UsageReporter(service_token="test-token")

                # Launch many concurrent _get_client calls
                tasks = [reporter._get_client() for _ in range(50)]
                clients = await asyncio.gather(*tasks)

                # Should only create once due to locking
                mock_client_class.assert_called_once()

                # All should return the same instance
                assert all(c == mock_instance for c in clients)


# ============================================================================
# Report Meeting Minutes Tests
# ============================================================================

class TestReportMeetingMinutes:
    """Tests for report_meeting_minutes method."""

    @pytest.mark.asyncio
    async def test_report_meeting_minutes_success(self, mock_httpx_client, mock_response_success):
        """Test successful meeting minutes reporting."""
        with patch.dict(os.environ, {
            "NODE_ENV": "development",
            "BYPASS_USAGE_CHECKS": "",
        }):
            from usage_reporter import UsageReporter

            mock_httpx_client.post.return_value = mock_response_success

            with patch("httpx.AsyncClient", return_value=mock_httpx_client):
                reporter = UsageReporter(
                    frontend_url="http://test:3000",
                    service_token="test-token",
                )

                result = await reporter.report_meeting_minutes(
                    user_id="user-123",
                    minutes=10,
                    room_id="room-456",
                )

                assert result.success is True
                assert result.event_type == "meeting-minutes"
                assert result.value == 10

    @pytest.mark.asyncio
    async def test_report_zero_minutes_skipped(self):
        """Test that zero minutes are not reported."""
        with patch.dict(os.environ, {
            "NODE_ENV": "development",
            "BYPASS_USAGE_CHECKS": "",
        }):
            from usage_reporter import UsageReporter

            reporter = UsageReporter(service_token="test-token")

            result = await reporter.report_meeting_minutes("user-123", 0)

            assert result.success is True
            assert result.value == 0

    @pytest.mark.asyncio
    async def test_report_negative_minutes_skipped(self):
        """Test that negative minutes are not reported."""
        with patch.dict(os.environ, {
            "NODE_ENV": "development",
            "BYPASS_USAGE_CHECKS": "",
        }):
            from usage_reporter import UsageReporter

            reporter = UsageReporter(service_token="test-token")

            result = await reporter.report_meeting_minutes("user-123", -5)

            assert result.success is True
            assert result.value == 0

    @pytest.mark.asyncio
    async def test_report_without_service_token(self):
        """Test that reporting fails without service token."""
        with patch.dict(os.environ, {
            "NODE_ENV": "development",
            "BYPASS_USAGE_CHECKS": "",
        }):
            from usage_reporter import UsageReporter

            reporter = UsageReporter(service_token="")

            result = await reporter.report_meeting_minutes("user-123", 10)

            assert result.success is False
            assert "INTERNAL_SERVICE_TOKEN" in result.error


# ============================================================================
# Check Meeting Limits Tests (Fail Closed)
# ============================================================================

class TestCheckMeetingLimitsFailClosed:
    """Tests for fail-closed behavior in check_meeting_limits."""

    @pytest.mark.asyncio
    async def test_fail_closed_on_http_error(self, mock_httpx_client):
        """Test that HTTP errors result in denial (fail closed)."""
        with patch.dict(os.environ, {
            "NODE_ENV": "production",  # Production mode
            "BYPASS_USAGE_CHECKS": "",
        }):
            from usage_reporter import UsageReporter

            error_response = MagicMock()
            error_response.status_code = 500
            error_response.json.return_value = {"error": "Server error"}
            mock_httpx_client.get.return_value = error_response

            with patch("httpx.AsyncClient", return_value=mock_httpx_client):
                reporter = UsageReporter(
                    frontend_url="http://test:3000",
                    service_token="test-token",
                )

                allowed, status = await reporter.check_meeting_limits("user-123")

                assert allowed is False
                assert "unavailable" in status.reason.lower()

    @pytest.mark.asyncio
    async def test_fail_closed_on_timeout(self, mock_httpx_client):
        """Test that timeouts result in denial (fail closed)."""
        with patch.dict(os.environ, {
            "NODE_ENV": "production",
            "BYPASS_USAGE_CHECKS": "",
        }):
            from usage_reporter import UsageReporter
            import httpx

            mock_httpx_client.get.side_effect = httpx.TimeoutException("Timeout")

            with patch("httpx.AsyncClient", return_value=mock_httpx_client):
                reporter = UsageReporter(
                    frontend_url="http://test:3000",
                    service_token="test-token",
                )

                allowed, status = await reporter.check_meeting_limits("user-123")

                assert allowed is False
                assert "timeout" in status.reason.lower()

    @pytest.mark.asyncio
    async def test_fail_closed_on_exception(self, mock_httpx_client):
        """Test that exceptions result in denial (fail closed)."""
        with patch.dict(os.environ, {
            "NODE_ENV": "production",
            "BYPASS_USAGE_CHECKS": "",
        }):
            from usage_reporter import UsageReporter

            mock_httpx_client.get.side_effect = Exception("Unknown error")

            with patch("httpx.AsyncClient", return_value=mock_httpx_client):
                reporter = UsageReporter(
                    frontend_url="http://test:3000",
                    service_token="test-token",
                )

                allowed, status = await reporter.check_meeting_limits("user-123")

                assert allowed is False
                assert "unavailable" in status.reason.lower()

    @pytest.mark.asyncio
    async def test_fail_closed_no_token_production(self):
        """Test that missing token in production denies access."""
        with patch.dict(os.environ, {
            "NODE_ENV": "production",
            "BYPASS_USAGE_CHECKS": "",
        }):
            from usage_reporter import UsageReporter

            reporter = UsageReporter(service_token="")

            allowed, status = await reporter.check_meeting_limits("user-123")

            assert allowed is False
            assert "configuration error" in status.reason.lower()


# ============================================================================
# Check Email Draft Limits Tests
# ============================================================================

class TestCheckEmailDraftLimits:
    """Tests for check_email_draft_limits method."""

    @pytest.mark.asyncio
    async def test_fail_closed_on_error(self, mock_httpx_client):
        """Test that errors result in denial."""
        with patch.dict(os.environ, {
            "NODE_ENV": "production",
            "BYPASS_USAGE_CHECKS": "",
        }):
            from usage_reporter import UsageReporter

            mock_httpx_client.get.side_effect = Exception("Network error")

            with patch("httpx.AsyncClient", return_value=mock_httpx_client):
                reporter = UsageReporter(
                    frontend_url="http://test:3000",
                    service_token="test-token",
                )

                allowed, status = await reporter.check_email_draft_limits("user-123")

                assert allowed is False
                assert "unavailable" in status.get("reason", "").lower()


# ============================================================================
# Headers Tests
# ============================================================================

class TestHeaders:
    """Tests for request header construction."""

    def test_headers_include_bearer_token(self):
        """Test that headers include Bearer token."""
        with patch.dict(os.environ, {
            "NODE_ENV": "development",
            "BYPASS_USAGE_CHECKS": "",
        }):
            from usage_reporter import UsageReporter

            reporter = UsageReporter(service_token="my-secret-token")
            headers = reporter._get_headers()

            assert headers["Authorization"] == "Bearer my-secret-token"
            assert headers["Content-Type"] == "application/json"


# ============================================================================
# Singleton Tests
# ============================================================================

class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_usage_reporter_returns_same_instance(self):
        """Test that get_usage_reporter returns the same instance."""
        with patch.dict(os.environ, {
            "NODE_ENV": "development",
            "BYPASS_USAGE_CHECKS": "",
        }):
            from usage_reporter import get_usage_reporter, _usage_reporter
            import usage_reporter as ur

            # Reset singleton
            ur._usage_reporter = None

            reporter1 = get_usage_reporter()
            reporter2 = get_usage_reporter()

            assert reporter1 is reporter2

    @pytest.mark.asyncio
    async def test_close_usage_reporter(self):
        """Test that close_usage_reporter cleans up properly."""
        with patch.dict(os.environ, {
            "NODE_ENV": "development",
            "BYPASS_USAGE_CHECKS": "",
        }):
            from usage_reporter import get_usage_reporter, close_usage_reporter
            import usage_reporter as ur

            # Reset and get reporter
            ur._usage_reporter = None
            reporter = get_usage_reporter()

            # Close it
            await close_usage_reporter()

            assert ur._usage_reporter is None


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
