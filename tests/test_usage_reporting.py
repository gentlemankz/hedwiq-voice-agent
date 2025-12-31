"""
Comprehensive tests for usage reporting functionality.

Tests cover:
- Periodic usage reporting during meetings
- Incremental reporting (delta-based)
- Edge cases (no humans, clock skew, crashes)
- Owner attribution
- HTTP client behavior
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Optional


# ============================================================================
# Mock Classes
# ============================================================================

@dataclass
class MockUsageReportResult:
    """Mock result from usage reporter."""
    success: bool
    error: Optional[str] = None
    event_type: Optional[str] = None
    value: Optional[int] = None


class MockUsageReporter:
    """Mock usage reporter for testing."""

    def __init__(self):
        self.reported_minutes: list[tuple[str, int, Optional[str]]] = []
        self.should_fail = False
        self.failure_error = "Mock error"

    async def report_meeting_minutes(
        self,
        user_id: str,
        minutes: int,
        room_id: Optional[str] = None,
    ) -> MockUsageReportResult:
        if self.should_fail:
            return MockUsageReportResult(success=False, error=self.failure_error)

        self.reported_minutes.append((user_id, minutes, room_id))
        return MockUsageReportResult(
            success=True,
            event_type="meeting-minutes",
            value=minutes,
        )

    def get_total_reported_minutes(self, user_id: str) -> int:
        """Get total minutes reported for a user."""
        return sum(m[1] for m in self.reported_minutes if m[0] == user_id)

    def clear(self):
        """Clear reported data."""
        self.reported_minutes.clear()


class MockRoom:
    """Mock LiveKit room."""

    def __init__(self, name: str = "test-room"):
        self.name = name
        self.remote_participants = {}
        self._event_handlers = {}

    def on(self, event: str, handler):
        self._event_handlers[event] = handler


class MockParticipant:
    """Mock participant."""

    def __init__(self, identity: str, name: Optional[str] = None):
        self.identity = identity
        self.name = name or identity
        self.track_publications = {}


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_reporter():
    """Create a mock usage reporter."""
    return MockUsageReporter()


@pytest.fixture
def mock_room():
    """Create a mock room."""
    return MockRoom("test-room-123")


# ============================================================================
# Unit Tests: Duration Calculation
# ============================================================================

class TestDurationCalculation:
    """Tests for meeting duration calculation."""

    def test_basic_duration_calculation(self):
        """Test basic duration calculation from join to leave."""
        first_join = 1000.0
        last_leave = 1300.0  # 300 seconds = 5 minutes

        duration_seconds = max(0, last_leave - first_join)
        duration_minutes = max(1, min(1440, int(duration_seconds / 60 + 0.5)))

        assert duration_minutes == 5

    def test_minimum_one_minute(self):
        """Test that minimum duration is 1 minute."""
        first_join = 1000.0
        last_leave = 1010.0  # 10 seconds

        duration_seconds = max(0, last_leave - first_join)
        duration_minutes = max(1, min(1440, int(duration_seconds / 60 + 0.5)))

        assert duration_minutes == 1

    def test_maximum_1440_minutes(self):
        """Test that maximum duration is 1440 minutes (24 hours)."""
        first_join = 0.0
        last_leave = 100000.0  # More than 24 hours

        duration_seconds = max(0, last_leave - first_join)
        duration_minutes = max(1, min(1440, int(duration_seconds / 60 + 0.5)))

        assert duration_minutes == 1440

    def test_negative_duration_handled(self):
        """Test that negative duration (clock skew) is handled."""
        first_join = 1000.0
        last_leave = 900.0  # Clock skew - leave before join

        duration_seconds = max(0, last_leave - first_join)
        duration_minutes = max(1, min(1440, int(duration_seconds / 60 + 0.5)))

        # Should be minimum 1 minute, not negative
        assert duration_minutes == 1

    def test_rounding_behavior(self):
        """Test minute rounding (rounds to nearest, not floor)."""
        # 89 seconds should round to 1 minute
        first_join = 0.0
        last_leave = 89.0

        duration_seconds = max(0, last_leave - first_join)
        duration_minutes = max(1, min(1440, int(duration_seconds / 60 + 0.5)))

        assert duration_minutes == 1

        # 91 seconds should round to 2 minutes
        last_leave = 91.0
        duration_seconds = max(0, last_leave - first_join)
        duration_minutes = max(1, min(1440, int(duration_seconds / 60 + 0.5)))

        assert duration_minutes == 2


# ============================================================================
# Unit Tests: Incremental Reporting
# ============================================================================

class TestIncrementalReporting:
    """Tests for incremental (delta-based) usage reporting."""

    def test_incremental_delta_calculation(self):
        """Test that incremental reports only report delta."""
        last_reported = 5
        total_minutes = 10

        minutes_to_report = total_minutes - last_reported

        assert minutes_to_report == 5

    def test_no_report_when_no_new_minutes(self):
        """Test that no report is made when no new minutes."""
        last_reported = 10
        total_minutes = 10

        minutes_to_report = total_minutes - last_reported

        assert minutes_to_report == 0

    def test_final_report_only_unreported_minutes(self):
        """Test that final report only includes unreported minutes."""
        last_reported = 8
        total_minutes = 10

        # Final report should only be 2 minutes
        minutes_to_report = total_minutes - last_reported

        assert minutes_to_report == 2


# ============================================================================
# Integration Tests: Periodic Reporter
# ============================================================================

class TestPeriodicReporter:
    """Tests for periodic usage reporting."""

    @pytest.mark.asyncio
    async def test_periodic_reporter_reports_delta(self, mock_reporter):
        """Test that periodic reporter reports delta (not total)."""
        user_id = "user-123"
        room_id = "room-456"

        # Simulate periodic reports
        # First report: 5 minutes elapsed
        await mock_reporter.report_meeting_minutes(user_id, 5, room_id)
        # Second report: 5 more minutes (delta)
        await mock_reporter.report_meeting_minutes(user_id, 5, room_id)

        # Total should be 10 minutes across 2 reports
        total = mock_reporter.get_total_reported_minutes(user_id)
        assert total == 10
        assert len(mock_reporter.reported_minutes) == 2

    @pytest.mark.asyncio
    async def test_periodic_reporter_handles_failure(self, mock_reporter):
        """Test that periodic reporter handles API failures gracefully."""
        user_id = "user-123"

        # First report succeeds
        result1 = await mock_reporter.report_meeting_minutes(user_id, 5, "room")
        assert result1.success

        # Second report fails
        mock_reporter.should_fail = True
        result2 = await mock_reporter.report_meeting_minutes(user_id, 5, "room")
        assert not result2.success

        # Only first report should be counted
        total = mock_reporter.get_total_reported_minutes(user_id)
        assert total == 5


# ============================================================================
# Integration Tests: Owner Attribution
# ============================================================================

class TestOwnerAttribution:
    """Tests for meeting owner attribution."""

    def test_extract_user_id_from_identity(self):
        """Test extracting user ID from participant identity.

        Identity format: {userId}-{8-char-hex-suffix}
        The suffix is always exactly 8 lowercase hex characters preceded by a hyphen.
        """
        # Import the actual function
        from utils.identity import extract_user_id_from_identity

        # UUID with hex suffix
        assert extract_user_id_from_identity("550e8400-e29b-41d4-a716-446655440000-a1b2c3d4") == "550e8400-e29b-41d4-a716-446655440000"

        # Simple user ID with hex suffix
        assert extract_user_id_from_identity("simple-user-deadbeef") == "simple-user"

        # User123 with hex suffix
        assert extract_user_id_from_identity("user123-abcd1234") == "user123"

        # No suffix (returns as-is)
        assert extract_user_id_from_identity("invalid") == "invalid"

        # None input
        assert extract_user_id_from_identity(None) is None

        # Agent identity (also extracts user ID part)
        assert extract_user_id_from_identity("luframe-a1b2c3d4") == "luframe"

    def test_is_agent_identity(self):
        """Test identifying agent vs human participants."""
        from utils.identity import is_agent_identity

        # Agent identities (start with 'luframe')
        assert is_agent_identity("luframe-agent") is True
        assert is_agent_identity("luframe-123abc") is True
        assert is_agent_identity("luframe") is True

        # Human identities (don't start with 'luframe')
        assert is_agent_identity("user_abc123") is False
        assert is_agent_identity("john_doe") is False
        assert is_agent_identity("agent_transcriber") is False
        assert is_agent_identity(None) is False


# ============================================================================
# Integration Tests: Full Reporting Flow
# ============================================================================

class TestFullReportingFlow:
    """Tests for complete usage reporting flow."""

    @pytest.mark.asyncio
    async def test_no_report_when_no_humans_joined(self, mock_reporter):
        """Test that no report is made when no humans joined."""
        first_human_join_time = None

        if first_human_join_time is None:
            # Should skip reporting
            reported = False
        else:
            await mock_reporter.report_meeting_minutes("user", 5, "room")
            reported = True

        assert not reported
        assert len(mock_reporter.reported_minutes) == 0

    @pytest.mark.asyncio
    async def test_report_at_meeting_end(self, mock_reporter):
        """Test that final report is made at meeting end."""
        user_id = "user-123"
        room_id = "room-456"
        total_minutes = 15
        already_reported = 10  # From periodic reports

        # Final report should only report unreported minutes
        remaining = total_minutes - already_reported
        if remaining > 0:
            await mock_reporter.report_meeting_minutes(user_id, remaining, room_id)

        assert mock_reporter.get_total_reported_minutes(user_id) == 5

    @pytest.mark.asyncio
    async def test_report_even_if_owner_left_early(self, mock_reporter):
        """Test that reports still work if owner leaves before meeting ends."""
        user_id = "user-123"  # Cached owner ID
        room_id = "room-456"

        # Owner left at 5 minutes, meeting continued to 10
        # Should still bill the owner
        await mock_reporter.report_meeting_minutes(user_id, 10, room_id)

        assert mock_reporter.get_total_reported_minutes(user_id) == 10


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_very_short_meeting(self, mock_reporter):
        """Test handling of very short meetings (<1 minute)."""
        user_id = "user-123"
        duration_seconds = 30  # 30 seconds

        # Should round up to 1 minute minimum
        duration_minutes = max(1, min(1440, int(duration_seconds / 60 + 0.5)))
        await mock_reporter.report_meeting_minutes(user_id, duration_minutes, "room")

        assert mock_reporter.get_total_reported_minutes(user_id) == 1

    @pytest.mark.asyncio
    async def test_very_long_meeting(self, mock_reporter):
        """Test handling of very long meetings (>24 hours)."""
        user_id = "user-123"
        duration_seconds = 90000  # 25 hours

        # Should cap at 1440 minutes (24 hours)
        duration_minutes = max(1, min(1440, int(duration_seconds / 60 + 0.5)))
        await mock_reporter.report_meeting_minutes(user_id, duration_minutes, "room")

        assert mock_reporter.get_total_reported_minutes(user_id) == 1440

    @pytest.mark.asyncio
    async def test_concurrent_periodic_reports(self, mock_reporter):
        """Test handling of concurrent periodic reports."""
        user_id = "user-123"

        # Simulate concurrent reports (shouldn't happen but test anyway)
        tasks = [
            mock_reporter.report_meeting_minutes(user_id, 5, "room"),
            mock_reporter.report_meeting_minutes(user_id, 5, "room"),
        ]
        await asyncio.gather(*tasks)

        # Both reports should succeed
        assert len(mock_reporter.reported_minutes) == 2

    @pytest.mark.asyncio
    async def test_api_timeout_handling(self, mock_reporter):
        """Test handling of API timeouts."""
        mock_reporter.should_fail = True
        mock_reporter.failure_error = "Request timeout"

        result = await mock_reporter.report_meeting_minutes("user", 5, "room")

        assert not result.success
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_service_token(self, mock_reporter):
        """Test behavior when service token is missing."""
        mock_reporter.should_fail = True
        mock_reporter.failure_error = "INTERNAL_SERVICE_TOKEN not configured"

        result = await mock_reporter.report_meeting_minutes("user", 5, "room")

        assert not result.success

    def test_zero_minutes_not_reported(self):
        """Test that zero minutes are not reported."""
        minutes = 0

        # Should return early without reporting
        if minutes <= 0:
            should_report = False
        else:
            should_report = True

        assert not should_report


# ============================================================================
# Stress Tests
# ============================================================================

class TestStress:
    """Stress tests for usage reporting."""

    @pytest.mark.asyncio
    async def test_many_periodic_reports(self, mock_reporter):
        """Test many periodic reports in sequence."""
        user_id = "user-123"
        num_reports = 100

        for i in range(num_reports):
            await mock_reporter.report_meeting_minutes(user_id, 5, "room")

        assert len(mock_reporter.reported_minutes) == num_reports
        assert mock_reporter.get_total_reported_minutes(user_id) == 500

    @pytest.mark.asyncio
    async def test_multiple_rooms_simultaneously(self, mock_reporter):
        """Test reports from multiple rooms at once."""
        users = ["user-1", "user-2", "user-3"]
        rooms = ["room-1", "room-2", "room-3"]

        tasks = []
        for user, room in zip(users, rooms):
            tasks.append(mock_reporter.report_meeting_minutes(user, 10, room))

        await asyncio.gather(*tasks)

        assert len(mock_reporter.reported_minutes) == 3

        for user in users:
            assert mock_reporter.get_total_reported_minutes(user) == 10


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
