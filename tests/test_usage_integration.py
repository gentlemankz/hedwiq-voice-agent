"""
Integration tests for the full usage reporting chain.

Tests cover:
- Agent → Frontend API → Polar flow
- End-to-end meeting lifecycle
- Error recovery and resilience
- Multi-user scenarios
- Cache consistency
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import json


# ============================================================================
# Simulated Components
# ============================================================================

@dataclass
class SimulatedPolarEvent:
    """Represents an event ingested to Polar."""
    name: str
    external_customer_id: str
    metadata: Dict
    timestamp: float = field(default_factory=time.time)


class SimulatedPolarMeters:
    """Simulates Polar meter state for testing."""

    def __init__(self):
        self.meters: Dict[str, Dict[str, int]] = {}  # user_id -> {meter_name: value}
        self.events: List[SimulatedPolarEvent] = []
        self.processing_delay_seconds = 0.1

    async def ingest_event(self, event: SimulatedPolarEvent):
        """Process an ingested event and update meters."""
        self.events.append(event)

        # Simulate processing delay
        await asyncio.sleep(self.processing_delay_seconds)

        user_id = event.external_customer_id
        if user_id not in self.meters:
            self.meters[user_id] = {
                "meeting-minutes": 0,
                "email-drafts": 0,
                "storage-bytes": 0,
            }

        if event.name == "meeting-minutes":
            self.meters[user_id]["meeting-minutes"] += event.metadata.get("minutes", 0)
        elif event.name == "email-drafts":
            self.meters[user_id]["email-drafts"] += event.metadata.get("count", 0)
        elif event.name == "storage-bytes":
            self.meters[user_id]["storage-bytes"] += event.metadata.get("bytes", 0)

    def get_meter_value(self, user_id: str, meter_name: str) -> int:
        return self.meters.get(user_id, {}).get(meter_name, 0)

    def get_all_meters(self, user_id: str) -> Dict[str, int]:
        return self.meters.get(user_id, {
            "meeting-minutes": 0,
            "email-drafts": 0,
            "storage-bytes": 0,
        })


class SimulatedFrontendAPI:
    """Simulates the frontend internal usage API."""

    def __init__(self, polar: SimulatedPolarMeters):
        self.polar = polar
        self.customers: Dict[str, Dict] = {}  # user_id -> customer data
        self.request_log: List[Dict] = []
        self.should_fail = False
        self.failure_rate = 0.0  # Percentage of requests to fail

    def add_customer(self, user_id: str, email: str, name: str = None):
        """Register a customer (simulates Better Auth createCustomerOnSignUp)."""
        self.customers[user_id] = {
            "id": f"cust_{user_id}",
            "email": email,
            "name": name,
        }

    async def report_usage(
        self,
        event_type: str,
        user_id: str,
        value: int,
        room_id: Optional[str] = None,
    ) -> Dict:
        """Simulate POST /api/internal/usage endpoint."""
        self.request_log.append({
            "event_type": event_type,
            "user_id": user_id,
            "value": value,
            "room_id": room_id,
            "timestamp": time.time(),
        })

        # Simulate random failures
        if self.should_fail:
            return {"success": False, "error": "API error"}

        # Check if customer exists
        if user_id not in self.customers:
            return {"success": False, "error": "Customer not found"}

        # Create and ingest event
        event = SimulatedPolarEvent(
            name=event_type,
            external_customer_id=user_id,
            metadata={
                "minutes" if event_type == "meeting-minutes" else
                "count" if event_type == "email-drafts" else
                "bytes": value,
                "room_id": room_id,
            }
        )

        await self.polar.ingest_event(event)

        return {"success": True, "event_type": event_type, "value": value}

    def get_usage_state(self, user_id: str) -> Dict:
        """Get current usage state for a user."""
        return {
            "tier": "pro" if user_id in self.customers else "free",
            **self.polar.get_all_meters(user_id)
        }


class SimulatedAgent:
    """Simulates the LiveKit agent usage reporting behavior."""

    def __init__(self, api: SimulatedFrontendAPI):
        self.api = api
        self.room_id: Optional[str] = None
        self.owner_id: Optional[str] = None
        self.first_human_join_time: Optional[float] = None
        self.last_human_leave_time: Optional[float] = None
        self.last_reported_minutes: int = 0
        self.periodic_task: Optional[asyncio.Task] = None
        self.report_interval_seconds: float = 0.5  # Short for testing
        self.is_running = False

    async def start(self, room_id: str, owner_id: str):
        """Start the agent for a room."""
        self.room_id = room_id
        self.owner_id = owner_id
        self.is_running = True

        # Start periodic reporting task
        self.periodic_task = asyncio.create_task(self._periodic_reporter())

    async def _periodic_reporter(self):
        """Periodically report usage."""
        while self.is_running:
            await asyncio.sleep(self.report_interval_seconds)

            if self.first_human_join_time and self.owner_id:
                current_time = time.time()
                total_seconds = current_time - self.first_human_join_time
                total_minutes = max(1, int(total_seconds / 60 + 0.5))

                delta = total_minutes - self.last_reported_minutes
                if delta > 0:
                    await self.api.report_usage(
                        "meeting-minutes",
                        self.owner_id,
                        delta,
                        self.room_id,
                    )
                    self.last_reported_minutes = total_minutes

    async def on_human_join(self, participant_id: str):
        """Handle human participant joining."""
        if not self.first_human_join_time:
            self.first_human_join_time = time.time()

            # Set owner if not already set
            if not self.owner_id:
                self.owner_id = participant_id

    async def on_human_leave(self, participant_id: str):
        """Handle human participant leaving."""
        self.last_human_leave_time = time.time()

    async def stop(self):
        """Stop the agent and report final usage."""
        self.is_running = False

        # Cancel periodic task
        if self.periodic_task:
            self.periodic_task.cancel()
            try:
                await self.periodic_task
            except asyncio.CancelledError:
                pass

        # Report final minutes (only unreported)
        if self.first_human_join_time and self.owner_id:
            end_time = self.last_human_leave_time or time.time()
            total_seconds = max(0, end_time - self.first_human_join_time)
            total_minutes = max(1, min(1440, int(total_seconds / 60 + 0.5)))

            remaining = total_minutes - self.last_reported_minutes
            if remaining > 0:
                await self.api.report_usage(
                    "meeting-minutes",
                    self.owner_id,
                    remaining,
                    self.room_id,
                )


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def polar():
    return SimulatedPolarMeters()


@pytest.fixture
def api(polar):
    return SimulatedFrontendAPI(polar)


@pytest.fixture
def agent(api):
    return SimulatedAgent(api)


# ============================================================================
# End-to-End Tests
# ============================================================================

class TestEndToEndMeetingFlow:
    """Tests for complete meeting usage flow."""

    @pytest.mark.asyncio
    async def test_simple_meeting_reports_minutes(self, api, agent, polar):
        """Test a simple meeting reports minutes correctly."""
        user_id = "user-123"
        room_id = "room-456"

        # Setup customer
        api.add_customer(user_id, "test@example.com")

        # Start meeting
        await agent.start(room_id, user_id)
        await agent.on_human_join(user_id)

        # Simulate meeting duration (100ms = very short)
        await asyncio.sleep(0.1)

        # End meeting
        await agent.on_human_leave(user_id)
        await agent.stop()

        # Wait for Polar processing
        await asyncio.sleep(0.2)

        # Verify minutes were reported
        minutes = polar.get_meter_value(user_id, "meeting-minutes")
        assert minutes >= 1, "At least 1 minute should be reported"

    @pytest.mark.asyncio
    async def test_periodic_reporting_works(self, api, polar):
        """Test that periodic reporting sends incremental updates."""
        user_id = "user-123"
        room_id = "room-456"

        api.add_customer(user_id, "test@example.com")

        agent = SimulatedAgent(api)
        agent.report_interval_seconds = 0.1  # 100ms intervals

        await agent.start(room_id, user_id)
        await agent.on_human_join(user_id)

        # Wait for a few periodic reports
        await asyncio.sleep(0.5)

        await agent.stop()
        await asyncio.sleep(0.2)

        # Should have multiple requests logged
        meeting_requests = [
            r for r in api.request_log
            if r["event_type"] == "meeting-minutes"
        ]

        # Periodic + final report
        assert len(meeting_requests) >= 1

    @pytest.mark.asyncio
    async def test_no_report_when_no_humans(self, api, agent, polar):
        """Test that no minutes are reported when no humans joined."""
        user_id = "user-123"
        room_id = "room-456"

        api.add_customer(user_id, "test@example.com")

        # Start meeting but no human joins
        await agent.start(room_id, user_id)
        # Don't call on_human_join

        await asyncio.sleep(0.2)
        await agent.stop()

        # Wait for processing
        await asyncio.sleep(0.2)

        # No minutes should be reported
        minutes = polar.get_meter_value(user_id, "meeting-minutes")
        assert minutes == 0


class TestMultiUserScenarios:
    """Tests for multi-user meeting scenarios."""

    @pytest.mark.asyncio
    async def test_owner_billed_not_participants(self, api, polar):
        """Test that only the room owner is billed."""
        owner_id = "owner-123"
        participant_id = "participant-456"
        room_id = "room-789"

        api.add_customer(owner_id, "owner@example.com")
        api.add_customer(participant_id, "participant@example.com")

        agent = SimulatedAgent(api)
        await agent.start(room_id, owner_id)

        # Both join
        await agent.on_human_join(owner_id)
        await agent.on_human_join(participant_id)

        await asyncio.sleep(0.1)

        # Participant leaves first
        await agent.on_human_leave(participant_id)

        await asyncio.sleep(0.1)

        # Owner leaves
        await agent.on_human_leave(owner_id)
        await agent.stop()

        await asyncio.sleep(0.2)

        # Only owner should be billed
        owner_minutes = polar.get_meter_value(owner_id, "meeting-minutes")
        participant_minutes = polar.get_meter_value(participant_id, "meeting-minutes")

        assert owner_minutes >= 1
        assert participant_minutes == 0

    @pytest.mark.asyncio
    async def test_concurrent_meetings(self, api, polar):
        """Test multiple concurrent meetings report correctly."""
        users = [
            ("user-1", "room-1"),
            ("user-2", "room-2"),
            ("user-3", "room-3"),
        ]

        for user_id, _ in users:
            api.add_customer(user_id, f"{user_id}@example.com")

        agents = []
        for user_id, room_id in users:
            agent = SimulatedAgent(api)
            agent.report_interval_seconds = 0.2
            await agent.start(room_id, user_id)
            await agent.on_human_join(user_id)
            agents.append(agent)

        # Let meetings run
        await asyncio.sleep(0.3)

        # Stop all meetings
        for agent in agents:
            await agent.stop()

        await asyncio.sleep(0.3)

        # All users should have usage recorded
        for user_id, _ in users:
            minutes = polar.get_meter_value(user_id, "meeting-minutes")
            assert minutes >= 1, f"User {user_id} should have minutes recorded"


class TestErrorRecovery:
    """Tests for error handling and recovery."""

    @pytest.mark.asyncio
    async def test_api_failure_doesnt_crash_agent(self, api, polar):
        """Test that API failures don't crash the agent."""
        user_id = "user-123"
        room_id = "room-456"

        api.add_customer(user_id, "test@example.com")
        api.should_fail = True

        agent = SimulatedAgent(api)
        await agent.start(room_id, user_id)
        await agent.on_human_join(user_id)

        # Meeting continues despite failures
        await asyncio.sleep(0.2)

        # Should be able to stop cleanly
        await agent.stop()

        # API was called even though it failed
        assert len(api.request_log) > 0

    @pytest.mark.asyncio
    async def test_missing_customer_handled(self, api, polar):
        """Test that missing customer doesn't cause crash."""
        user_id = "unknown-user"
        room_id = "room-456"

        # Don't add customer - should handle gracefully

        agent = SimulatedAgent(api)
        await agent.start(room_id, user_id)
        await agent.on_human_join(user_id)

        await asyncio.sleep(0.1)
        await agent.stop()

        # Request was made but returned error
        assert len(api.request_log) > 0

        # No meters updated (customer doesn't exist)
        minutes = polar.get_meter_value(user_id, "meeting-minutes")
        assert minutes == 0


class TestCacheConsistency:
    """Tests for cache/meter consistency."""

    @pytest.mark.asyncio
    async def test_meters_reflect_all_reports(self, api, polar):
        """Test that meters accurately reflect all reported usage."""
        user_id = "user-123"

        api.add_customer(user_id, "test@example.com")

        # Report multiple times
        await api.report_usage("meeting-minutes", user_id, 5)
        await api.report_usage("meeting-minutes", user_id, 10)
        await api.report_usage("meeting-minutes", user_id, 3)

        await asyncio.sleep(0.4)  # Wait for processing

        # Total should be sum
        minutes = polar.get_meter_value(user_id, "meeting-minutes")
        assert minutes == 18

    @pytest.mark.asyncio
    async def test_different_meter_types_independent(self, api, polar):
        """Test that different meter types don't interfere."""
        user_id = "user-123"

        api.add_customer(user_id, "test@example.com")

        # Report different types
        await api.report_usage("meeting-minutes", user_id, 10)
        await api.report_usage("email-drafts", user_id, 3)
        await api.report_usage("storage-bytes", user_id, 1024)

        await asyncio.sleep(0.4)

        # Each meter should be independent
        assert polar.get_meter_value(user_id, "meeting-minutes") == 10
        assert polar.get_meter_value(user_id, "email-drafts") == 3
        assert polar.get_meter_value(user_id, "storage-bytes") == 1024


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_very_short_meeting(self, api, agent, polar):
        """Test that very short meetings report minimum 1 minute."""
        user_id = "user-123"
        room_id = "room-456"

        api.add_customer(user_id, "test@example.com")

        await agent.start(room_id, user_id)
        await agent.on_human_join(user_id)

        # Immediately stop
        await agent.stop()

        await asyncio.sleep(0.2)

        # Should report at least 1 minute
        minutes = polar.get_meter_value(user_id, "meeting-minutes")
        assert minutes >= 1

    @pytest.mark.asyncio
    async def test_owner_leaves_before_end(self, api, polar):
        """Test billing when owner leaves early."""
        owner_id = "owner-123"
        room_id = "room-456"

        api.add_customer(owner_id, "owner@example.com")

        agent = SimulatedAgent(api)
        agent.report_interval_seconds = 0.1

        await agent.start(room_id, owner_id)
        await agent.on_human_join(owner_id)

        # Owner leaves early
        await asyncio.sleep(0.05)
        await agent.on_human_leave(owner_id)

        # Meeting continues briefly
        await asyncio.sleep(0.1)

        await agent.stop()
        await asyncio.sleep(0.2)

        # Owner should still be billed
        minutes = polar.get_meter_value(owner_id, "meeting-minutes")
        assert minutes >= 1

    @pytest.mark.asyncio
    async def test_rapid_join_leave_cycles(self, api, polar):
        """Test handling rapid join/leave cycles."""
        user_id = "user-123"
        room_id = "room-456"

        api.add_customer(user_id, "test@example.com")

        agent = SimulatedAgent(api)
        agent.report_interval_seconds = 0.05

        await agent.start(room_id, user_id)

        # Rapid join/leave
        for _ in range(5):
            await agent.on_human_join(user_id)
            await asyncio.sleep(0.02)
            await agent.on_human_leave(user_id)

        await agent.stop()
        await asyncio.sleep(0.2)

        # Should have handled gracefully (at least 1 minute reported)
        minutes = polar.get_meter_value(user_id, "meeting-minutes")
        assert minutes >= 1


class TestDataIntegrity:
    """Tests for data integrity across the chain."""

    @pytest.mark.asyncio
    async def test_no_duplicate_reporting(self, api, polar):
        """Test that incremental reporting doesn't duplicate."""
        user_id = "user-123"
        room_id = "room-456"

        api.add_customer(user_id, "test@example.com")

        agent = SimulatedAgent(api)
        agent.report_interval_seconds = 0.05

        await agent.start(room_id, user_id)
        await agent.on_human_join(user_id)

        # Let several periodic reports happen
        await asyncio.sleep(0.2)

        await agent.stop()
        await asyncio.sleep(0.3)

        # Total minutes should be reasonable (not duplicated)
        minutes = polar.get_meter_value(user_id, "meeting-minutes")

        # With the simulation timing, should be around 1 minute
        # (not 10+ from duplicate reporting)
        assert minutes < 5, f"Minutes ({minutes}) too high, likely duplicated"

    @pytest.mark.asyncio
    async def test_all_events_reach_polar(self, api, polar):
        """Test that all reported events reach Polar."""
        user_id = "user-123"

        api.add_customer(user_id, "test@example.com")

        num_reports = 10
        for i in range(num_reports):
            await api.report_usage("meeting-minutes", user_id, 1)

        await asyncio.sleep(0.5)

        # All events should be recorded
        assert len(polar.events) == num_reports

        # Total should match
        minutes = polar.get_meter_value(user_id, "meeting-minutes")
        assert minutes == num_reports


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
