"""
Usage Reporter Module

Reports usage data to Polar via the frontend's internal usage API.
Used by the agent to track meeting minutes for billing.

Usage:
    from usage_reporter import UsageReporter

    reporter = UsageReporter()

    # Report meeting minutes
    await reporter.report_meeting_minutes(user_id, minutes, room_id=room_id)

    # Check user limits before starting
    can_start, status = await reporter.check_meeting_limits(user_id)
"""

import os
import asyncio
import logging
import warnings
from typing import Optional
from dataclasses import dataclass
import httpx

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
INTERNAL_SERVICE_TOKEN = os.getenv("INTERNAL_SERVICE_TOKEN", "")

# Environment flag to bypass usage checks in development.
# Set BYPASS_USAGE_CHECKS=true for local testing without frontend API.
# Uses NODE_ENV for consistency with frontend (defaults to "development" if not set).
BYPASS_USAGE_CHECKS = os.getenv("BYPASS_USAGE_CHECKS", "").lower() == "true"
IS_DEVELOPMENT = os.getenv("NODE_ENV", "development").lower() != "production"

# C2 Fix: CRITICAL production guard to prevent billing bypass
# This check runs at module load time to fail fast
if BYPASS_USAGE_CHECKS:
    if not IS_DEVELOPMENT:
        raise RuntimeError(
            "FATAL: BYPASS_USAGE_CHECKS=true is not allowed in production. "
            "This would disable all billing. Remove this environment variable."
        )
    else:
        warnings.warn(
            "BYPASS_USAGE_CHECKS is enabled - usage will NOT be billed. "
            "This should only be used for local development.",
            UserWarning,
            stacklevel=1
        )

# Usage event types (must match frontend USAGE_EVENTS)
MEETING_MINUTES = "meeting-minutes"
EMAIL_DRAFTS = "email-drafts"
STORAGE_BYTES = "storage-bytes"

# L1 Fix: Define named constants instead of magic numbers
FREE_TIER_MINUTES_LIMIT = 300  # Default free tier monthly minutes
DEV_EMAIL_DRAFTS_LIMIT = 10   # Synthetic limit for dev testing
PARTICIPANT_WAIT_TIMEOUT_SECONDS = 3.0  # Wait time for participant identification

# Retry configuration for usage reporting
MAX_RETRY_ATTEMPTS = 3  # Number of retry attempts
INITIAL_RETRY_DELAY_SECONDS = 1.0  # Initial delay before first retry
MAX_RETRY_DELAY_SECONDS = 10.0  # Maximum delay between retries


# ============================================================================
# Types
# ============================================================================

@dataclass
class UsageReportResult:
    """Result of a usage report operation."""
    success: bool
    error: Optional[str] = None
    event_type: Optional[str] = None
    value: Optional[int] = None


@dataclass
class MeetingLimitStatus:
    """Status of a user's meeting limits."""
    allowed: bool
    tier: str
    minutes_used: int
    minutes_limit: int
    remaining_minutes: int
    reason: Optional[str] = None


# ============================================================================
# Usage Reporter Class
# ============================================================================

class UsageReporter:
    """
    Reports usage events to Polar via the frontend's internal API.

    The frontend proxies these requests to Polar with proper authentication.
    """

    def __init__(
        self,
        frontend_url: Optional[str] = None,
        service_token: Optional[str] = None,
        timeout: float = 10.0,
    ):
        self.frontend_url = frontend_url or FRONTEND_URL
        self.service_token = service_token or INTERNAL_SERVICE_TOKEN
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        # H3 Fix: Add lock to prevent race condition in HTTP client creation
        self._client_lock = asyncio.Lock()

        if not self.service_token:
            logger.warning(
                "[UsageReporter] INTERNAL_SERVICE_TOKEN not configured. "
                "Usage reporting will fail."
            )

    async def _get_client(self) -> httpx.AsyncClient:
        """
        Get or create HTTP client (thread-safe).

        H3 Fix: Uses asyncio.Lock to prevent race condition where multiple
        concurrent calls could create multiple clients, causing resource leaks.
        """
        async with self._client_lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(timeout=self.timeout)
            return self._client

    async def close(self) -> None:
        """
        Close the HTTP client (thread-safe).

        L2 Fix: Ensure proper cleanup with lock to prevent race conditions.
        """
        async with self._client_lock:
            if self._client and not self._client.is_closed:
                await self._client.aclose()
                self._client = None

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with authentication."""
        return {
            "Authorization": f"Bearer {self.service_token}",
            "Content-Type": "application/json",
        }

    async def _report_usage(
        self,
        user_id: str,
        event_type: str,
        value: int,
        metadata: Optional[dict] = None,
    ) -> UsageReportResult:
        """
        Report a usage event to the frontend API.

        Args:
            user_id: The user's ID
            event_type: Type of usage event
            value: The value to report
            metadata: Optional additional metadata

        Returns:
            UsageReportResult with success status
        """
        # Debug: Log request details
        logger.debug(
            f"[UsageReporter] Preparing request: "
            f"user={user_id}, event={event_type}, value={value}, "
            f"url={self.frontend_url}/api/internal/usage, "
            f"token_configured={bool(self.service_token)}"
        )

        if not self.service_token:
            logger.error(
                "[UsageReporter] INTERNAL_SERVICE_TOKEN not configured. "
                "Cannot report usage. Set this in your environment."
            )
            return UsageReportResult(
                success=False,
                error="INTERNAL_SERVICE_TOKEN not configured",
            )

        try:
            client = await self._get_client()

            payload = {
                "userId": user_id,
                "eventType": event_type,
                "value": value,
            }

            if metadata:
                payload["metadata"] = metadata

            logger.debug(f"[UsageReporter] Sending POST to {self.frontend_url}/api/internal/usage")

            response = await client.post(
                f"{self.frontend_url}/api/internal/usage",
                json=payload,
                headers=self._get_headers(),
            )

            logger.debug(f"[UsageReporter] Response status: {response.status_code}")

            if response.status_code == 200 or response.status_code == 201:
                data = response.json()
                logger.info(
                    f"[UsageReporter] SUCCESS: Reported {event_type}={value} "
                    f"for user {user_id} (response: {data})"
                )
                return UsageReportResult(
                    success=True,
                    event_type=event_type,
                    value=value,
                )
            else:
                error_data = response.json() if response.content else {}
                error_msg = error_data.get("error", f"HTTP {response.status_code}")
                logger.error(
                    f"[UsageReporter] FAILED: HTTP {response.status_code} - {error_msg} "
                    f"(user={user_id}, event={event_type}, value={value})"
                )
                return UsageReportResult(
                    success=False,
                    error=error_msg,
                )

        except httpx.TimeoutException as e:
            logger.error(
                f"[UsageReporter] TIMEOUT: Request to {self.frontend_url} timed out "
                f"after {self.timeout}s (user={user_id}, event={event_type})"
            )
            return UsageReportResult(success=False, error=f"Request timeout: {e}")
        except httpx.ConnectError as e:
            logger.error(
                f"[UsageReporter] CONNECTION ERROR: Could not connect to {self.frontend_url} "
                f"(user={user_id}, event={event_type}). Error: {e}"
            )
            return UsageReportResult(success=False, error=f"Connection error: {e}")
        except Exception as e:
            logger.error(
                f"[UsageReporter] UNEXPECTED ERROR: {type(e).__name__}: {e} "
                f"(user={user_id}, event={event_type})"
            )
            return UsageReportResult(success=False, error=str(e))

    async def report_meeting_minutes(
        self,
        user_id: str,
        minutes: int,
        room_id: Optional[str] = None,
        meeting_id: Optional[str] = None,
        session_id: Optional[str] = None,
        source: str = "agent",
    ) -> UsageReportResult:
        """
        Report meeting minutes usage with automatic retry on failure.

        Args:
            user_id: The user's ID
            minutes: Number of minutes to report
            room_id: Optional LiveKit room ID
            meeting_id: Optional meeting ID
            session_id: Optional session ID
            source: Source identifier for deduplication (default: "agent")

        Returns:
            UsageReportResult
        """
        if minutes <= 0:
            logger.debug(f"[UsageReporter] Skipping report for {minutes} minutes (<=0)")
            return UsageReportResult(success=True, value=0)

        metadata = {"source": source, "timestamp": asyncio.get_event_loop().time()}
        if room_id:
            metadata["roomId"] = room_id
        if meeting_id:
            metadata["meetingId"] = meeting_id
        if session_id:
            metadata["sessionId"] = session_id

        # Use retry wrapper for reliability
        return await self._report_usage_with_retry(
            user_id=user_id,
            event_type=MEETING_MINUTES,
            value=minutes,
            metadata=metadata,
        )

    async def _report_usage_with_retry(
        self,
        user_id: str,
        event_type: str,
        value: int,
        metadata: Optional[dict] = None,
    ) -> UsageReportResult:
        """
        Report usage with exponential backoff retry on failure.

        Implements reliable usage reporting to prevent data loss due to
        transient network failures or service unavailability.

        Args:
            user_id: The user's ID
            event_type: Type of usage event
            value: The value to report
            metadata: Optional additional metadata

        Returns:
            UsageReportResult with success status
        """
        last_error = None
        delay = INITIAL_RETRY_DELAY_SECONDS

        for attempt in range(MAX_RETRY_ATTEMPTS):
            result = await self._report_usage(
                user_id=user_id,
                event_type=event_type,
                value=value,
                metadata=metadata,
            )

            if result.success:
                if attempt > 0:
                    logger.info(
                        f"[UsageReporter] Retry succeeded on attempt {attempt + 1} "
                        f"for {event_type}:{value} user {user_id}"
                    )
                return result

            last_error = result.error
            logger.warning(
                f"[UsageReporter] Attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS} failed "
                f"for {event_type}:{value} user {user_id}: {result.error}"
            )

            # Don't sleep after the last attempt
            if attempt < MAX_RETRY_ATTEMPTS - 1:
                logger.debug(f"[UsageReporter] Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
                # Exponential backoff with cap
                delay = min(delay * 2, MAX_RETRY_DELAY_SECONDS)

        logger.error(
            f"[UsageReporter] All {MAX_RETRY_ATTEMPTS} retry attempts failed "
            f"for {event_type}:{value} user {user_id}. Last error: {last_error}"
        )

        return UsageReportResult(
            success=False,
            error=f"Failed after {MAX_RETRY_ATTEMPTS} attempts: {last_error}",
        )

    async def report_email_draft(
        self,
        user_id: str,
        count: int = 1,
        meeting_id: Optional[str] = None,
        action_type: Optional[str] = None,
        source: str = "agent",
    ) -> UsageReportResult:
        """
        Report email draft generation with automatic retry on failure.

        Args:
            user_id: The user's ID
            count: Number of drafts generated (default 1)
            meeting_id: Optional meeting ID
            action_type: Optional action type that triggered the draft
            source: Source identifier for deduplication (default: "agent")

        Returns:
            UsageReportResult
        """
        if count <= 0:
            logger.debug(f"[UsageReporter] Skipping email draft report for count={count}")
            return UsageReportResult(success=True, value=0)

        metadata = {"source": source, "timestamp": asyncio.get_event_loop().time()}
        if meeting_id:
            metadata["meetingId"] = meeting_id
        if action_type:
            metadata["actionType"] = action_type

        # Use retry wrapper for reliability
        return await self._report_usage_with_retry(
            user_id=user_id,
            event_type=EMAIL_DRAFTS,
            value=count,
            metadata=metadata,
        )

    async def check_meeting_limits(self, user_id: str) -> tuple[bool, MeetingLimitStatus]:
        """
        Check if a user can start/join a meeting based on their limits.

        FAIL CLOSED: If we cannot verify usage, deny access to prevent billing bypass.
        This is consistent with email draft checks and frontend behavior.

        Failure modes:
        - Production: Fail closed on any error
        - Development with BYPASS_USAGE_CHECKS=true: Allow with free tier limits
        - Development without bypass: Fail closed (same as production)

        Args:
            user_id: The user's ID

        Returns:
            Tuple of (allowed, status)
        """
        if not self.service_token:
            # In development with bypass enabled, allow with free tier limits
            if IS_DEVELOPMENT and BYPASS_USAGE_CHECKS:
                logger.warning(
                    "[UsageReporter] INTERNAL_SERVICE_TOKEN not configured. "
                    "Allowing meeting with free tier limits (dev bypass)."
                )
                return True, MeetingLimitStatus(
                    allowed=True,
                    tier="free",
                    minutes_used=0,
                    minutes_limit=FREE_TIER_MINUTES_LIMIT,  # L1: Use named constant
                    remaining_minutes=FREE_TIER_MINUTES_LIMIT,
                    reason="Development mode - usage checks bypassed",
                )

            # FAIL CLOSED in production or without explicit bypass
            logger.error(
                "[UsageReporter] INTERNAL_SERVICE_TOKEN not configured. "
                "Denying meeting access. Set BYPASS_USAGE_CHECKS=true in dev to bypass."
            )
            return False, MeetingLimitStatus(
                allowed=False,
                tier="free",
                minutes_used=0,
                minutes_limit=0,
                remaining_minutes=0,
                reason="Service configuration error - unable to verify usage limits",
            )

        try:
            client = await self._get_client()

            response = await client.get(
                f"{self.frontend_url}/api/internal/usage",
                params={"userId": user_id, "checkType": "meeting"},
                headers=self._get_headers(),
            )

            if response.status_code == 200:
                data = response.json()
                meeting = data.get("meeting", {})

                status = MeetingLimitStatus(
                    allowed=meeting.get("allowed", False),  # Default to False for safety
                    tier=meeting.get("tier", "free"),
                    minutes_used=meeting.get("minutesUsed", 0),
                    minutes_limit=meeting.get("minutesLimit", 0),
                    remaining_minutes=meeting.get("remainingMinutes", 0),
                    reason=meeting.get("reason"),
                )

                return status.allowed, status
            else:
                logger.error(
                    f"[UsageReporter] Failed to check limits: HTTP {response.status_code}"
                )
                # FAIL CLOSED: Deny meeting if we can't verify limits
                return False, MeetingLimitStatus(
                    allowed=False,
                    tier="free",
                    minutes_used=0,
                    minutes_limit=0,
                    remaining_minutes=0,
                    reason="Unable to verify usage limits - service temporarily unavailable",
                )

        except httpx.TimeoutException:
            logger.error("[UsageReporter] Timeout checking meeting limits")
            # FAIL CLOSED on timeout
            return False, MeetingLimitStatus(
                allowed=False,
                tier="free",
                minutes_used=0,
                minutes_limit=0,
                remaining_minutes=0,
                reason="Request timeout - please try again",
            )
        except Exception as e:
            logger.error(f"[UsageReporter] Error checking limits: {e}")
            # FAIL CLOSED on any error
            return False, MeetingLimitStatus(
                allowed=False,
                tier="free",
                minutes_used=0,
                minutes_limit=0,
                remaining_minutes=0,
                reason="Service temporarily unavailable",
            )

    async def check_email_draft_limits(self, user_id: str) -> tuple[bool, dict]:
        """
        Check if a user can create email drafts.

        FAIL CLOSED: If we cannot verify usage, deny access to prevent billing bypass.
        This is consistent with meeting limit checks and frontend behavior.

        Failure modes:
        - Production: Fail closed on any error
        - Development with BYPASS_USAGE_CHECKS=true: Allow with free tier limits
        - Development without bypass: Fail closed (same as production)

        Args:
            user_id: The user's ID

        Returns:
            Tuple of (allowed, status_dict)
        """
        if not self.service_token:
            # In development with bypass enabled, allow with synthetic limits for testing.
            # NOTE: This returns a synthetic limit of 10 drafts for dev testing purposes.
            # In production, free tier has NO email drafts (0 limit). This synthetic limit
            # is intentionally different to facilitate local development and testing
            # without requiring a Pro/Business subscription.
            if IS_DEVELOPMENT and BYPASS_USAGE_CHECKS:
                logger.warning(
                    "[UsageReporter] INTERNAL_SERVICE_TOKEN not configured. "
                    f"Allowing email drafts with SYNTHETIC limit of {DEV_EMAIL_DRAFTS_LIMIT} (dev bypass). "
                    "Note: Free tier has 0 email drafts in production."
                )
                return True, {
                    "allowed": True,
                    "draftsUsed": 0,
                    "draftsLimit": DEV_EMAIL_DRAFTS_LIMIT,  # L1: Use named constant
                    "remainingDrafts": DEV_EMAIL_DRAFTS_LIMIT,
                    "reason": "Development mode - synthetic limit for testing (not real allowance)",
                }

            # FAIL CLOSED in production or without explicit bypass
            logger.error(
                "[UsageReporter] INTERNAL_SERVICE_TOKEN not configured. "
                "Denying email draft access. Set BYPASS_USAGE_CHECKS=true in dev to bypass."
            )
            return False, {"reason": "Service configuration error - unable to verify usage limits"}

        try:
            client = await self._get_client()

            response = await client.get(
                f"{self.frontend_url}/api/internal/usage",
                params={"userId": user_id, "checkType": "email-draft"},
                headers=self._get_headers(),
            )

            if response.status_code == 200:
                data = response.json()
                email_draft = data.get("emailDraft", {})
                # Default to False for safety if 'allowed' is not present
                return email_draft.get("allowed", False), email_draft
            else:
                logger.error(
                    f"[UsageReporter] Failed to check email draft limits: HTTP {response.status_code}"
                )
                # FAIL CLOSED: Deny if we can't verify limits
                return False, {"reason": f"Unable to verify limits - HTTP {response.status_code}"}

        except httpx.TimeoutException:
            logger.error("[UsageReporter] Timeout checking email draft limits")
            # FAIL CLOSED on timeout
            return False, {"reason": "Request timeout - please try again"}
        except Exception as e:
            logger.error(f"[UsageReporter] Error checking email limits: {e}")
            # FAIL CLOSED on any error
            return False, {"reason": "Service temporarily unavailable"}


# ============================================================================
# Singleton Instance
# ============================================================================

_usage_reporter: Optional[UsageReporter] = None


def get_usage_reporter() -> UsageReporter:
    """Get the singleton UsageReporter instance."""
    global _usage_reporter
    if _usage_reporter is None:
        _usage_reporter = UsageReporter()
    return _usage_reporter


async def close_usage_reporter() -> None:
    """Close the singleton UsageReporter."""
    global _usage_reporter
    if _usage_reporter:
        await _usage_reporter.close()
        _usage_reporter = None
