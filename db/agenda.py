"""
Agenda Database Client for Luframe Agent - Phase 4 Implementation

Provides direct PostgreSQL access for agenda operations.
The agent reads agenda from the database (created by frontend)
and writes status updates as topics progress.

Uses asyncpg for async database operations that don't block
the transcription pipeline.

Environment Variables:
    DATABASE_URL: PostgreSQL connection string (same as frontend)
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

logger = logging.getLogger("luframe-agenda-db")


def _utc_now_naive() -> datetime:
    """
    Get current UTC time as a naive datetime (no timezone info).

    PostgreSQL 'timestamp' columns (without time zone) expect naive datetimes.
    asyncpg will raise an error if you try to insert a timezone-aware datetime
    into a 'timestamp without time zone' column.

    This function returns datetime.utcnow() which is naive but represents UTC.
    """
    return datetime.utcnow()


class AgendaDB:
    """
    Async database client for agenda operations.

    Connects to the same PostgreSQL database as the frontend
    using the DATABASE_URL environment variable.

    Key Operations:
    - get_agenda_for_room: Load agenda with items for a room
    - update_item_status: Update item status (pending/in_progress/completed/skipped)
    - update_current_item_index: Update the current active item
    - start_meeting: Mark meeting as started
    - end_meeting: Mark meeting as ended
    """

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database client.

        Args:
            database_url: PostgreSQL connection string (optional, uses DATABASE_URL env var)
        """
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError(
                "DATABASE_URL environment variable is required for agenda tracking"
            )
        self._pool = None
        self._connect_lock = asyncio.Lock()

    async def connect(self):
        """
        Create connection pool.

        Uses lazy initialization - only connects when first needed.
        Thread-safe via asyncio lock.
        """
        if self._pool is not None:
            return

        async with self._connect_lock:
            # Double-check after acquiring lock
            if self._pool is not None:
                return

            try:
                import asyncpg
            except ImportError:
                raise ImportError(
                    "asyncpg is required for agenda database access. "
                    "Install it with: pip install asyncpg"
                )

            # Disable prepared statement caching for pgbouncer compatibility
            # Supabase uses pgbouncer with transaction pooling, which doesn't
            # support prepared statements. Setting statement_cache_size=0
            # forces asyncpg to use simple query protocol instead.
            self._pool = await asyncpg.create_pool(
                self.database_url,
                min_size=1,
                max_size=5,
                command_timeout=10.0,
                statement_cache_size=0,  # Required for Supabase/pgbouncer
            )
            logger.info("Connected to PostgreSQL database for agenda tracking")

    async def close(self):
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Closed PostgreSQL connection pool")

    async def _ensure_connected(self):
        """Ensure database connection is established."""
        if self._pool is None:
            await self.connect()

    # =========================================================================
    # Read Operations
    # =========================================================================

    async def get_agenda_for_room(self, room_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch agenda with items for a room.

        Returns the agenda in a format matching the frontend types:
        {
            "id": "agenda-xxx",
            "roomId": "room-xxx",
            "createdBy": "user-xxx",
            "itemCount": 5,
            "status": "active",
            "currentItemIndex": 1,
            "version": 2,
            "meetingStartedAt": "2024-01-01T00:00:00Z",
            "meetingEndedAt": null,
            "items": [...]
        }

        Args:
            room_id: LiveKit room ID

        Returns:
            Agenda dict with items, or None if no agenda exists
        """
        await self._ensure_connected()

        async with self._pool.acquire() as conn:
            # Get agenda
            agenda_row = await conn.fetchrow(
                """
                SELECT id, room_id, created_by, item_count, status,
                       current_item_index, version, meeting_started_at,
                       meeting_ended_at, created_at, updated_at
                FROM agenda
                WHERE room_id = $1
                """,
                room_id
            )

            if not agenda_row:
                logger.debug(f"No agenda found for room {room_id}")
                return None

            # Get items
            item_rows = await conn.fetch(
                """
                SELECT id, agenda_id, order_index, title, description,
                       estimated_duration, presenter, status,
                       started_at, completed_at, actual_duration,
                       start_transcript_ref, end_transcript_ref,
                       created_at, updated_at
                FROM agenda_item
                WHERE agenda_id = $1
                ORDER BY order_index ASC
                """,
                agenda_row["id"]
            )

            # Convert to dict format matching frontend types
            agenda = {
                "id": agenda_row["id"],
                "roomId": agenda_row["room_id"],
                "createdBy": agenda_row["created_by"],
                "itemCount": agenda_row["item_count"],
                "status": agenda_row["status"],
                "currentItemIndex": agenda_row["current_item_index"],
                "version": agenda_row["version"],
                "meetingStartedAt": (
                    agenda_row["meeting_started_at"].isoformat()
                    if agenda_row["meeting_started_at"] else None
                ),
                "meetingEndedAt": (
                    agenda_row["meeting_ended_at"].isoformat()
                    if agenda_row["meeting_ended_at"] else None
                ),
                "items": [
                    {
                        "id": row["id"],
                        "agendaId": row["agenda_id"],
                        "orderIndex": row["order_index"],
                        "title": row["title"],
                        "description": row["description"],
                        "estimatedDuration": row["estimated_duration"],
                        "presenter": row["presenter"],
                        "status": row["status"],
                        "startedAt": (
                            row["started_at"].isoformat()
                            if row["started_at"] else None
                        ),
                        "completedAt": (
                            row["completed_at"].isoformat()
                            if row["completed_at"] else None
                        ),
                        "actualDuration": row["actual_duration"],
                        "startTranscriptRef": row["start_transcript_ref"],
                        "endTranscriptRef": row["end_transcript_ref"],
                    }
                    for row in item_rows
                ],
            }

            logger.info(
                f"Loaded agenda for room {room_id}: "
                f"{agenda['itemCount']} items, status={agenda['status']}"
            )

            return agenda

    async def get_agenda_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single agenda item by ID.

        Args:
            item_id: Agenda item ID

        Returns:
            Item dict or None if not found
        """
        await self._ensure_connected()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, agenda_id, order_index, title, description,
                       estimated_duration, presenter, status,
                       started_at, completed_at, actual_duration,
                       start_transcript_ref, end_transcript_ref
                FROM agenda_item
                WHERE id = $1
                """,
                item_id
            )

            if not row:
                return None

            return {
                "id": row["id"],
                "agendaId": row["agenda_id"],
                "orderIndex": row["order_index"],
                "title": row["title"],
                "description": row["description"],
                "estimatedDuration": row["estimated_duration"],
                "presenter": row["presenter"],
                "status": row["status"],
                "startedAt": row["started_at"].isoformat() if row["started_at"] else None,
                "completedAt": row["completed_at"].isoformat() if row["completed_at"] else None,
                "actualDuration": row["actual_duration"],
                "startTranscriptRef": row["start_transcript_ref"],
                "endTranscriptRef": row["end_transcript_ref"],
            }

    # =========================================================================
    # Write Operations
    # =========================================================================

    async def update_item_status(
        self,
        item_id: str,
        status: str,
        transcript_ref: Optional[str] = None,
        started_at: Optional[str] = None
    ) -> bool:
        """
        Update an agenda item's status and timestamps.

        FIX (R1+R2): Added started_at parameter to avoid extra query.
        Duration is now calculated in SQL or passed from caller.

        Args:
            item_id: Agenda item ID
            status: New status (pending, in_progress, completed, skipped)
            transcript_ref: Optional transcript segment reference
            started_at: Optional ISO timestamp when item started (for duration calc)

        Returns:
            True if update succeeded, False otherwise
        """
        await self._ensure_connected()

        # Use naive UTC datetime for PostgreSQL 'timestamp without time zone' columns
        now = _utc_now_naive()

        async with self._pool.acquire() as conn:
            # Build update query based on status
            if status == "in_progress":
                result = await conn.execute(
                    """
                    UPDATE agenda_item
                    SET status = $1,
                        started_at = $2,
                        start_transcript_ref = COALESCE($3, start_transcript_ref),
                        updated_at = $2
                    WHERE id = $4
                    """,
                    status, now, transcript_ref, item_id
                )
            elif status in ("completed", "skipped"):
                # Calculate duration from passed started_at or in SQL
                # This eliminates the extra get_agenda_item() query
                if started_at:
                    # Duration passed from caller (preferred)
                    try:
                        start_time = datetime.fromisoformat(
                            started_at.replace("Z", "+00:00")
                        )
                        # Convert start_time to naive if it's timezone-aware
                        if start_time.tzinfo is not None:
                            start_time = start_time.replace(tzinfo=None)
                        # Both now and start_time are now naive (UTC)
                        actual_duration = int((now - start_time).total_seconds())
                    except Exception as e:
                        logger.warning(f"Failed to calculate duration from passed started_at: {e}")
                        actual_duration = None
                else:
                    actual_duration = None

                # If duration couldn't be calculated from param, let SQL handle it
                result = await conn.execute(
                    """
                    UPDATE agenda_item
                    SET status = $1,
                        completed_at = $2,
                        actual_duration = COALESCE(
                            $3,
                            CASE WHEN started_at IS NOT NULL
                                 THEN EXTRACT(EPOCH FROM ($2 - started_at))::integer
                                 ELSE actual_duration
                            END
                        ),
                        end_transcript_ref = COALESCE($4, end_transcript_ref),
                        updated_at = $2
                    WHERE id = $5
                    """,
                    status, now, actual_duration, transcript_ref, item_id
                )
            else:
                result = await conn.execute(
                    """
                    UPDATE agenda_item
                    SET status = $1, updated_at = $2
                    WHERE id = $3
                    """,
                    status, now, item_id
                )

            success = result == "UPDATE 1"
            if success:
                logger.debug(f"Updated item {item_id} to status={status}")
            else:
                logger.warning(f"Failed to update item {item_id}")

            return success

    async def update_current_item_index(
        self,
        agenda_id: str,
        current_item_index: Optional[int]
    ) -> bool:
        """
        Update the current item index on an agenda.

        Args:
            agenda_id: Agenda ID
            current_item_index: New current item index (null if none)

        Returns:
            True if update succeeded
        """
        await self._ensure_connected()

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE agenda
                SET current_item_index = $1, updated_at = $2
                WHERE id = $3
                """,
                current_item_index, _utc_now_naive(), agenda_id
            )

            success = result == "UPDATE 1"
            if success:
                logger.debug(f"Updated current_item_index to {current_item_index}")
            return success

    async def start_meeting(self, agenda_id: str) -> bool:
        """
        Mark the meeting as started.

        Sets meeting_started_at timestamp on the agenda.

        Args:
            agenda_id: Agenda ID

        Returns:
            True if update succeeded
        """
        await self._ensure_connected()

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE agenda
                SET meeting_started_at = $1, updated_at = $1
                WHERE id = $2
                """,
                _utc_now_naive(), agenda_id
            )

            success = result == "UPDATE 1"
            if success:
                logger.info(f"Marked meeting started for agenda {agenda_id}")
            return success

    async def end_meeting(self, agenda_id: str) -> bool:
        """
        Mark the meeting as ended.

        Sets meeting_ended_at timestamp and status to 'completed'.

        Args:
            agenda_id: Agenda ID

        Returns:
            True if update succeeded
        """
        await self._ensure_connected()

        # Use naive UTC datetime for PostgreSQL 'timestamp without time zone' columns
        now = _utc_now_naive()

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE agenda
                SET status = 'completed',
                    meeting_ended_at = $1,
                    updated_at = $1
                WHERE id = $2
                """,
                now, agenda_id
            )

            success = result == "UPDATE 1"
            if success:
                logger.info(f"Marked meeting ended for agenda {agenda_id}")
            return success

    # =========================================================================
    # Utility Operations
    # =========================================================================

    async def agenda_exists(self, room_id: str) -> bool:
        """
        Check if an agenda exists for a room.

        Args:
            room_id: LiveKit room ID

        Returns:
            True if agenda exists
        """
        await self._ensure_connected()

        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM agenda WHERE room_id = $1)",
                room_id
            )
            return result

    async def is_agenda_active(self, room_id: str) -> bool:
        """
        Check if a room has an active agenda (status = 'active').

        Args:
            room_id: LiveKit room ID

        Returns:
            True if agenda exists and is active
        """
        await self._ensure_connected()

        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1 FROM agenda
                    WHERE room_id = $1 AND status = 'active'
                )
                """,
                room_id
            )
            return result

    async def get_agenda_version(self, room_id: str) -> Optional[int]:
        """
        Get the current version of an agenda.

        Used for cache invalidation and conflict detection.

        Args:
            room_id: LiveKit room ID

        Returns:
            Version number or None if no agenda
        """
        await self._ensure_connected()

        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT version FROM agenda WHERE room_id = $1",
                room_id
            )
