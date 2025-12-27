"""
Identity parsing utilities for LiveKit participants.

Provides functions to extract user IDs from LiveKit participant identities
for billing attribution.

Identity Format:
    {userId}-{8-char-hex-suffix}

    Examples:
    - "550e8400-e29b-41d4-a716-446655440000-a1b2c3d4" -> "550e8400-e29b-41d4-a716-446655440000"
    - "simple-user-deadbeef" -> "simple-user"
    - "user123-abcd1234" -> "user123"

The suffix is always exactly 8 lowercase hex characters preceded by a hyphen.
This module correctly handles user IDs that contain hyphens (like UUIDs).
"""

from typing import Optional
import re
from livekit import rtc

# LiveKit identity format: {userId}-{8-char-hex-suffix}
# The suffix is always exactly 8 lowercase hex characters
IDENTITY_SUFFIX_PATTERN = re.compile(r'-[0-9a-f]{8}$')

# Agent identity prefix
AGENT_IDENTITY_PREFIX = "luframe"


def extract_user_id_from_identity(identity: Optional[str]) -> Optional[str]:
    """
    Extract user ID from LiveKit participant identity.

    Identity format: {userId}-{8-char-hex-suffix}
    The suffix is always exactly 8 lowercase hex characters preceded by a hyphen.

    Args:
        identity: The participant's LiveKit identity

    Returns:
        The userId portion, or None if extraction fails

    Examples:
        >>> extract_user_id_from_identity("550e8400-e29b-41d4-a716-446655440000-a1b2c3d4")
        '550e8400-e29b-41d4-a716-446655440000'
        >>> extract_user_id_from_identity("simple-user-deadbeef")
        'simple-user'
        >>> extract_user_id_from_identity("invalid")
        None
        >>> extract_user_id_from_identity(None)
        None
    """
    if not identity:
        return None

    # Check if identity ends with -{8 hex chars}
    match = IDENTITY_SUFFIX_PATTERN.search(identity)
    if match:
        # Return everything before the suffix
        return identity[:match.start()]

    # Fallback: return as-is if format doesn't match
    # This handles edge cases where identity might not have suffix
    return identity


def is_agent_identity(identity: Optional[str]) -> bool:
    """
    Check if the identity belongs to an agent (not a human).

    Args:
        identity: The participant's LiveKit identity

    Returns:
        True if this is an agent identity, False otherwise
    """
    return bool(identity and identity.startswith(AGENT_IDENTITY_PREFIX))


def get_meeting_owner_from_room(
    room: rtc.Room,
    cached_owner: Optional[str] = None
) -> Optional[str]:
    """
    Get the meeting owner's user ID for billing attribution.

    Identifies the meeting owner by finding the first human (non-agent)
    participant in the room.

    Args:
        room: The LiveKit room instance
        cached_owner: Previously identified owner (returned if set)

    Returns:
        User ID of the meeting owner, or None if not determinable
    """
    if cached_owner:
        return cached_owner

    for participant in room.remote_participants.values():
        if not is_agent_identity(participant.identity):
            user_id = extract_user_id_from_identity(participant.identity)
            if user_id:
                return user_id

    return None
