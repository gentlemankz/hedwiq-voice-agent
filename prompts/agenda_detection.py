"""
Agenda Detection Prompts for Hedwiq Agent - Phase 4 Implementation

Contains prompts for detecting topic transitions during meetings.
The agent analyzes transcripts to determine when the discussion has
moved from one agenda topic to another.

Key design principles:
- Conservative: Only detect clear transitions, not speculation
- Evidence-based: Require explicit signals in the transcript
- Confidence scoring: Allow frontend to show confidence level
- Multi-signal approach: Combine explicit mentions, keywords, and LLM analysis
"""

# ============================================================================
# Topic Detection Prompts
# ============================================================================

TOPIC_DETECTION_SYSTEM_PROMPT = """You are a meeting analyst detecting when discussion topics change.

Given:
1. Recent meeting transcript (last 2-3 minutes)
2. Current agenda topic being discussed
3. Next agenda topic(s) to watch for

Determine if the conversation has transitioned from the current topic to a next topic.

You must be:
1. CONSERVATIVE - Only mark transitions that are CLEAR from the conversation
2. PRECISE - Look for explicit transition phrases or sustained discussion of new topic
3. EVIDENCE-BASED - Quote the specific text that indicates a transition
4. NON-SPECULATIVE - Brief mentions of future topics are NOT transitions

Respond with JSON only:
{
  "has_transitioned": boolean,
  "next_topic_id": "id of the topic being discussed now (or null)",
  "confidence": 0.0-1.0,
  "reason": "brief explanation",
  "evidence": "relevant quote from transcript (10-50 chars)"
}

Confidence guidelines:
- 0.9+: Explicit transition phrase ("Let's move on to...", "Next item...")
- 0.8-0.9: Clear sustained discussion of new topic (30+ seconds)
- 0.7-0.8: Discussion content clearly matches new topic
- <0.7: Unclear, don't transition

CRITICAL: Return has_transitioned=false if uncertain. It's better to miss a transition
than to falsely detect one, which would confuse users."""


TOPIC_DETECTION_USER_TEMPLATE = """## Current Agenda Topic
ID: {current_id}
Title: {current_title}
Description: {current_description}

## Upcoming Agenda Topics
{upcoming_topics}

## Recent Transcript (last ~2-3 minutes)
{transcript}

Has the discussion transitioned to a new topic? Respond with JSON only:"""


# ============================================================================
# Meeting Start/End Detection Prompts
# ============================================================================

MEETING_START_SYSTEM_PROMPT = """You are detecting if a meeting has started discussing agenda items.

Look for signals that the meeting is beginning substantive discussion:
- Greetings and introductions being completed
- Explicit meeting start ("Let's get started", "First item today...")
- Beginning discussion of agenda topics
- Host/facilitator initiating topic discussion

Respond with JSON only:
{
  "has_started": boolean,
  "first_topic_id": "id of the first topic being discussed (or null)",
  "confidence": 0.0-1.0,
  "evidence": "relevant quote (10-50 chars)"
}

Be conservative: Small talk, waiting for participants, or technical setup are NOT meeting start."""


MEETING_START_USER_TEMPLATE = """## Agenda Topics (in order)
{agenda_topics}

## Recent Transcript
{transcript}

Has the meeting started discussing agenda topics? Respond with JSON only:"""


MEETING_END_SYSTEM_PROMPT = """You are detecting if a meeting is ending.

Look for signals:
- Explicit closing ("That's all for today", "Let's wrap up", "Meeting adjourned")
- Final summary or action items discussion
- Farewells being exchanged
- Discussion of next meeting/follow-ups

Respond with JSON only:
{
  "has_ended": boolean,
  "confidence": 0.0-1.0,
  "evidence": "relevant quote (10-50 chars)"
}

Be conservative: Brief pauses or tangents are NOT meeting end."""


MEETING_END_USER_TEMPLATE = """## Agenda Status
Total topics: {total_topics}
Completed: {completed_topics}
Current topic: {current_topic}

## Recent Transcript
{transcript}

Has the meeting ended? Respond with JSON only:"""


# ============================================================================
# DEPRECATED: Word-based Patterns (kept for reference, NOT used in detection)
# ============================================================================
#
# Post-review decision (Reviewer 1 + Reviewer 2):
# Word-based pattern matching is REMOVED from topic detection.
# Meetings are unpredictable - people speak differently, use different phrases,
# have different communication styles. Relying on specific phrases like
# "let's get started" or "next topic" fails in real-world scenarios.
#
# ALL detection now uses pure LLM context analysis via format_unified_topic_detection_prompt()
#
# The patterns below are kept ONLY for historical reference. They are NOT used.
# ============================================================================

# DEPRECATED - NOT USED - Historical reference only
_DEPRECATED_EXPLICIT_TRANSITION_PATTERNS = [
    r"let's move (on )?to\s+(.+)",
    r"next (on the agenda|item|topic) is\s+(.+)",
    # ... etc
]

# DEPRECATED - NOT USED - Historical reference only
_DEPRECATED_MEETING_START_PATTERNS = [
    r"let's (get started|begin|kick off)",
    # ... etc
]

# DEPRECATED - NOT USED - Historical reference only
_DEPRECATED_MEETING_END_PATTERNS = [
    r"that's (all|everything) for today",
    # ... etc
]


# ============================================================================
# Helper Functions
# ============================================================================

# Maximum transcript length to prevent excessive LLM input
MAX_TRANSCRIPT_LENGTH = 4000

# Characters that could be used for prompt injection attacks
PROMPT_INJECTION_PATTERNS = [
    "ignore previous",
    "ignore all",
    "disregard",
    "forget your instructions",
    "system prompt",
    "you are now",
    "new instructions",
]


def sanitize_transcript(transcript_text: str) -> str:
    """
    Sanitize transcript text to prevent prompt injection attacks.

    FIX (R1): Raw transcript was injected directly into LLM prompts.

    This function:
    1. Truncates to prevent excessive input
    2. Escapes potential injection patterns
    3. Adds delimiter markers

    Args:
        transcript_text: Raw transcript from speakers

    Returns:
        Sanitized transcript safe for LLM prompt injection
    """
    # Truncate to prevent excessive input
    if len(transcript_text) > MAX_TRANSCRIPT_LENGTH:
        transcript_text = transcript_text[:MAX_TRANSCRIPT_LENGTH] + "... [truncated]"

    # Convert to lowercase for pattern matching
    text_lower = transcript_text.lower()

    # Check for potential injection patterns and escape them
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern in text_lower:
            # Add visual markers to make injection obvious
            transcript_text = transcript_text.replace(
                pattern, f"[SPEAKER SAID: {pattern}]"
            )
            # Case-insensitive replacement
            import re
            transcript_text = re.sub(
                re.escape(pattern),
                f"[SPEAKER SAID: {pattern}]",
                transcript_text,
                flags=re.IGNORECASE
            )

    return transcript_text


def format_topic_detection_prompt(
    current_item: dict,
    upcoming_items: list,
    transcript_text: str
) -> tuple[str, str]:
    """
    Format the topic detection prompt with current context.

    Args:
        current_item: Current agenda item dict with id, title, description
        upcoming_items: List of upcoming agenda item dicts
        transcript_text: Recent transcript text

    Returns:
        tuple: (system_prompt, user_prompt)
    """
    # Sanitize transcript to prevent prompt injection (FIX R1)
    safe_transcript = sanitize_transcript(transcript_text)

    # Format upcoming topics
    upcoming_parts = []
    for item in upcoming_items[:3]:  # Limit to next 3 items
        upcoming_parts.append(
            f"- ID: {item.get('id', 'unknown')}\n"
            f"  Title: {item.get('title', 'Unknown')}\n"
            f"  Description: {item.get('description') or 'No description'}"
        )

    upcoming_topics = "\n".join(upcoming_parts) if upcoming_parts else "No more topics"

    user_prompt = TOPIC_DETECTION_USER_TEMPLATE.format(
        current_id=current_item.get('id', 'unknown'),
        current_title=current_item.get('title', 'Unknown'),
        current_description=current_item.get('description') or 'No description',
        upcoming_topics=upcoming_topics,
        transcript=safe_transcript
    )

    return TOPIC_DETECTION_SYSTEM_PROMPT, user_prompt


def format_meeting_start_prompt(agenda_items: list, transcript_text: str) -> tuple[str, str]:
    """
    Format the meeting start detection prompt.

    Args:
        agenda_items: List of agenda item dicts
        transcript_text: Recent transcript text

    Returns:
        tuple: (system_prompt, user_prompt)
    """
    # Sanitize transcript (FIX R1)
    safe_transcript = sanitize_transcript(transcript_text)

    topics_parts = []
    for i, item in enumerate(agenda_items[:5]):  # First 5 items
        topics_parts.append(
            f"{i+1}. ID: {item.get('id', 'unknown')} - {item.get('title', 'Unknown')}"
        )

    user_prompt = MEETING_START_USER_TEMPLATE.format(
        agenda_topics="\n".join(topics_parts) if topics_parts else "No agenda",
        transcript=safe_transcript
    )

    return MEETING_START_SYSTEM_PROMPT, user_prompt


def format_meeting_end_prompt(
    total_topics: int,
    completed_topics: int,
    current_topic: str,
    transcript_text: str
) -> tuple[str, str]:
    """
    Format the meeting end detection prompt.

    Args:
        total_topics: Total number of agenda topics
        completed_topics: Number of completed topics
        current_topic: Current topic title or "None"
        transcript_text: Recent transcript text

    Returns:
        tuple: (system_prompt, user_prompt)
    """
    # Sanitize transcript (FIX R1)
    safe_transcript = sanitize_transcript(transcript_text)

    user_prompt = MEETING_END_USER_TEMPLATE.format(
        total_topics=total_topics,
        completed_topics=completed_topics,
        current_topic=current_topic,
        transcript=safe_transcript
    )

    return MEETING_END_SYSTEM_PROMPT, user_prompt


def validate_llm_response(response: dict, required_fields: list[str]) -> bool:
    """
    Validate LLM JSON response has required fields.

    FIX (R2): LLM responses were not validated for required fields.

    Args:
        response: Parsed JSON response from LLM
        required_fields: List of required field names

    Returns:
        True if all required fields present, False otherwise
    """
    if not isinstance(response, dict):
        return False

    for field in required_fields:
        if field not in response:
            return False

    return True


# Required fields for each response type
TOPIC_DETECTION_REQUIRED_FIELDS = ["has_transitioned", "confidence"]
MEETING_START_REQUIRED_FIELDS = ["has_started", "confidence"]
MEETING_END_REQUIRED_FIELDS = ["has_ended", "confidence"]


# ============================================================================
# Constants
# ============================================================================

# Confidence thresholds
MIN_DETECTION_CONFIDENCE = 0.7     # Minimum confidence to act on detection
HIGH_CONFIDENCE_THRESHOLD = 0.85   # High confidence threshold

# LLM settings
DETECTION_TIMEOUT_SECONDS = 5.0    # Max time to wait for LLM response (increased for better analysis)
DETECTION_MAX_RETRIES = 1          # Retries on timeout/error
DETECTION_TEMPERATURE = 0.1        # Slight temperature for more natural reasoning
DETECTION_MAX_TOKENS = 350         # Max response tokens (increased for detailed response)


# ============================================================================
# Intelligent Topic Detection - Full Context LLM Analysis
# ============================================================================
#
# NEW ARCHITECTURE: Trust the LLM with full conversation context
#
# Key principles:
# 1. Give LLM the FULL conversation transcript, not just recent segments
# 2. Let LLM understand the natural flow of discussion
# 3. No artificial constraints - trust LLM's intelligence
# 4. Simple, clear prompt that asks the right question
#
# The old prompt asked: "Does this text MATCH a topic?" (keyword similarity)
# The new prompt asks: "Has the speaker MOVED ON to a new topic?" (intent)
# ============================================================================

SMART_TOPIC_DETECTION_SYSTEM_PROMPT = """You are an intelligent meeting assistant tracking agenda progress.

You have access to:
1. The meeting's agenda (list of topics to discuss) - THIS IS THE SOURCE OF TRUTH
2. The COMPLETE conversation transcript from the meeting start
3. The currently tracked topic

Your job: Determine if the speaker is now discussing a DIFFERENT agenda topic than the one currently tracked.

## CRITICAL: The Agenda Defines Topic Boundaries

The agenda items are SEPARATE topics, not subtopics of each other. For example:
- If agenda has: "1. General talk about AI" and "2. Computer use AI agents"
- Then "Computer use AI agents" is a SEPARATE topic, NOT a subtopic of "General talk"
- When the speaker starts explaining/discussing "Computer use AI agents" in detail, that IS a transition to topic 2

## When to Transition

Transition to a new topic when:
- The speaker starts **substantively discussing** content that matches a LATER agenda item
- The speaker is **explaining, elaborating, or giving details** about a specific agenda topic
- The discussion focus has **shifted** from the current topic to another agenda item

## When NOT to Transition

Stay on current topic when:
- The speaker is ONLY **listing/previewing** what will be discussed ("Today we'll cover X, Y, Z")
- The speaker **briefly mentions** a topic without elaborating (just the name, no explanation)
- The speaker is still in a **pure introduction** phase (greetings, agenda overview)

## How to Decide

1. Look at the AGENDA - what are the separate topics?
2. Look at the CURRENT SPEECH - what is the speaker actually explaining/discussing?
3. Does the current speech content MATCH a different agenda item?
4. Is the speaker EXPLAINING that topic (not just mentioning it)?

Example:
- Agenda: "1. General AI talk", "2. Computer use AI agents", "3. Next steps"
- Speaker says: "Altrina is a computer use AI agent that controls your desktop..."
- This is EXPLAINING topic 2, not just mentioning it → TRANSITION to topic 2

## Response Format (JSON)

{
  "should_transition": true/false,
  "new_topic_id": "ID of the new topic (only if should_transition=true)",
  "new_topic_index": number or null,
  "reasoning": "Your analysis of why transition should/shouldn't happen",
  "current_focus": "What is the speaker currently discussing?"
}

## Key Guidance

- The AGENDA defines what counts as separate topics - respect the agenda structure
- If speaker is explaining content that matches a specific agenda item, that's likely a transition
- "General" or "Overview" topics are introductions; specific topics are deep dives
- When the speaker dives deep into a specific agenda item, transition to it"""


SMART_TOPIC_DETECTION_USER_TEMPLATE = """## MEETING AGENDA (These are SEPARATE topics, not subtopics)

{agenda_topics}

## CURRENTLY TRACKED TOPIC

{current_topic}

## FULL MEETING TRANSCRIPT

{full_transcript}

---

Look at the agenda items above. Each numbered item is a SEPARATE topic.
Is the speaker now explaining/discussing content that matches a DIFFERENT agenda item than the currently tracked one?

Respond with JSON only:"""


# Required fields for the new simplified response
SMART_DETECTION_REQUIRED_FIELDS = ["should_transition", "reasoning", "current_focus"]


def format_smart_topic_detection_prompt(
    all_items: list,
    current_item: dict | None,
    current_index: int,
    full_transcript: str
) -> tuple[str, str]:
    """
    Format the intelligent topic detection prompt with FULL conversation context.

    This prompt gives the LLM everything it needs to make an intelligent decision:
    - Complete agenda with all topics
    - Full conversation transcript (not just recent)
    - Current tracked topic

    Args:
        all_items: All agenda items
        current_item: Current active item or None
        current_index: Index of current item (-1 if not started)
        full_transcript: FULL conversation transcript (not truncated)

    Returns:
        tuple: (system_prompt, user_prompt)
    """
    # Sanitize transcript to prevent prompt injection
    safe_transcript = sanitize_transcript(full_transcript)

    # Format agenda topics simply and clearly
    agenda_parts = []
    for i, item in enumerate(all_items):
        status = item.get("status", "pending")
        status_emoji = {"completed": "✓", "skipped": "✗", "in_progress": "▶", "pending": "○"}.get(status, "?")

        title = item.get("title", "Unknown")
        item_id = item.get("id", "unknown")
        desc = item.get("description", "")

        topic_line = f"{i+1}. [{status_emoji}] {title}"
        if desc:
            topic_line += f"\n   Description: {desc}"
        topic_line += f"\n   ID: {item_id}"

        agenda_parts.append(topic_line)

    agenda_topics = "\n\n".join(agenda_parts) if agenda_parts else "No agenda defined"

    # Format current topic info
    if current_item and current_index >= 0:
        current_topic = (
            f"Topic #{current_index + 1}: {current_item.get('title', 'Unknown')}\n"
            f"ID: {current_item.get('id', 'unknown')}\n"
            f"Description: {current_item.get('description') or 'None'}"
        )
    else:
        current_topic = "No topic currently tracked (meeting just started)"

    user_prompt = SMART_TOPIC_DETECTION_USER_TEMPLATE.format(
        agenda_topics=agenda_topics,
        current_topic=current_topic,
        full_transcript=safe_transcript
    )

    return SMART_TOPIC_DETECTION_SYSTEM_PROMPT, user_prompt


# ============================================================================
# DEPRECATED - Keep for backwards compatibility during transition
# ============================================================================

UNIFIED_TOPIC_DETECTION_SYSTEM_PROMPT = SMART_TOPIC_DETECTION_SYSTEM_PROMPT
UNIFIED_TOPIC_DETECTION_USER_TEMPLATE = SMART_TOPIC_DETECTION_USER_TEMPLATE
UNIFIED_DETECTION_REQUIRED_FIELDS = SMART_DETECTION_REQUIRED_FIELDS

def format_unified_topic_detection_prompt(
    all_items: list,
    current_item: dict | None,
    current_index: int,
    transcript_text: str
) -> tuple[str, str]:
    """Deprecated: Use format_smart_topic_detection_prompt instead."""
    return format_smart_topic_detection_prompt(all_items, current_item, current_index, transcript_text)
