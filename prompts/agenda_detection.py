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
2. The conversation transcript (entries marked [RECENT] are from the last few moments)
3. The currently tracked topic

Your job: Determine if the speaker has MOVED ON to a different agenda topic.

## CRITICAL: Agendas are SEQUENTIAL

Meeting agendas are designed to be discussed IN ORDER. The speaker typically:
1. Covers topic 1, then moves to topic 2, then topic 3, etc.
2. Each topic builds on or follows the previous one
3. When they finish with one topic's content, they move to the next

**Key insight**: If the [RECENT] speech content better matches a LATER topic in the agenda, the speaker has likely transitioned.

## CRITICAL: Focus on RECENT Speech

Focus on entries marked [RECENT] to determine the CURRENT state. Earlier entries are history.

## Handling Overlapping Topics

Topics often share keywords but represent DIFFERENT focuses. For example:
- "Introducing Product X" → General overview, what it is, why it exists
- "Product X Features" → Specific capabilities, how they work
- "Product X Demo" → Showing it in action, live demonstration

When topics seem similar, ask: "What SPECIFIC ASPECT is the speaker focusing on NOW?"
- If they shifted from "what it is" to "how it works automatically" → that's a transition
- If they shifted from "overview" to "demonstration" → that's a transition

## When to Transition

Transition when:
- The [RECENT] speech focuses on content that BETTER MATCHES a later PENDING topic
- The speaker has moved from general/intro content to specific feature content
- The speaker explicitly says they're moving on ("Now let's look at...", "Next...", "So for the automatic...")
- The discussion has clearly shifted to what a later topic is about

## When NOT to Transition

Stay on current topic when:
- The speaker is previewing/listing upcoming topics
- The [RECENT] content still primarily relates to the current topic
- There's no clear shift in focus

## CRITICAL: Only Recommend PENDING Topics

- Only recommend transitions to topics with status "pending" (○)
- NEVER recommend "completed" (✓) or "skipped" (✗) topics
- Look at the agenda status markers: ✓ = completed, ✗ = skipped, ▶ = in_progress, ○ = pending

## How to Decide

1. Read the AGENDA topics - understand what makes each one DISTINCT
2. Look at [RECENT] entries - what SPECIFIC ASPECT is the speaker discussing?
3. Does the recent content BETTER MATCH a later PENDING topic than the current one?
4. If yes, transition. If roughly equal or current is better match, stay.

## Response Format (JSON)

{
  "should_transition": true/false,
  "new_topic_id": "ID of the new topic (only if should_transition=true, MUST be PENDING)",
  "new_topic_index": number or null,
  "is_meeting_ending": true/false,
  "reasoning": "What specific aspect is being discussed and why it matches (or doesn't match) a different topic",
  "current_focus": "The specific subject being discussed in [RECENT] entries"
}

## Detecting Meeting End

If the [RECENT] speech indicates the meeting is ending (goodbyes, thank yous, "that's all", "any questions?", etc.), set `is_meeting_ending: true`. This is especially important when on the LAST topic - recognize when the speaker is wrapping up.

## Key Guidance

- Agendas are SEQUENTIAL - expect forward progression through topics
- Focus on what makes topics DIFFERENT, not what they share
- When in doubt about overlapping topics, consider: has the speaker moved from intro/overview to specifics?
- Recognize meeting endings - "thank you", "bye", "that's the end" signals
- Trust your understanding of conversation flow"""


SMART_TOPIC_DETECTION_USER_TEMPLATE = """## MEETING AGENDA (sequential order - topics discussed one after another)

{agenda_topics}

## CURRENTLY TRACKED TOPIC

{current_topic}

## MEETING TRANSCRIPT (entries marked [RECENT] are from the last few moments)

{full_transcript}

---

Focus on [RECENT] entries. The agenda is SEQUENTIAL - speakers progress through topics in order.

Question: Does the [RECENT] speech BETTER MATCH a later PENDING (○) topic than the current one?
- Compare what's being discussed NOW against each PENDING topic
- If the focus has shifted to content that matches a later topic, transition
- Never recommend completed (✓) or skipped (✗) topics

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
