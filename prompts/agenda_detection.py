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
# Unified Topic Detection (Pure LLM Context Analysis) - REVISED
# ============================================================================

UNIFIED_TOPIC_DETECTION_SYSTEM_PROMPT = """You are a meeting analyst detecting topic transitions in real-time.

Your task is to detect when the conversation has **ACTUALLY MOVED ON** to a DIFFERENT topic - not just mentioned it.

## CRITICAL DISTINCTION: Mentions vs Actual Discussion

**MENTIONING a topic is NOT the same as DISCUSSING it:**
- "Today we'll talk about AI agents, coding agents, and computer use agents" = MENTIONING (still on current topic)
- "So let me explain how computer use agents work. They use screenshots to..." = ACTUALLY DISCUSSING (transition!)

**Signs of ACTUAL topic discussion (transition):**
- Speaker is EXPLAINING or ELABORATING on the new topic
- Multiple sentences ABOUT the new topic's content
- Detailed information, examples, or analysis of the new topic
- The new topic is the MAIN FOCUS, not a preview/mention

**Signs of just MENTIONING (NO transition):**
- Listing topics that will be covered ("we'll discuss X, Y, and Z")
- Brief reference while still on current topic
- Introducing/previewing what's coming next
- Single sentence mentions without elaboration

## How to Analyze

1. Read the RECENT transcript carefully
2. Ask: Is the speaker **actively discussing** a different topic, or just **mentioning** it?
3. Only recommend transition if there is **sustained discussion** of the new topic (not just a mention)

## Response Format (JSON only)

{
  "should_transition": true/false,
  "recommended_topic_id": "ID of topic that best matches RECENT discussion",
  "recommended_topic_index": number (0-based index),
  "match_scores": {
    "current_topic": 0.0-1.0,
    "recommended_topic": 0.0-1.0
  },
  "confidence": 0.0-1.0,
  "reason": "Brief explanation",
  "transition_signal": "What in the transcript indicates topic change (or 'none')",
  "is_actual_discussion": true/false
}

## Confidence Guidelines

- **0.90+**: Speaker is clearly explaining/discussing the new topic in detail
- **0.80-0.90**: Strong indication of new topic discussion (multiple sentences about it)
- **0.70-0.80**: Possible transition - some discussion of new topic
- **<0.70**: Just mentions, previews, or unclear - DO NOT transition

## CRITICAL RULES

1. **MENTIONS ARE NOT TRANSITIONS** - "We'll cover coding agents next" is NOT a transition to coding agents
2. **Require SUSTAINED discussion** - at least 2-3 sentences actually ABOUT the new topic
3. **Introductions stay on current topic** - "Today we'll discuss X, Y, Z" keeps us on the intro/current topic
4. **When in doubt, DON'T transition** - false positives are worse than missed transitions
5. **Look for EXPLANATION, not just KEYWORDS** - seeing "computer use" isn't enough; they must be EXPLAINING it"""


UNIFIED_TOPIC_DETECTION_USER_TEMPLATE = """## MEETING AGENDA (in order)

{all_topics}

---

## CURRENTLY TRACKED TOPIC

{current_topic_info}

---

## RECENT TRANSCRIPT (analyze for topic detection)

{transcript}

---

## YOUR TASK

1. Read the transcript above (focus on the MOST RECENT statements)
2. Compare how well the discussion matches the CURRENT topic vs OTHER topics
3. If the recent discussion better matches a DIFFERENT topic, set should_transition=true

Respond with JSON only:"""


# Required fields for unified detection response
UNIFIED_DETECTION_REQUIRED_FIELDS = ["should_transition", "recommended_topic_id", "confidence"]


def format_unified_topic_detection_prompt(
    all_items: list,
    current_item: dict | None,
    current_index: int,
    transcript_text: str
) -> tuple[str, str]:
    """
    Format the unified topic detection prompt.

    This is the PRIMARY detection method - pure LLM context analysis.
    No word patterns, no keyword matching. The LLM analyzes conversation
    content and determines if a topic transition should occur.

    REVISED: Now explicitly asks for transition detection, not just topic matching.
    Includes match scores for current vs recommended topic to enable comparison.

    Args:
        all_items: All agenda items (not just upcoming)
        current_item: Current active item or None
        current_index: Index of current item (-1 if not started)
        transcript_text: Recent transcript text

    Returns:
        tuple: (system_prompt, user_prompt)
    """
    # Sanitize transcript to prevent prompt injection
    safe_transcript = sanitize_transcript(transcript_text)

    # Format all topics with their details - emphasize NON-current topics
    topics_parts = []
    for i, item in enumerate(all_items):
        status = item.get("status", "pending")

        # Status indicator with emphasis
        if status == "completed":
            status_indicator = " ✓ [COMPLETED - skip this]"
        elif status == "skipped":
            status_indicator = " ✗ [SKIPPED - skip this]"
        elif status == "in_progress":
            status_indicator = " ▶ [CURRENT]"
        else:
            status_indicator = " ○ [PENDING - could transition to this]"

        desc = item.get("description") or "No description provided"

        # Include more context for pending topics to help LLM match
        topics_parts.append(
            f"Topic {i + 1}: {item.get('title', 'Unknown')}{status_indicator}\n"
            f"   ID: {item.get('id', 'unknown')}\n"
            f"   Description: {desc}"
        )

    all_topics = "\n\n".join(topics_parts) if topics_parts else "No agenda topics defined"

    # Format current topic info with clear emphasis
    if current_item and current_index >= 0:
        current_topic_info = (
            f"Topic Index: {current_index} (0-based)\n"
            f"Topic ID: {current_item.get('id', 'unknown')}\n"
            f"Title: \"{current_item.get('title', 'Unknown')}\"\n"
            f"Description: {current_item.get('description') or 'No description'}\n"
            f"\n⚠️ IMPORTANT: Only stay on this topic if the RECENT transcript clearly matches it.\n"
            f"If the discussion has moved to content matching another topic, recommend transition!"
        )
    else:
        current_topic_info = (
            "No topic currently active (meeting just started)\n"
            "→ Recommend the topic that best matches the current discussion."
        )

    user_prompt = UNIFIED_TOPIC_DETECTION_USER_TEMPLATE.format(
        all_topics=all_topics,
        current_topic_info=current_topic_info,
        transcript=safe_transcript
    )

    return UNIFIED_TOPIC_DETECTION_SYSTEM_PROMPT, user_prompt
