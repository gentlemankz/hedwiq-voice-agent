"""
Agenda Tracking Prompts for Hedwiq Agent

Contains prompts for detecting topic progression and agenda item completion
during meeting conversations. Uses conservative detection to minimize false positives.

The agent analyzes recent transcript to determine:
1. Whether the current agenda topic appears complete
2. Whether discussion has moved to a different agenda topic
3. Confidence level of the detection
"""

# System prompt - defines role and detection rules
AGENDA_TRACKING_SYSTEM_PROMPT = """You are a meeting progress tracker.
Your job is to determine if a meeting has moved from one agenda topic to another.

You will receive:
1. The meeting agenda with numbered topics
2. The current topic being discussed
3. Recent transcript of the conversation

Your task:
1. Determine if the current topic appears COMPLETE based on the conversation
2. Identify if speakers have explicitly or implicitly moved to a new topic
3. Provide confidence score and evidence

IMPORTANT RULES:
- Be CONSERVATIVE - only mark complete when clearly done
- Look for EXPLICIT transitions: "Let's move on to...", "Next topic...", "Moving forward...", "That covers...", "Let's discuss..."
- Look for IMPLICIT transitions: discussion clearly shifted to next agenda item content
- A topic is complete when:
  * Speakers explicitly conclude it ("That covers the technical requirements", "We've finished discussing...")
  * Discussion naturally shifts to the next agenda topic's content
  * Someone initiates the next topic without explicit transition
- Do NOT mark complete just because there's:
  * A pause in conversation
  * A brief tangent or off-topic comment
  * Someone asking a clarifying question about the current topic
  * General discussion still related to current topic
- Confidence thresholds:
  * 0.9+: Explicit verbal transition ("Let's move on to...")
  * 0.8-0.89: Clear implicit transition (discussion shifted to next item's content)
  * 0.7-0.79: Possible transition but uncertain
  * Below 0.7: Not enough evidence for transition
- Return JSON only, no explanation outside JSON"""


# User prompt template - uses .format() with variables
AGENDA_TRACKING_USER_TEMPLATE = """Analyze if the current meeting topic is complete.

MEETING AGENDA:
{agenda_items}

CURRENT TOPIC (index {current_index}):
Title: {current_topic_title}
Description: {current_topic_description}

NEXT TOPIC (if any):
{next_topic_info}

RECENT TRANSCRIPT:
{transcript}

Determine:
1. Is the current topic "{current_topic_title}" complete?
2. Has discussion moved to a different topic?
3. If yes, which agenda item (by index)?

Return JSON only:
{{
  "current_topic_complete": true/false,
  "confidence": 0.0-1.0,
  "evidence": "Brief quote or description (max 100 chars)",
  "next_topic_started": true/false,
  "next_topic_index": null or 0-N
}}"""


def format_agenda_items(items: list) -> str:
    """Format agenda items for inclusion in prompt."""
    if not items:
        return "No agenda items."

    lines = []
    for idx, item in enumerate(items):
        lead_by = f" (Led by {item.get('lead_by', 'TBD')})" if item.get('lead_by') else ""
        duration = f" - {item.get('estimated_minutes', '?')} min" if item.get('estimated_minutes') else ""
        lines.append(f"{idx}. {item.get('title', 'Untitled')}{lead_by}{duration}")
        if item.get('description'):
            lines.append(f"   Description: {item['description'][:100]}...")

    return "\n".join(lines)


def format_next_topic_info(items: list, current_index: int) -> str:
    """Format information about the next topic."""
    next_idx = current_index + 1
    if next_idx >= len(items):
        return "No next topic (this is the last item)"

    next_item = items[next_idx]
    parts = [f"Title: {next_item.get('title', 'Untitled')}"]
    if next_item.get('description'):
        parts.append(f"Description: {next_item['description'][:100]}")

    return "\n".join(parts)


def format_tracking_prompt(
    agenda_items: list,
    current_index: int,
    transcript_entries: list,
) -> tuple[str, str]:
    """
    Format the complete tracking prompt for LLM analysis.

    Args:
        agenda_items: List of agenda item dicts with title, description, etc.
        current_index: Index of the currently active agenda item
        transcript_entries: List of recent transcript entries with speaker and text

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Format agenda items
    agenda_formatted = format_agenda_items(agenda_items)

    # Get current topic info
    current_item = agenda_items[current_index] if 0 <= current_index < len(agenda_items) else {}
    current_title = current_item.get('title', 'Unknown Topic')
    current_description = current_item.get('description', 'No description provided')

    # Format next topic info
    next_topic_info = format_next_topic_info(agenda_items, current_index)

    # Format transcript
    transcript_lines = []
    for entry in transcript_entries:
        speaker = entry.get('speaker_identity', entry.get('speaker', 'unknown'))
        text = entry.get('text', '')
        transcript_lines.append(f"[{speaker}]: {text}")
    transcript_formatted = "\n".join(transcript_lines) if transcript_lines else "No recent transcript."

    # Build user prompt
    user_prompt = AGENDA_TRACKING_USER_TEMPLATE.format(
        agenda_items=agenda_formatted,
        current_index=current_index,
        current_topic_title=current_title,
        current_topic_description=current_description,
        next_topic_info=next_topic_info,
        transcript=transcript_formatted,
    )

    return AGENDA_TRACKING_SYSTEM_PROMPT, user_prompt


# Additional prompts for edge cases

AGENDA_START_DETECTION_PROMPT = """The meeting has started but no topic has been marked as in_progress yet.
Analyze the transcript to determine if discussion has begun on the first agenda item.

FIRST AGENDA ITEM:
Title: {first_topic_title}
Description: {first_topic_description}

RECENT TRANSCRIPT:
{transcript}

Has discussion of the first topic begun? Return JSON only:
{{
  "first_topic_started": true/false,
  "confidence": 0.0-1.0,
  "evidence": "Brief quote or description (max 100 chars)"
}}"""


AGENDA_COMPLETION_CHECK_PROMPT = """Analyze if the meeting agenda is fully complete.

MEETING AGENDA:
{agenda_items}

CURRENT STATUS:
- Items completed: {completed_count}/{total_count}
- Last completed item: {last_completed_title}

RECENT TRANSCRIPT:
{transcript}

Signs the meeting is wrapping up:
- "That's everything on the agenda"
- "Let's wrap up"
- "Any final questions?"
- "Thanks everyone for the productive meeting"

Is the entire agenda complete? Return JSON only:
{{
  "agenda_complete": true/false,
  "confidence": 0.0-1.0,
  "evidence": "Brief quote or description (max 100 chars)"
}}"""
