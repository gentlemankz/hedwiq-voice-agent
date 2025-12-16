"""
Action Classification Prompts for Hedwiq Agent - Phase 1 of Real-Time Actions

Contains LLM prompts for classifying action items by execution type.
Classification determines how actions can be automated (email, task, calendar).

Key design principles:
- Clear taxonomy of action types
- Context-aware classification using surrounding transcript
- Metadata extraction for execution (recipient hints, urgency)
- Conservative classification with confidence scores
"""

# Action classification system prompt
ACTION_CLASSIFICATION_SYSTEM_PROMPT = """You are an action classifier for meeting action items.
Your job is to classify an action item by its execution type and extract metadata.

ACTION TYPES:
1. email_followup - Following up with someone via email
   Triggers: "send email", "email about", "follow up with", "reach out to", "drop a line"
   Example: "John will email Sarah about the project timeline"

2. email_share - Sharing documents or information via email
   Triggers: "share with", "send to", "forward to", "pass along", "distribute"
   Example: "I'll share the report with the team"

3. email_schedule - Scheduling a meeting via email
   Triggers: "schedule meeting", "set up a call", "book a meeting", "arrange time"
   Example: "Let's schedule a follow-up call with the client"

4. task_create - Creating a task in a project management tool
   Triggers: "create task", "add to backlog", "file a ticket", "track this"
   Example: "Add the bug fix to the sprint backlog"

5. calendar_event - Blocking time or setting reminders
   Triggers: "block time", "add to calendar", "remind me", "set a reminder"
   Example: "I need to block 2 hours for code review this week"

6. manual - Default when action doesn't fit other categories
   Use when: Action is too vague, internal process, or no automation fits
   Example: "We need to think about restructuring the team"

METADATA EXTRACTION:
- recipient_hint: Who should receive the email/task (name, role, or team)
- subject_hint: What the email/action is about
- assignee_hint: Who is responsible for the action
- datetime_hint: Any time references (e.g., "by Friday", "next week")
- urgency: low/normal/high/critical based on language

CLASSIFICATION RULES:
1. Be CONSERVATIVE - only classify with high confidence
2. If multiple types could apply, choose the most specific
3. Email types take precedence when explicit communication is needed
4. Default to "manual" if classification is uncertain
5. Extract all available metadata hints from context

Return JSON only. No markdown, no explanation."""


ACTION_CLASSIFICATION_USER_TEMPLATE = """Classify this action item and extract metadata.

ACTION ITEM: "{action_content}"

SPEAKER: {speaker_name}

SURROUNDING TRANSCRIPT CONTEXT:
{transcript_context}

INSTRUCTIONS:
1. Determine the most appropriate action_type
2. Extract any metadata hints from the action and context
3. Assess urgency based on language used
4. Provide a confidence score (0.0-1.0)

Respond with JSON:
{{
  "action_type": "email_followup|email_share|email_schedule|task_create|calendar_event|manual",
  "confidence": 0.0-1.0,
  "metadata": {{
    "recipient_hint": "name or role of recipient (or null)",
    "subject_hint": "topic/subject (or null)",
    "assignee_hint": "person assigned (or null)",
    "datetime_hint": "time reference (or null)",
    "urgency": "low|normal|high|critical"
  }},
  "rationale": "brief explanation of classification"
}}

CRITICAL:
- Only use high confidence (0.7+) for clear matches
- Extract metadata hints that would help with execution
- Urgency defaults to "normal" unless explicit urgency language is used

JSON only:"""


# Alternative prompt for batch classification (multiple actions at once)
ACTION_BATCH_CLASSIFICATION_USER_TEMPLATE = """Classify these action items and extract metadata.

ACTION ITEMS TO CLASSIFY:
{action_items}

TRANSCRIPT CONTEXT:
{transcript_context}

For each action item, determine:
1. action_type: email_followup, email_share, email_schedule, task_create, calendar_event, or manual
2. confidence: 0.0-1.0
3. metadata: recipient_hint, subject_hint, assignee_hint, datetime_hint, urgency

Respond with JSON array:
[
  {{
    "action_id": "id from above",
    "action_type": "...",
    "confidence": 0.0-1.0,
    "metadata": {{
      "recipient_hint": "...",
      "subject_hint": "...",
      "assignee_hint": "...",
      "datetime_hint": "...",
      "urgency": "low|normal|high|critical"
    }},
    "rationale": "..."
  }}
]

JSON only:"""


def format_classification_prompt(
    action_content: str,
    speaker_name: str,
    transcript_context: str,
) -> tuple[str, str]:
    """
    Format the classification prompt with action and context.

    Args:
        action_content: The action item content to classify
        speaker_name: Name of the speaker who mentioned the action
        transcript_context: Surrounding transcript for context

    Returns:
        tuple: (system_prompt, user_prompt)
    """
    user_prompt = ACTION_CLASSIFICATION_USER_TEMPLATE.format(
        action_content=action_content,
        speaker_name=speaker_name or "Unknown",
        transcript_context=transcript_context or "No additional context available.",
    )

    return ACTION_CLASSIFICATION_SYSTEM_PROMPT, user_prompt


def format_batch_classification_prompt(
    action_items: list,
    transcript_context: str,
) -> tuple[str, str]:
    """
    Format the batch classification prompt for multiple actions.

    Args:
        action_items: List of dicts with {id, content, speaker}
        transcript_context: Shared transcript context

    Returns:
        tuple: (system_prompt, user_prompt)
    """
    items_text = "\n".join([
        f"[{item['id']}] ({item.get('speaker', 'Unknown')}): {item['content']}"
        for item in action_items
    ])

    user_prompt = ACTION_BATCH_CLASSIFICATION_USER_TEMPLATE.format(
        action_items=items_text,
        transcript_context=transcript_context or "No additional context available.",
    )

    return ACTION_CLASSIFICATION_SYSTEM_PROMPT, user_prompt


# Urgency detection patterns (for reference, actual classification uses LLM)
URGENCY_PATTERNS = {
    "critical": [
        "immediately", "right now", "urgent", "asap", "critical",
        "emergency", "drop everything"
    ],
    "high": [
        "today", "by end of day", "eod", "priority", "important",
        "as soon as possible", "this morning", "this afternoon"
    ],
    "low": [
        "when you get a chance", "eventually", "no rush", "low priority",
        "whenever", "at some point", "nice to have"
    ],
    # "normal" is the default
}


# Classification confidence thresholds
MIN_CLASSIFICATION_CONFIDENCE = 0.7   # Below this, classify as "manual"
HIGH_CONFIDENCE_THRESHOLD = 0.85      # High confidence for auto-execution

# Timeout settings
CLASSIFICATION_TIMEOUT_SECONDS = 3.0  # Max time to wait for LLM response
CLASSIFICATION_MAX_RETRIES = 1        # Number of retries on timeout/error
