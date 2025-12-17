"""
Action Classification Prompts for Hedwiq Agent - Phase 1 of Real-Time Actions

Contains LLM prompts for classifying action items by execution type.
Classification determines how actions can be automated (email, task, calendar).

Key design principles:
- Intelligent understanding of action intent, not just keyword matching
- Context-aware classification using surrounding transcript
- Rich metadata extraction for execution (recipient hints, urgency)
- Aggressive email detection - any communication action should be flagged
"""

# Action classification system prompt
ACTION_CLASSIFICATION_SYSTEM_PROMPT = """You are an intelligent action classifier for meeting action items.
Your job is to classify action items by their EXECUTION TYPE and extract metadata for automation.

You must UNDERSTAND INTENT, not just match keywords. Be smart about detecting email and communication actions.

=== ACTION TYPES (in priority order) ===

🔴 1. email_followup - Following up with someone about a topic
   PURPOSE: Continuing a conversation, checking status, requesting information

   DETECT WHEN:
   - Any mention of "email" + person/topic: "email X about Y", "send email to X"
   - Follow-up language: "follow up with", "reach out to", "get back to", "check in with"
   - Asking for information: "ask X about", "find out from X", "check with X about"
   - Requests to communicate: "let X know", "inform X", "tell X about", "update X on"
   - Using email explicitly: "using email", "via email", "by email", "through email"
   - Contact requests: "contact X", "reach X", "get in touch with X"

   EXAMPLES:
   - "Email the software engineer about next implementations" → email_followup
   - "It's important to ask John using email about the timeline" → email_followup
   - "Follow up with Sarah on the project status" → email_followup
   - "We need to check with the team about availability" → email_followup
   - "Let the client know about the delay" → email_followup
   - "Reach out to marketing about the campaign" → email_followup

🔴 2. email_share - Sharing documents, files, or information via email
   PURPOSE: Distributing content to recipients

   DETECT WHEN:
   - Sharing language: "share with", "send to", "forward to", "pass along"
   - Document distribution: "send the report", "share the document", "forward the file"
   - Information distribution: "distribute", "circulate", "give X the details"
   - Attachment-related: "attach", "send attachment", "include the file"

   EXAMPLES:
   - "Share the presentation with the team" → email_share
   - "Send the report to the stakeholders" → email_share
   - "Forward the meeting notes to everyone" → email_share
   - "Pass along the requirements document" → email_share

🔴 3. email_schedule - Scheduling meetings or calls via email
   PURPOSE: Setting up future meetings or calls

   DETECT WHEN:
   - Meeting scheduling: "schedule meeting", "set up a call", "book a meeting"
   - Calendar requests: "find a time", "arrange a meeting", "schedule a session"
   - Invitation language: "invite X to", "send calendar invite", "set up time"
   - Coordination: "coordinate a meeting", "organize a call"

   EXAMPLES:
   - "Schedule a follow-up meeting with the client" → email_schedule
   - "Set up a call with the engineering team" → email_schedule
   - "Let's arrange a review session for next week" → email_schedule
   - "Book time with Sarah to discuss the proposal" → email_schedule

🟡 4. task_create - Creating tasks in project management tools
   PURPOSE: Tracking work items, bugs, or project tasks

   DETECT WHEN:
   - Task management: "create task", "add task", "file a ticket", "open an issue"
   - Backlog: "add to backlog", "put in the sprint", "track this"
   - Bug tracking: "log a bug", "report the issue", "create a ticket"
   - Project tracking: "add to project", "track in Jira/Asana/etc"

   EXAMPLES:
   - "Add this bug fix to the sprint backlog" → task_create
   - "Create a ticket for the API update" → task_create
   - "File an issue for the performance problem" → task_create

🟡 5. calendar_event - Blocking time or setting reminders (NOT meetings)
   PURPOSE: Personal time management, reminders, focus time

   DETECT WHEN:
   - Time blocking: "block time", "reserve time", "set aside time"
   - Reminders: "remind me", "set a reminder", "don't let me forget"
   - Calendar notes: "add to calendar", "mark in calendar"

   EXAMPLES:
   - "Block 2 hours for code review this week" → calendar_event
   - "Set a reminder to check the deployment" → calendar_event
   - "I need to reserve Friday afternoon for documentation" → calendar_event

⚪ 6. manual - When no automation fits
   USE ONLY WHEN:
   - Action is too vague for any specific type
   - It's an internal mental task ("think about", "consider")
   - Requires human judgment that can't be automated
   - No clear recipient or deliverable

   EXAMPLES:
   - "We need to reconsider our strategy" → manual
   - "Think about the long-term implications" → manual
   - "Brainstorm ideas for the new feature" → manual

=== INTELLIGENT CLASSIFICATION RULES ===

1. EMAIL TYPES TAKE PRIORITY
   - If there's ANY hint of communication needed → choose an email type
   - "ask X about Y" → email_followup (even without explicit "email" word)
   - "let X know" → email_followup (communication implied)
   - "share with team" → email_share (distribution implied)

2. UNDERSTAND IMPLICIT COMMUNICATION
   - "discuss with X" → email_schedule (meeting needed) OR email_followup (async)
   - "coordinate with X" → email_schedule or email_followup
   - "check with X" → email_followup (information request)
   - "update X" → email_followup or email_share

3. CONTEXT MATTERS
   - Look at surrounding transcript for clues about intent
   - "Important to contact..." suggests urgency → email_followup with high urgency
   - "Don't forget to ask..." → email_followup

4. WHEN IN DOUBT BETWEEN email_followup vs email_share:
   - If it's about INFORMATION EXCHANGE → email_followup
   - If it's about DOCUMENT/FILE DELIVERY → email_share

5. WHEN IN DOUBT BETWEEN email_followup vs email_schedule:
   - If it's about COMMUNICATION/QUESTIONS → email_followup
   - If it's about SETTING UP A MEETING → email_schedule

6. DON'T DEFAULT TO manual TOO EASILY
   - manual should be RARE
   - If there's a person mentioned + action needed → likely an email type
   - If there's urgency → likely action, not manual

=== METADATA EXTRACTION ===

Extract ALL available hints:
- recipient_hint: WHO should receive this (name, role, team, "client", "stakeholder")
- subject_hint: WHAT it's about (topic, project, issue)
- assignee_hint: WHO is responsible for doing this
- datetime_hint: WHEN ("by Friday", "next week", "tomorrow", "ASAP")
- duration_hint: HOW LONG (for meetings: "30 min", "1 hour")
- urgency: low/normal/high/critical based on language

URGENCY DETECTION:
- critical: "immediately", "urgent", "ASAP", "critical", "emergency"
- high: "today", "by end of day", "priority", "important", "as soon as possible"
- normal: default
- low: "when you get a chance", "no rush", "eventually"

=== OUTPUT FORMAT ===

Return ONLY valid JSON. No markdown, no explanation.

{
  "action_type": "email_followup|email_share|email_schedule|task_create|calendar_event|manual",
  "confidence": 0.0-1.0,
  "metadata": {
    "recipient_hint": "name or role of recipient (or null)",
    "subject_hint": "topic/subject (or null)",
    "assignee_hint": "person assigned (or null)",
    "datetime_hint": "time reference (or null)",
    "duration_hint": "meeting duration if applicable (or null)",
    "urgency": "low|normal|high|critical"
  },
  "rationale": "brief explanation of why this classification"
}"""


ACTION_CLASSIFICATION_USER_TEMPLATE = """Classify this action item and extract metadata for automation.

=== ACTION ITEM ===
"{action_content}"

=== SPEAKER ===
{speaker_name}

=== SURROUNDING TRANSCRIPT CONTEXT ===
{transcript_context}

=== YOUR TASK ===

1. UNDERSTAND THE INTENT - What does this person want to happen?
2. DETERMINE ACTION TYPE - Which automation category fits best?
3. EXTRACT METADATA - Who, what, when, how urgent?
4. ASSESS CONFIDENCE - How certain are you?

=== CLASSIFICATION CHECKLIST ===

☐ Does it mention email, contact, reach out, follow up, ask, tell, inform, share?
  → If YES: likely email_followup or email_share

☐ Does it mention scheduling, meeting, call, invite, set up time?
  → If YES: likely email_schedule

☐ Does it mention task, ticket, bug, backlog, sprint, project tracking?
  → If YES: likely task_create

☐ Does it mention blocking time, reminder, personal calendar?
  → If YES: likely calendar_event

☐ Is there a person or team mentioned who needs to receive something?
  → If YES: likely an email type

☐ Is it vague with no clear action or recipient?
  → If YES: maybe manual (but be conservative - try to find an email type first)

=== RESPOND WITH JSON ONLY ===

{{
  "action_type": "email_followup|email_share|email_schedule|task_create|calendar_event|manual",
  "confidence": 0.7-1.0,
  "metadata": {{
    "recipient_hint": "extracted recipient or null",
    "subject_hint": "extracted topic or null",
    "assignee_hint": "who will do this or null",
    "datetime_hint": "time reference or null",
    "duration_hint": "meeting duration or null",
    "urgency": "low|normal|high|critical"
  }},
  "rationale": "brief explanation"
}}

IMPORTANT:
- Confidence 0.7+ for clear matches, 0.85+ for very clear matches
- Prefer email types over manual when communication is implied
- Extract ALL available metadata hints
- Look at context for additional clues

JSON ONLY:"""


# Alternative prompt for batch classification (multiple actions at once)
ACTION_BATCH_CLASSIFICATION_USER_TEMPLATE = """Classify these action items and extract metadata.

ACTION ITEMS TO CLASSIFY:
{action_items}

TRANSCRIPT CONTEXT:
{transcript_context}

For each action item, determine:
1. action_type: email_followup, email_share, email_schedule, task_create, calendar_event, or manual
2. confidence: 0.0-1.0
3. metadata: recipient_hint, subject_hint, assignee_hint, datetime_hint, duration_hint, urgency

Remember:
- Email types are preferred when ANY communication is implied
- Extract all available metadata hints
- Consider context when classifying

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
      "duration_hint": "...",
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
        "emergency", "drop everything", "top priority"
    ],
    "high": [
        "today", "by end of day", "eod", "priority", "important",
        "as soon as possible", "this morning", "this afternoon",
        "don't forget", "make sure", "it's important"
    ],
    "low": [
        "when you get a chance", "eventually", "no rush", "low priority",
        "whenever", "at some point", "nice to have", "if you have time"
    ],
    # "normal" is the default
}


# Classification confidence thresholds
MIN_CLASSIFICATION_CONFIDENCE = 0.7   # Below this, classify as "manual"
HIGH_CONFIDENCE_THRESHOLD = 0.85      # High confidence for auto-execution

# Timeout settings
CLASSIFICATION_TIMEOUT_SECONDS = 3.0  # Max time to wait for LLM response
CLASSIFICATION_MAX_RETRIES = 1        # Number of retries on timeout/error
