"""
Email Draft Generation Prompts for Luframe Agent - Phase 3 of Real-Time Actions

Contains LLM prompts for generating professional email drafts from meeting action items.
Drafts are generated when ActionClassifier identifies email-type actions.

Key design principles:
- Professional but personable tone
- Context-aware from meeting transcript
- Clear call-to-action in each email
- Concise (typically 3-5 sentences)
- Reference specific meeting discussion points
"""

# Email draft generation system prompt
EMAIL_DRAFT_SYSTEM_PROMPT = """You are an expert email composer for business meetings.
Your job is to generate professional, contextually appropriate email drafts based on action items from meetings.

TONE & STYLE:
- Professional but personable (not overly formal or stiff)
- Clear and direct communication
- Concise - typically 3-5 sentences for the main body
- Action-oriented with clear next steps
- Reference the meeting naturally without being verbose

EMAIL STRUCTURE:
1. Brief greeting
2. Context reference (mention the meeting briefly)
3. Main message (the purpose/request)
4. Clear call-to-action or next step
5. Professional closing

RECIPIENT HANDLING:
- If a specific person is mentioned, use their name
- If a role/team is mentioned, address appropriately
- If unclear, use a generic but professional greeting

DO NOT:
- Be overly verbose or include unnecessary pleasantries
- Use overly formal language ("I hope this email finds you well")
- Include placeholder text like [INSERT NAME] or [DATE]
- Add email signatures (the app handles that)
- Include meeting links unless specifically requested

Return JSON only. No markdown, no explanation."""


EMAIL_DRAFT_USER_TEMPLATE = """Generate a professional email draft based on this meeting action item.

ACTION ITEM: "{action_content}"

ACTION TYPE: {action_type}
- email_followup: Following up on a discussion point
- email_share: Sharing information/documents
- email_schedule: Requesting to schedule a meeting/call

SPEAKER WHO MENTIONED THIS: {speaker_name}

METADATA HINTS:
- Recipient hint: {recipient_hint}
- Subject hint: {subject_hint}
- Urgency: {urgency}

MEETING CONTEXT:
- Meeting title: {meeting_title}
- Participants: {participants}
- Agenda topics: {agenda_topics}

RECENT TRANSCRIPT (for additional context):
{transcript_context}

INSTRUCTIONS:
1. Generate an appropriate email subject line
2. Identify the most likely recipient(s) from context
3. Write a professional but friendly email body
4. Include a clear call-to-action
5. Rate your confidence in the draft quality

Respond with JSON:
{{
  "subject": "Email subject line",
  "recipients": [
    {{
      "name": "Recipient name or role",
      "source": "explicit|inferred|participant"
    }}
  ],
  "body": "Full email body text (no greeting/closing - just the content)",
  "greeting": "Hi [Name]," or "Hello team,",
  "closing": "Best regards," or "Thanks,",
  "confidence": 0.0-1.0,
  "rationale": "Brief explanation of draft approach"
}}

CRITICAL:
- Subject should be specific and informative
- Body should be 2-5 sentences typically
- Recipients should be based on who is mentioned or implied
- Confidence reflects how well the context supports the draft

JSON only:"""


# Alternative prompt for scheduling-type emails (email_schedule)
EMAIL_SCHEDULE_DRAFT_TEMPLATE = """Generate a meeting request email based on this action item.

ACTION ITEM: "{action_content}"

SPEAKER: {speaker_name}

MEETING CONTEXT:
- Current meeting: {meeting_title}
- Participants: {participants}

METADATA:
- Recipient hint: {recipient_hint}
- Time hint: {datetime_hint}
- Duration hint: {duration_hint}

RECENT TRANSCRIPT:
{transcript_context}

Generate a professional meeting request email.

Respond with JSON:
{{
  "subject": "Meeting request subject",
  "recipients": [
    {{"name": "...", "source": "explicit|inferred|participant"}}
  ],
  "body": "Meeting request body (include purpose, suggest times if hint available)",
  "greeting": "Hi [Name],",
  "closing": "Looking forward to connecting,",
  "confidence": 0.0-1.0,
  "rationale": "..."
}}

JSON only:"""


# Alternative prompt for sharing-type emails (email_share)
EMAIL_SHARE_DRAFT_TEMPLATE = """Generate an email to share information/documents based on this action item.

ACTION ITEM: "{action_content}"

SPEAKER: {speaker_name}

MEETING CONTEXT:
- Meeting: {meeting_title}
- Participants: {participants}
- Topics discussed: {agenda_topics}

METADATA:
- Recipient hint: {recipient_hint}
- Subject hint: {subject_hint}

RECENT TRANSCRIPT:
{transcript_context}

Generate a professional email for sharing the mentioned content.

Respond with JSON:
{{
  "subject": "Sharing: [topic]",
  "recipients": [
    {{"name": "...", "source": "explicit|inferred|participant"}}
  ],
  "body": "Body explaining what's being shared and why",
  "greeting": "Hi [Name],",
  "closing": "Let me know if you have questions,",
  "confidence": 0.0-1.0,
  "rationale": "..."
}}

JSON only:"""


def format_email_draft_prompt(
    action_content: str,
    action_type: str,
    speaker_name: str,
    recipient_hint: str | None,
    subject_hint: str | None,
    urgency: str,
    meeting_title: str | None,
    participants: list[str],
    agenda_topics: list[str],
    transcript_context: str,
    datetime_hint: str | None = None,
    duration_hint: str | None = None,
) -> tuple[str, str]:
    """
    Format the email draft generation prompt.

    Args:
        action_content: The action item content
        action_type: Type of email action (email_followup, email_share, email_schedule)
        speaker_name: Name of person who mentioned the action
        recipient_hint: Hint about intended recipient
        subject_hint: Hint about email subject
        urgency: Urgency level (low, normal, high, critical)
        meeting_title: Title of the meeting
        participants: List of meeting participant names
        agenda_topics: List of agenda topics
        transcript_context: Recent transcript for context
        datetime_hint: Time hint for scheduling emails
        duration_hint: Duration hint for scheduling emails

    Returns:
        tuple: (system_prompt, user_prompt)
    """
    # Choose template based on action type
    if action_type == "email_schedule":
        user_template = EMAIL_SCHEDULE_DRAFT_TEMPLATE.format(
            action_content=action_content,
            speaker_name=speaker_name or "Unknown",
            meeting_title=meeting_title or "Meeting",
            participants=", ".join(participants) if participants else "Not specified",
            recipient_hint=recipient_hint or "Not specified",
            datetime_hint=datetime_hint or "Not specified",
            duration_hint=duration_hint or "Not specified",
            transcript_context=transcript_context or "No additional context.",
        )
    elif action_type == "email_share":
        user_template = EMAIL_SHARE_DRAFT_TEMPLATE.format(
            action_content=action_content,
            speaker_name=speaker_name or "Unknown",
            meeting_title=meeting_title or "Meeting",
            participants=", ".join(participants) if participants else "Not specified",
            agenda_topics=", ".join(agenda_topics) if agenda_topics else "General discussion",
            recipient_hint=recipient_hint or "Not specified",
            subject_hint=subject_hint or "Not specified",
            transcript_context=transcript_context or "No additional context.",
        )
    else:
        # Default: email_followup
        user_template = EMAIL_DRAFT_USER_TEMPLATE.format(
            action_content=action_content,
            action_type=action_type,
            speaker_name=speaker_name or "Unknown",
            recipient_hint=recipient_hint or "Not specified",
            subject_hint=subject_hint or "Not specified",
            urgency=urgency or "normal",
            meeting_title=meeting_title or "Meeting",
            participants=", ".join(participants) if participants else "Not specified",
            agenda_topics=", ".join(agenda_topics) if agenda_topics else "General discussion",
            transcript_context=transcript_context or "No additional context.",
        )

    return EMAIL_DRAFT_SYSTEM_PROMPT, user_template


# Timeout and retry settings
DRAFT_GENERATION_TIMEOUT_SECONDS = 5.0
DRAFT_GENERATION_MAX_RETRIES = 1

# Confidence thresholds
MIN_DRAFT_CONFIDENCE = 0.6           # Minimum confidence to publish
HIGH_DRAFT_CONFIDENCE = 0.8          # High confidence drafts

# Example templates for reference (not used in prompts)
EXAMPLE_FOLLOWUP_EMAIL = """
Subject: Follow-up on project timeline discussion

Hi Sarah,

Following up on our conversation during today's product sync - I wanted to confirm
the revised timeline we discussed for the Q1 launch. You mentioned targeting March 15th
instead of the original March 1st date.

Could you share the updated project plan when you have a chance? I want to make sure
we're aligned before the stakeholder review next week.

Thanks,
"""

EXAMPLE_SHARE_EMAIL = """
Subject: Sharing the Q4 metrics dashboard as discussed

Hi team,

As mentioned in our meeting, I'm sharing the Q4 metrics dashboard that John walked
us through. This includes the customer satisfaction scores and NPS trends we reviewed.

Let me know if you have any questions or need access to the underlying data.

Best,
"""

EXAMPLE_SCHEDULE_EMAIL = """
Subject: Follow-up meeting request: Technical architecture review

Hi Alex,

Per our discussion today, I'd like to schedule a deeper dive into the technical
architecture questions that came up. Would you have 30 minutes later this week?

I'm generally available Thursday and Friday afternoon. Let me know what works for you.

Looking forward to connecting,
"""
