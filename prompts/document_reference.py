"""
Document Reference Prompts for Hedwiq Agent - Phase 3 Implementation

Contains the single LLM alignment prompt for validating document references.
The alignment step runs AFTER hybrid retrieval returns top-k candidates.

Key design principles:
- Single LLM call per segment (not 3 layers like v1)
- Clear validation criteria to avoid false positives
- Structured JSON output for reliable parsing
- Evidence span extraction for frontend highlighting
"""

# Single LLM alignment prompt for Phase 3
# This validates retrieval candidates against the transcript
DOCUMENT_ALIGNMENT_SYSTEM_PROMPT = """You are a document reference detector.
Your job is to determine if a speech segment references content from document sections.

You must be:
1. PRECISE - Only confirm references with clear factual overlap
2. CONSERVATIVE - The speaker must be discussing content FROM the document, not just similar topics
3. EVIDENCE-BASED - Extract exact text spans that prove the reference

A valid reference requires:
- Specific factual content from the document is being discussed
- Not just topical similarity (e.g., both mention "revenue" is not enough)
- Clear evidence that the speaker is referencing the document content

Return JSON only. No markdown, no explanation."""


DOCUMENT_ALIGNMENT_USER_TEMPLATE = """Determine if this speech references any of the document sections below.

SPEECH: "{transcript}"

CANDIDATE DOCUMENT SECTIONS:
{candidate_sections}

INSTRUCTIONS:
1. Look for SPECIFIC factual overlap between speech and document sections
2. The speaker must be discussing content FROM the document, not just similar topics
3. Copy the EXACT evidence span from the document (10-50 characters)
4. A vague topical match is NOT a valid reference

If a clear reference exists, respond with JSON:
{{"found": true, "section_id": "segment ID from candidates", "page_number": N, "evidence_span": "exact text from document", "confidence": 0.7-1.0, "rationale": "brief explanation of why this is a clear reference"}}

If NO clear reference (speech is about similar topic but not from document):
{{"found": false, "rationale": "brief explanation of why no reference found"}}

CRITICAL: Only return found=true if the speaker is CLEARLY referencing specific content from the document, not just discussing similar topics.

JSON only:"""


def format_alignment_prompt(
    transcript: str,
    candidates: list,
) -> tuple[str, str]:
    """
    Format the alignment prompt with transcript and candidates.

    Args:
        transcript: The speech transcript to analyze
        candidates: List of RetrievalCandidate objects or dicts

    Returns:
        tuple: (system_prompt, user_prompt)
    """
    # Format candidate sections
    sections_parts = []
    for c in candidates:
        # Handle both RetrievalCandidate objects and dicts
        if hasattr(c, 'segment_id'):
            section_id = c.segment_id
            page_number = c.page_number
            section_title = c.section_title or 'Section'
            content = c.content
        else:
            section_id = c.get('segment_id', c.get('id', 'unknown'))
            page_number = c.get('page_number', 1)
            section_title = c.get('section_title') or 'Section'
            content = c.get('content', '')

        sections_parts.append(
            f"[{section_id}] Page {page_number} - {section_title}:\n{content}"
        )

    candidate_sections = "\n\n".join(sections_parts)

    user_prompt = DOCUMENT_ALIGNMENT_USER_TEMPLATE.format(
        transcript=transcript,
        candidate_sections=candidate_sections
    )

    return DOCUMENT_ALIGNMENT_SYSTEM_PROMPT, user_prompt


# Confidence thresholds
MIN_ALIGNMENT_CONFIDENCE = 0.7  # Minimum confidence to publish a reference
HIGH_CONFIDENCE_THRESHOLD = 0.85  # High confidence references (for potential highlighting)

# Timeout settings for LLM alignment
ALIGNMENT_TIMEOUT_SECONDS = 6.0  # Max time to wait for LLM response (longer to reduce drop-offs)
ALIGNMENT_MAX_RETRIES = 2  # Number of retries on timeout/error
