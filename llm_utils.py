"""
Utility functions for Luframe Agent.

Common helpers shared across modules.
"""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger("luframe-agent")


def clean_llm_json_response(response: str) -> str:
    """
    Clean markdown code fences from LLM JSON responses.

    LLMs often wrap JSON in markdown code blocks like:
    ```json
    {"key": "value"}
    ```

    This function strips those wrappers to get clean JSON.

    Args:
        response: Raw LLM response text

    Returns:
        Cleaned response with markdown fences removed
    """
    cleaned = response.strip()

    # Remove ```json or ``` prefix
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    # Remove ``` suffix
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()


def parse_llm_json(response: str) -> Optional[Any]:
    """
    Parse JSON from an LLM response, handling markdown code fences.

    Args:
        response: Raw LLM response text

    Returns:
        Parsed JSON data, or None if parsing fails
    """
    cleaned = clean_llm_json_response(response)

    if not cleaned:
        return None

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM JSON: {e}")
        logger.debug(f"Raw response: {response}")
        return None
