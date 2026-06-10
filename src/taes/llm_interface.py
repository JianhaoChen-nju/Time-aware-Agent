"""LLM interface for generating candidate events."""

import json
import logging
import os
import re
import time
from typing import List, Optional

from .base import State

logger = logging.getLogger(__name__)


def get_llm():
    """Create a ChatOpenAI instance from .env configuration."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.environ['LLM_MODEL_NAME'],
        temperature=1,
        api_key=os.environ['LLM_API_KEY'],
        base_url=os.environ['LLM_BASE_URL'],
        request_timeout=int(os.environ.get('TAES_LLM_REQUEST_TIMEOUT', '180')),
    )


def call_llm(llm, prompt: str) -> str:
    """Call LLM and return the text response."""
    attempts = int(os.environ.get('TAES_LLM_MAX_RETRIES', '2'))
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            last_error = e
            logger.warning("LLM call failed on attempt %s/%s: %s", attempt, attempts, e)
            if attempt < attempts:
                time.sleep(2 * attempt)
    raise last_error


def parse_json_from_response(text: str) -> Optional[List]:
    """Extract JSON array from LLM response, handling various formats."""
    # Try ```json ... ``` markers
    if '```json' in text:
        try:
            content = text.split('```json')[1].split('```')[0].strip()
            return json.loads(content)
        except (IndexError, json.JSONDecodeError):
            pass

    # Try ``` ... ``` markers
    if '```' in text:
        try:
            content = text.split('```')[1].split('```')[0].strip()
            return json.loads(content)
        except (IndexError, json.JSONDecodeError):
            pass

    # Try to find JSON array directly
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try to find JSON object
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            result = json.loads(match.group())
            return [result] if isinstance(result, dict) else result
        except json.JSONDecodeError:
            pass

    logger.warning(f"Failed to parse JSON from LLM response: {text[:200]}...")
    return None
