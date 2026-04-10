"""
JSON parsing and repair utilities for LLM output.

The fine-tuned model *usually* returns clean JSON, but it can and will
hallucinate malformed output.  These helpers try progressively harder
repair strategies before giving up.
"""

from __future__ import annotations

import json
import re


def strip_markdown_fences(text: str) -> str:
    """Remove ```json … ``` or ``` … ``` code fences."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def extract_first_json_object(text: str) -> str | None:
    """
    Walk through *text* and return the first balanced { … } substring,
    respecting JSON string escaping.  Returns None if no opening brace
    is found; returns the substring from the first '{' to end-of-text
    if brackets never balance (caller can still attempt bracket closing).
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape_next:
            escape_next = False
            continue

        if ch == "\\" and in_string:
            escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # Brackets never balanced — return from first '{' to end so the
    # bracket-closer can attempt a fix.
    return text[start:]


def fix_trailing_commas(text: str) -> str:
    """Remove trailing commas immediately before ``}`` or ``]``."""
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*\]", "]", text)
    return text


def close_unclosed_brackets(text: str) -> str:
    """
    Append closing ``}`` / ``]`` characters for any brackets or braces
    that were opened but never closed (common when vLLM truncates output
    due to max_tokens).
    """
    stack: list[str] = []
    in_string = False
    escape_next = False

    for ch in text:
        if escape_next:
            escape_next = False
            continue

        if ch == "\\" and in_string:
            escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch in ("{", "["):
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()

    # Close in reverse order
    closers = {"[": "]", "{": "}"}
    while stack:
        text += closers[stack.pop()]

    return text


# ---- public entry point ---------------------------------------------------


def parse_llm_output(raw_text: str) -> dict | None:
    """
    Attempt to parse LLM output as a JSON **dict**.

    Repair pipeline (each step feeds into the next):
      1. Direct ``json.loads()`` on the stripped text
      2. Strip markdown code fences, then parse
      3. Extract the first ``{ … }`` object from surrounding prose
      4. Fix trailing commas before ``}`` and ``]``
      5. Close any unclosed brackets / braces

    Returns the parsed ``dict``, or ``None`` if every strategy fails.
    """
    if not raw_text or not raw_text.strip():
        return None

    text = raw_text.strip()

    # --- Attempt 1: direct parse ------------------------------------------
    result = _try_parse(text)
    if result is not None:
        return result

    # --- Attempt 2: strip markdown fences ---------------------------------
    stripped = strip_markdown_fences(text)
    result = _try_parse(stripped)
    if result is not None:
        return result

    # --- Attempt 3: extract first JSON object from prose ------------------
    extracted = extract_first_json_object(stripped)
    if extracted is not None:
        result = _try_parse(extracted)
        if result is not None:
            return result
    else:
        # Nothing to work with — no '{' at all
        return None

    # From here on we work with the extracted substring.
    candidate = extracted

    # --- Attempt 4: fix trailing commas -----------------------------------
    candidate = fix_trailing_commas(candidate)
    result = _try_parse(candidate)
    if result is not None:
        return result

    # --- Attempt 5: close unclosed brackets -------------------------------
    candidate = close_unclosed_brackets(candidate)
    result = _try_parse(candidate)
    if result is not None:
        return result

    return None


def _try_parse(text: str) -> dict | None:
    """Return parsed dict or None."""
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None