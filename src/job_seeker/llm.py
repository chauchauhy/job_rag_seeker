"""Helpers for calling an LLM (via the opencode CLI) and parsing JSON output."""

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from job_seeker.config import settings
from job_seeker.logging_setup import get_logger

logger = get_logger(__name__)

__all__ = [
    "OPENCODE_TIMEOUT_SECONDS",
    "find_opencode_binary",
    "run_llm",
    "send_request_to_openai",
    "extract_json_with_llm",
]


def find_opencode_binary() -> str:
    """Locate the opencode executable, either on PATH or in the npm global install."""
    found = shutil.which("opencode")
    if found:
        return found
    npm_global = Path(os.environ.get("APPDATA", "")) / "npm" / "node_modules" / "opencode-ai" / "bin" / "opencode.exe"
    if npm_global.is_file():
        return str(npm_global)
    raise FileNotFoundError("Cannot locate the opencode executable. Add it to PATH.")


OPENCODE_TIMEOUT_SECONDS = int(os.getenv("OPENCODE_TIMEOUT_SECONDS", "120"))


def _run_opencode(
    prompt: str, model: str, timeout: int
) -> subprocess.CompletedProcess:
    """Invoke the opencode CLI, preferring a warm ``opencode serve`` server.

    When ``OPENCODE_SERVER_URL`` is set, ``run --attach`` reuses a running
    server (started once per container) instead of cold-starting a fresh CLI
    on every call. Falls back to the direct subprocess if the server is
    unreachable or errors.
    """
    binary = find_opencode_binary()
    server_url = os.getenv("OPENCODE_SERVER_URL", "").strip()
    started = time.time()
    if server_url:
        try:
            completed = subprocess.run(
                [
                    binary,
                    "run",
                    "--attach",
                    server_url,
                    "--format",
                    "json",
                    "-m",
                    model,
                ],
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            logger.info(
                "opencode run (attach %s) model=%s rc=%d in %.1fs",
                server_url,
                model,
                completed.returncode,
                time.time() - started,
            )
            return completed
        except (subprocess.SubprocessError, OSError):
            logger.warning(
                "opencode serve unreachable at %s; falling back to direct invocation",
                server_url,
            )
    completed = subprocess.run(
        [binary, "run", "--format", "json", "-m", model],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    logger.info(
        "opencode run (direct) model=%s rc=%d in %.1fs",
        model,
        completed.returncode,
        time.time() - started,
    )
    return completed


def run_llm(prompt: str, model: str | None = None, timeout: int = OPENCODE_TIMEOUT_SECONDS) -> str:
    """Send a prompt to the opencode CLI and return the raw text response.

    The CLI is invoked in JSON event-stream mode; `text` parts are concatenated
    into a single string.

    Raises:
        RuntimeError: if the opencode CLI fails or times out.
    """
    model = model or settings.opencode_model
    try:
        completed = _run_opencode(prompt, model, timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"opencode run timed out after {timeout}s "
            "(slow model/network, or it is waiting for input)."
        ) from exc
    if completed.returncode != 0:
        raise RuntimeError(f"opencode run failed:\n{completed.stderr}")

    text = ""
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") == "text":
            text += event["part"]["text"]
    return text


def send_request_to_openai(prompt: str, model: str | None = None) -> str:
    """Send a prompt to the OpenAI API and return the raw text response.

    The API is invoked in JSON event-stream mode; `text` parts are concatenated
    into a single string.
    """
    import openai

    model = model or settings.opencode_model
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    text = ""
    for event in response:
        if event["choices"][0]["finish_reason"] is not None:
            break
        if "delta" in event["choices"][0]:
            text += event["choices"][0]["delta"].get("content", "")
    return text


def _extract_json_object(text: str) -> str | None:
    """Return the first complete, balanced JSON object found in ``text``.

    Scans from the first ``{`` and walks forward tracking brace depth while
    respecting string literals. This tolerates LLM output that appends trailing
    reasoning or even a repeated/second JSON object after the answer.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json_with_llm(prompt: str, model: str | None = None) -> dict:
    """Ask the LLM to answer a prompt and return the first JSON object in the reply.

    Args:
        prompt: Full prompt, including the expected JSON schema.
        model: Optional model override; defaults to settings.opencode_model.

    Returns:
        Parsed JSON object.

    Raises:
        RuntimeError: if the opencode CLI invocation fails.
        ValueError: if the LLM output contains no parseable JSON object.
    """
    text = run_llm(prompt, model=model)
    json_text = _extract_json_object(text)
    if json_text is not None:
        text = json_text
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse LLM output as JSON: {exc}\nRaw output:\n{text}") from exc
