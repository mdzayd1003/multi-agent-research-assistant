"""LLM backends.

``OllamaLLM`` talks to a local Ollama server over HTTP with nothing but the
standard library. ``FakeLLM`` is deterministic and offline: it reads the role
tag out of the prompt and returns a contract-shaped reply built from whatever
context it was given, which is what lets the whole orchestration path be tested
without a model.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol

_ROLE_RE = re.compile(r"^ROLE:\s*(\w+)", re.MULTILINE)
_QUESTION_RE = re.compile(r"^QUESTION:\s*(.+)$", re.MULTILINE)
_CITATION_RE = re.compile(r"\[([^\]\s]+#\d+)\]")


class LLM(Protocol):
    name: str

    def complete(self, prompt: str, *, system: str = "", temperature: float = 0.0) -> str:
        ...


class LLMError(RuntimeError):
    """The backend could not be reached or returned an error."""


@dataclass
class OllamaLLM:
    """Minimal Ollama client. No SDK, no retries, a hard timeout."""

    model: str = "llama3.1:8b"
    host: str = "http://localhost:11434"
    timeout: float = 120.0
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"ollama:{self.model}"

    def complete(self, prompt: str, *, system: str = "", temperature: float = 0.0) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {"temperature": temperature},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host.rstrip('/')}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"ollama request failed: {exc}") from exc
        return str(payload.get("response", ""))


@dataclass
class FakeLLM:
    """Deterministic stand-in that honours the JSON contract.

    Behaviour by role:

    * ``orchestrator`` -- returns a two-subtask plan derived from the question.
    * ``researcher``   -- calls ``search_corpus`` once, then answers with the
      citations that came back, so a broken retriever shows up as a broken answer.
    * ``analyst``/``writer`` -- answer straight from the context block.
    * ``critic``       -- approves when the draft carries a citation, otherwise
      asks for one. That makes the revise loop observable in tests.
    """

    name: str = "fake"
    calls: List[str] = field(default_factory=list)
    scripted: Optional[Callable[[str, str], Optional[str]]] = None

    def complete(self, prompt: str, *, system: str = "", temperature: float = 0.0) -> str:
        self.calls.append(prompt)
        if self.scripted is not None:
            override = self.scripted(system, prompt)
            if override is not None:
                return override

        role_match = _ROLE_RE.search(system) or _ROLE_RE.search(prompt)
        role = role_match.group(1) if role_match else "researcher"
        question_match = _QUESTION_RE.search(prompt)
        question = question_match.group(1).strip() if question_match else "the question"
        citations = _CITATION_RE.findall(prompt)
        has_observations = "OBSERVATIONS:" in prompt

        if role == "orchestrator":
            return json.dumps(
                {
                    "subtasks": [
                        {"id": "s1", "question": f"What do the sources say about {question}", "agent": "researcher"},
                        {"id": "s2", "question": f"What follows from that for {question}", "agent": "analyst"},
                    ]
                }
            )

        if role == "critic":
            approved = bool(citations)
            return json.dumps(
                {
                    "thought": "checked the draft for citations",
                    "tool_calls": [],
                    "final": {
                        "approved": approved,
                        "issues": [] if approved else ["the draft cites no source"],
                        "answer": "looks supported" if approved else "add at least one citation",
                    },
                }
            )

        if role == "researcher" and not has_observations:
            return json.dumps(
                {
                    "thought": "I need the corpus before I can answer",
                    "tool_calls": [{"name": "search_corpus", "arguments": {"query": question, "k": 4}}],
                    "final": None,
                }
            )

        answer = f"[{role}] {question}"
        if citations:
            answer += " — supported by " + ", ".join(f"[{c}]" for c in dict.fromkeys(citations))
        return json.dumps(
            {
                "thought": f"{role} answering from the provided context",
                "tool_calls": [],
                "final": {"answer": answer, "citations": list(dict.fromkeys(citations))},
            }
        )


def build_llm(backend: str = "fake", *, model: str = "llama3.1:8b", host: str = "http://localhost:11434") -> LLM:
    if backend == "fake":
        return FakeLLM()
    if backend == "ollama":
        return OllamaLLM(model=model, host=host)
    raise ValueError(f"unknown llm backend: {backend!r}")
