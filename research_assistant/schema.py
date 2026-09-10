"""The JSON contract every agent speaks.

One contract for all four worker agents keeps the orchestrator simple: an agent
either asks for tools, or it finishes. A malformed reply is a ``ParseFailure``
and is recorded as such -- it is never silently retried, because a model that
cannot hold the format is exactly the thing worth measuring.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class ParseFailure(ValueError):
    """Raised when a model reply does not satisfy the contract."""

    def __init__(self, reason: str, raw: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.raw = raw


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentReply:
    thought: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    final: Optional[Dict[str, Any]] = None

    @property
    def is_final(self) -> bool:
        return self.final is not None


def extract_json_object(text: str) -> Dict[str, Any]:
    """Pull the first JSON object out of a model reply.

    Models wrap JSON in prose or code fences often enough that stripping those
    is part of the contract rather than a workaround. What is *not* tolerated is
    JSON that parses but does not match the schema.
    """
    if not text or not text.strip():
        raise ParseFailure("empty reply", text)

    candidates: List[str] = []
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)

    for candidate in candidates:
        candidate = candidate.strip()
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            continue
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ParseFailure("no JSON object found in reply", text)


def parse_agent_reply(text: str) -> AgentReply:
    payload = extract_json_object(text)

    thought = payload.get("thought", "")
    if not isinstance(thought, str):
        raise ParseFailure("'thought' must be a string", text)

    raw_calls = payload.get("tool_calls", []) or []
    if not isinstance(raw_calls, list):
        raise ParseFailure("'tool_calls' must be a list", text)

    calls: List[ToolCall] = []
    for item in raw_calls:
        if not isinstance(item, dict) or "name" not in item:
            raise ParseFailure("each tool call needs a 'name'", text)
        arguments = item.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ParseFailure("tool call 'arguments' must be an object", text)
        calls.append(ToolCall(name=str(item["name"]), arguments=arguments))

    final = payload.get("final")
    if final is not None and not isinstance(final, dict):
        raise ParseFailure("'final' must be an object or null", text)
    if final is None and not calls:
        raise ParseFailure("reply has neither tool calls nor a final answer", text)

    return AgentReply(thought=thought, tool_calls=calls, final=final)


@dataclass
class Subtask:
    id: str
    question: str
    agent: str = "researcher"


def parse_plan(text: str, *, allowed_agents: Optional[List[str]] = None) -> List[Subtask]:
    payload = extract_json_object(text)
    raw = payload.get("subtasks")
    if not isinstance(raw, list) or not raw:
        raise ParseFailure("'subtasks' must be a non-empty list", text)

    subtasks: List[Subtask] = []
    for position, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ParseFailure("each subtask must be an object", text)
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ParseFailure("each subtask needs a non-empty 'question'", text)
        agent = str(item.get("agent", "researcher"))
        if allowed_agents is not None and agent not in allowed_agents:
            raise ParseFailure(f"unknown agent {agent!r} in plan", text)
        subtasks.append(
            Subtask(id=str(item.get("id", f"s{position}")), question=question.strip(), agent=agent)
        )
    return subtasks


AGENT_CONTRACT = """You must reply with a single JSON object and nothing else.

{
  "thought": "one or two sentences of reasoning",
  "tool_calls": [{"name": "<tool>", "arguments": {...}}],
  "final": null
}

Set "tool_calls" to [] and "final" to an object once you can answer:

{
  "thought": "...",
  "tool_calls": [],
  "final": {"answer": "...", "citations": ["source.md#0"]}
}
"""
