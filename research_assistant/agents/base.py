"""The agent loop shared by researcher, analyst, writer and critic.

One loop, one contract, four system prompts. The loop is deliberately small:
think -> maybe call tools -> observe -> think again, up to ``max_steps``, and a
reply that breaks the contract ends the agent rather than triggering a retry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..llm import LLM
from ..schema import AGENT_CONTRACT, AgentReply, ParseFailure, parse_agent_reply
from ..tools import ToolError, ToolRegistry
from ..trace import Tracer


@dataclass
class AgentResult:
    agent: str
    answer: str
    citations: List[str] = field(default_factory=list)
    steps: int = 0
    parse_failures: int = 0
    tool_errors: int = 0
    ok: bool = True
    raw_final: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "answer": self.answer,
            "citations": self.citations,
            "steps": self.steps,
            "parse_failures": self.parse_failures,
            "tool_errors": self.tool_errors,
            "ok": self.ok,
        }


class Agent:
    """Base class. Subclasses supply ``role`` and ``instructions``."""

    role: str = "agent"
    instructions: str = "Answer the question."
    allowed_tools: Optional[List[str]] = None

    def __init__(self, llm: LLM, registry: ToolRegistry, tracer: Tracer, *, max_steps: int = 3) -> None:
        self.llm = llm
        self.registry = registry
        self.tracer = tracer
        self.max_steps = max_steps

    # -- prompt -------------------------------------------------------------
    def system_prompt(self) -> str:
        tools = self.registry.describe()
        if self.allowed_tools is not None:
            tools = "\n".join(
                line for line in tools.splitlines() if any(f"- {t}(" in line for t in self.allowed_tools)
            )
        return (
            f"ROLE: {self.role}\n\n{self.instructions}\n\n"
            f"TOOLS AVAILABLE:\n{tools or '- (none)'}\n\n{AGENT_CONTRACT}"
        )

    def build_prompt(self, question: str, context: str, observations: List[str]) -> str:
        parts = [f"QUESTION: {question}"]
        if context:
            parts.append(f"CONTEXT:\n{context}")
        if observations:
            parts.append("OBSERVATIONS:\n" + "\n".join(observations))
        return "\n\n".join(parts)

    # -- loop ---------------------------------------------------------------
    def run(self, question: str, context: str = "") -> AgentResult:
        result = AgentResult(agent=self.role, answer="")
        observations: List[str] = []
        system = self.system_prompt()

        for step in range(1, self.max_steps + 1):
            result.steps = step
            prompt = self.build_prompt(question, context, observations)
            raw = self.llm.complete(prompt, system=system)
            self.tracer.llm_call(self.role, prompt, raw, step=step)

            try:
                reply: AgentReply = parse_agent_reply(raw)
            except ParseFailure as failure:
                result.parse_failures += 1
                result.ok = False
                self.tracer.parse_failure(self.role, failure.reason, failure.raw)
                result.answer = f"{self.role} produced an unparseable reply: {failure.reason}"
                return result

            if reply.is_final:
                final = reply.final or {}
                result.raw_final = final
                result.answer = str(final.get("answer", "")).strip()
                citations = final.get("citations") or []
                result.citations = [str(c) for c in citations if isinstance(c, (str, int))]
                result.ok = bool(result.answer)
                self.tracer.message(self.role, "orchestrator", result.answer, citations=result.citations)
                return result

            for call in reply.tool_calls:
                try:
                    output = self.registry.call(call.name, call.arguments)
                    self.tracer.tool_call(self.role, call.name, call.arguments, output)
                    observations.append(f"{call.name} -> {_render(output)}")
                except ToolError as exc:
                    result.tool_errors += 1
                    self.tracer.tool_call(self.role, call.name, call.arguments, str(exc), ok=False)
                    observations.append(f"{call.name} -> ERROR: {exc}")

        result.ok = False
        result.answer = f"{self.role} hit the {self.max_steps}-step budget without answering"
        self.tracer.message(self.role, "orchestrator", result.answer, exhausted=True)
        return result


def _render(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        rendered = []
        for item in output:
            if isinstance(item, dict) and "citation" in item:
                rendered.append(f"[{item['citation']}] {item.get('text', '')}")
            else:
                rendered.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(rendered)
    return json.dumps(output, ensure_ascii=False, default=str)
