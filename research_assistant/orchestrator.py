"""The orchestrator: plan, dispatch, write, critique, revise once.

The revise step is capped at one pass on purpose. An uncapped critic loop looks
impressive in a demo and mostly produces drift; one pass is enough to catch the
common failure (a draft with no citation) and keeps the cost predictable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .agents import AnalystAgent, CriticAgent, ResearcherAgent, WriterAgent
from .agents.base import AgentResult
from .ingest import Retriever
from .llm import LLM
from .schema import ParseFailure, Subtask, parse_plan
from .tools import ToolRegistry, default_registry
from .trace import Tracer
from .vectorstore import format_context

PLANNER_SYSTEM = """ROLE: orchestrator

You break a research question into two to four subtasks and assign each one to
an agent. Available agents: researcher (finds evidence), analyst (reasons over
evidence). Reply with a single JSON object and nothing else:

{"subtasks": [{"id": "s1", "question": "...", "agent": "researcher"}]}
"""

AGENT_TYPES = {
    "researcher": ResearcherAgent,
    "analyst": AnalystAgent,
    "writer": WriterAgent,
    "critic": CriticAgent,
}


@dataclass
class RunReport:
    question: str
    answer: str
    citations: List[str] = field(default_factory=list)
    plan: List[Dict[str, str]] = field(default_factory=list)
    subtask_results: List[Dict[str, Any]] = field(default_factory=list)
    critique: Dict[str, Any] = field(default_factory=dict)
    revised: bool = False
    parse_failures: int = 0
    tool_errors: int = 0
    run_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "plan": self.plan,
            "subtasks": self.subtask_results,
            "critique": self.critique,
            "revised": self.revised,
            "parse_failures": self.parse_failures,
            "tool_errors": self.tool_errors,
        }


class Orchestrator:
    def __init__(
        self,
        llm: LLM,
        retriever: Retriever,
        *,
        tracer: Optional[Tracer] = None,
        registry: Optional[ToolRegistry] = None,
        offline: bool = True,
        max_steps: int = 3,
        context_k: int = 4,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.tracer = tracer or Tracer()
        self.registry = registry or default_registry(retriever, offline=offline)
        self.max_steps = max_steps
        self.context_k = context_k

    # -- planning -----------------------------------------------------------
    def plan(self, question: str) -> List[Subtask]:
        raw = self.llm.complete(f"QUESTION: {question}", system=PLANNER_SYSTEM)
        self.tracer.llm_call("orchestrator", question, raw, stage="plan")
        try:
            return parse_plan(raw, allowed_agents=sorted(AGENT_TYPES))
        except ParseFailure as failure:
            self.tracer.parse_failure("orchestrator", failure.reason, failure.raw)
            # A planner that cannot produce a plan still gets one subtask, so the
            # run degrades to single-agent RAG instead of failing outright.
            return [Subtask(id="s1", question=question, agent="researcher")]

    # -- full run -----------------------------------------------------------
    def run(self, question: str) -> RunReport:
        report = RunReport(question=question, answer="", run_id=self.tracer.run_id)
        self.tracer.emit("run_start", "orchestrator", question=question)

        subtasks = self.plan(question)
        report.plan = [{"id": s.id, "question": s.question, "agent": s.agent} for s in subtasks]

        findings: List[str] = []
        for subtask in subtasks:
            agent_cls = AGENT_TYPES.get(subtask.agent, ResearcherAgent)
            agent = agent_cls(self.llm, self.registry, self.tracer, max_steps=self.max_steps)
            self.tracer.message("orchestrator", subtask.agent, subtask.question, subtask_id=subtask.id)
            context = self._context_for(subtask)
            result = agent.run(subtask.question, context=context)
            report.subtask_results.append({"id": subtask.id, **result.to_dict()})
            report.parse_failures += result.parse_failures
            report.tool_errors += result.tool_errors
            if result.answer:
                findings.append(f"({subtask.id}, {subtask.agent}) {result.answer}")

        draft = self._write(question, findings)
        report.parse_failures += draft.parse_failures
        report.tool_errors += draft.tool_errors

        critique = self._critique(question, draft)
        report.parse_failures += critique.parse_failures
        report.critique = critique.raw_final or {"approved": True, "issues": []}

        answer, citations = draft.answer, draft.citations
        if not report.critique.get("approved", True):
            revised = self._write(question, findings + [f"(critic) {'; '.join(report.critique.get('issues', []))}"])
            report.revised = True
            report.parse_failures += revised.parse_failures
            if revised.answer:
                answer, citations = revised.answer, revised.citations

        report.answer = answer
        report.citations = list(dict.fromkeys(citations))
        self.tracer.emit("run_end", "orchestrator", answer_chars=len(answer), citations=report.citations)
        return report

    # -- helpers ------------------------------------------------------------
    def _context_for(self, subtask: Subtask) -> str:
        if subtask.agent == "researcher":
            # the researcher is expected to search for itself
            return ""
        hits = self.retriever.retrieve(subtask.question, k=self.context_k)
        context, _ = format_context(hits)
        return context

    def _write(self, question: str, findings: List[str]) -> AgentResult:
        writer = WriterAgent(self.llm, self.registry, self.tracer, max_steps=1)
        return writer.run(question, context="FINDINGS:\n" + "\n".join(findings))

    def _critique(self, question: str, draft: AgentResult) -> AgentResult:
        critic = CriticAgent(self.llm, self.registry, self.tracer, max_steps=1)
        payload = json.dumps({"draft": draft.answer, "citations": draft.citations}, ensure_ascii=False)
        return critic.run(question, context=f"DRAFT:\n{payload}")
