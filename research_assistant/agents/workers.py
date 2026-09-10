"""The four worker agents. Only the system prompt differs between them."""

from __future__ import annotations

from .base import Agent


class ResearcherAgent(Agent):
    role = "researcher"
    allowed_tools = ["search_corpus", "web_search", "utc_now"]
    instructions = (
        "Gather evidence before you answer. Search the corpus first; only reach for the web "
        "when the corpus clearly does not cover the question. Quote the citation tag of every "
        "chunk you rely on, in the form source.md#3. Never state a fact you did not retrieve."
    )


class AnalystAgent(Agent):
    role = "analyst"
    allowed_tools = ["calculator", "search_corpus"]
    instructions = (
        "You are given evidence collected by the researcher. Draw out what follows from it: "
        "comparisons, trade-offs, numbers. Use the calculator for arithmetic rather than doing "
        "it in your head. Say plainly when the evidence does not support a conclusion."
    )


class WriterAgent(Agent):
    role = "writer"
    allowed_tools: list[str] = []
    instructions = (
        "Turn the collected findings into one coherent answer for the reader. Keep every "
        "citation tag that supports a claim. Do not introduce facts that are not in the findings."
    )


class CriticAgent(Agent):
    role = "critic"
    allowed_tools = ["search_corpus"]
    instructions = (
        "Review the draft against the evidence. Check three things: every factual claim carries "
        "a citation, no claim contradicts the evidence, and the answer actually addresses the "
        "question. Reply with a final object shaped "
        '{"approved": bool, "issues": [string], "answer": string}.'
    )
