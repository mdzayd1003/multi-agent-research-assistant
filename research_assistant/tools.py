"""The tool surface agents can call.

A tool is a name, a JSON-schema-ish description of its arguments and a callable.
The registry validates required arguments before dispatch so a hallucinated
argument list becomes a tool error the agent can see, not a Python traceback.
"""

from __future__ import annotations

import ast
import json
import operator
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Sequence

from .vectorstore import ScoredChunk


class ToolError(RuntimeError):
    """Raised when a tool cannot run with the arguments it was given."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, str]
    required: Sequence[str]
    run: Callable[..., Any]

    def spec(self) -> str:
        args = ", ".join(
            f"{k}: {v}{'' if k in self.required else ' (optional)'}" for k, v in self.parameters.items()
        )
        return f"- {self.name}({args}) — {self.description}"


@dataclass
class ToolRegistry:
    tools: Dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self.tools

    def describe(self) -> str:
        return "\n".join(tool.spec() for tool in self.tools.values())

    def call(self, name: str, arguments: Dict[str, Any]) -> Any:
        tool = self.tools.get(name)
        if tool is None:
            raise ToolError(f"no such tool: {name!r}; available: {sorted(self.tools)}")
        missing = [key for key in tool.required if key not in arguments]
        if missing:
            raise ToolError(f"{name} is missing required argument(s): {missing}")
        unknown = [key for key in arguments if key not in tool.parameters]
        if unknown:
            raise ToolError(f"{name} got unexpected argument(s): {unknown}")
        return tool.run(**arguments)


# -- individual tools -------------------------------------------------------
def make_corpus_search_tool(retriever) -> Tool:
    def run(query: str, k: int = 4) -> List[Dict[str, Any]]:
        hits: List[ScoredChunk] = retriever.retrieve(query, k=int(k))
        return [
            {"citation": hit.chunk.citation, "score": round(hit.score, 4), "text": hit.chunk.text}
            for hit in hits
        ]

    return Tool(
        name="search_corpus",
        description="semantic search over the ingested corpus",
        parameters={"query": "string", "k": "integer"},
        required=["query"],
        run=run,
    )


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    raise ToolError("calculator only accepts + - * / // % ** on numbers")


def make_calculator_tool() -> Tool:
    def run(expression: str) -> float:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ToolError(f"could not parse expression: {exc}") from exc
        try:
            return _eval_node(tree)
        except ZeroDivisionError as exc:
            raise ToolError("division by zero") from exc

    return Tool(
        name="calculator",
        description="evaluate a plain arithmetic expression",
        parameters={"expression": "string"},
        required=["expression"],
        run=run,
    )


def make_clock_tool() -> Tool:
    def run() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    return Tool(
        name="utc_now",
        description="current UTC timestamp, for anything time sensitive",
        parameters={},
        required=[],
        run=run,
    )


def make_web_search_tool(*, offline: bool = False, timeout: float = 15.0) -> Tool:
    """DuckDuckGo's instant-answer endpoint. Refuses to run when offline."""

    def run(query: str, k: int = 3) -> List[Dict[str, str]]:
        if offline:
            raise ToolError("web_search is disabled in offline mode")
        url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
            {"q": query, "format": "json", "no_html": 1, "no_redirect": 1}
        )
        try:  # pragma: no cover - needs network
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:  # pragma: no cover
            raise ToolError(f"web search failed: {exc}") from exc
        return _flatten_ddg(payload, int(k))  # pragma: no cover

    return Tool(
        name="web_search",
        description="DuckDuckGo instant answers (disabled with --offline)",
        parameters={"query": "string", "k": "integer"},
        required=["query"],
        run=run,
    )


def _flatten_ddg(payload: Dict[str, Any], k: int) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    abstract = payload.get("AbstractText")
    if abstract:
        results.append({"title": payload.get("Heading", ""), "snippet": abstract, "url": payload.get("AbstractURL", "")})

    def walk(topics: Any) -> None:
        for topic in topics or []:
            if len(results) >= k:
                return
            if isinstance(topic, dict) and "Topics" in topic:
                walk(topic["Topics"])
            elif isinstance(topic, dict) and topic.get("Text"):
                results.append({"title": topic.get("Text", "")[:80], "snippet": topic.get("Text", ""), "url": topic.get("FirstURL", "")})

    walk(payload.get("RelatedTopics"))
    return results[:k]


def default_registry(retriever, *, offline: bool = True) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(make_corpus_search_tool(retriever))
    registry.register(make_calculator_tool())
    registry.register(make_clock_tool())
    registry.register(make_web_search_tool(offline=offline))
    return registry
