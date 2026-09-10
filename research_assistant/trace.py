"""JSONL tracing for every LLM call, tool call and agent-to-agent message.

The trace is the point of the project as much as the answer is: without it you
cannot tell whether the writer agent actually used what the researcher found.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TraceEvent:
    kind: str
    actor: str
    payload: Dict[str, Any]
    run_id: str
    seq: int
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "seq": self.seq,
                "ts": round(self.timestamp, 6),
                "kind": self.kind,
                "actor": self.actor,
                **self.payload,
            },
            ensure_ascii=False,
            sort_keys=False,
        )


class Tracer:
    """Append-only JSONL writer. Thread-safe; keeps events in memory too."""

    def __init__(self, path: Optional[str | Path] = None, run_id: Optional[str] = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.path = Path(path) if path else None
        self.events: List[TraceEvent] = []
        self._lock = threading.Lock()
        self._seq = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, kind: str, actor: str, **payload: Any) -> TraceEvent:
        with self._lock:
            self._seq += 1
            event = TraceEvent(kind=kind, actor=actor, payload=payload, run_id=self.run_id, seq=self._seq)
            self.events.append(event)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(event.to_json() + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        return event

    # convenience wrappers, so call sites read as documentation
    def llm_call(self, actor: str, prompt: str, response: str, **extra: Any) -> TraceEvent:
        return self.emit(
            "llm_call",
            actor,
            prompt_chars=len(prompt),
            prompt_preview=prompt[-400:],
            response=response,
            **extra,
        )

    def tool_call(self, actor: str, name: str, arguments: Dict[str, Any], result: Any, ok: bool = True) -> TraceEvent:
        return self.emit("tool_call", actor, tool=name, arguments=arguments, ok=ok, result=_truncate(result))

    def message(self, sender: str, recipient: str, content: Any, **extra: Any) -> TraceEvent:
        return self.emit("message", sender, recipient=recipient, content=_truncate(content), **extra)

    def parse_failure(self, actor: str, reason: str, raw: str) -> TraceEvent:
        return self.emit("parse_failure", actor, reason=reason, raw=raw[:1000])

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for event in self.events:
            out[event.kind] = out.get(event.kind, 0) + 1
        return out


def _truncate(value: Any, limit: int = 2000) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"... [{len(value) - limit} more chars]"
    return value


def read_trace(path: str | Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
