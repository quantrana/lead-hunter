from __future__ import annotations

from pathlib import Path

from .models import ErrorEvent, TraceEvent
from .storage import Storage
from .utils import append_jsonl


class TraceLogger:
    def __init__(self, storage: Storage, run_log_path: str | Path, errors_path: str | Path) -> None:
        self.storage = storage
        self.run_log_path = Path(run_log_path)
        self.errors_path = Path(errors_path)

    def trace(self, event: TraceEvent) -> None:
        self.storage.save_trace(event)
        append_jsonl(self.run_log_path, event.model_dump())

    def error(self, event: ErrorEvent) -> None:
        self.storage.save_error(event)
        append_jsonl(self.errors_path, event.model_dump())
