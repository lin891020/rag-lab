"""Project logging: one call at the entry point, named loggers everywhere else.

    from <pkg>.log import setup_logging
    setup_logging("<app>")                  # entry point only
    logger = logging.getLogger(__name__)    # every module

stdout gets a human line at INFO; logs/<app>.jsonl gets one JSON object per
record at DEBUG, with the traceback in "exc" when logger.exception() is used.
Pass context as extra={"ctx": {...}}; it lands in the JSON as "ctx".
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_HUMAN = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        row: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        ctx = getattr(record, "ctx", None)
        if ctx:
            row["ctx"] = ctx
        if record.exc_info:
            row["exc"] = self.formatException(record.exc_info)
        return json.dumps(row, ensure_ascii=False)


def setup_logging(app: str, *, level: str = "INFO", log_dir: Path | str = "logs") -> Path:
    """Attach a stdout handler and a JSONL file handler to the root logger. Safe to call twice."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / f"{app}.jsonl"
    if getattr(root, "_setup_logging_done", None) == str(out):
        return out
    # Remove only the handlers this function installed earlier. Anything else on the
    # root logger (pytest's caplog, a host application's handler) stays attached.
    for h in getattr(root, "_setup_logging_handlers", []):
        root.removeHandler(h)
    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(getattr(logging, level.upper()))
    stream.setFormatter(logging.Formatter(_HUMAN, datefmt="%H:%M:%S"))
    file = logging.FileHandler(out, encoding="utf-8")
    file.setLevel(logging.DEBUG)
    file.setFormatter(JsonFormatter())
    root.addHandler(stream)
    root.addHandler(file)
    root._setup_logging_handlers = [stream, file]  # type: ignore[attr-defined]
    root._setup_logging_done = str(out)  # type: ignore[attr-defined]
    return out
