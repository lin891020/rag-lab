"""Experiments deliberately do not share code, so two of them may both have run.py, report.py…
When pytest collects more than one experiment in one process, the first import wins and the
other experiment's tests load the wrong module. Before each test module is imported, put its own
experiment directory first on sys.path and drop the cached experiment-local modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
EXPERIMENTS = ROOT / "experiments"
LOCAL = {
    "run",
    "report",
    "score",
    "prompts",
    "chunkers",
    "metrics",
    "retrievers",
    "fetch_corpus",
    "judge",
    "tables",
    "select_rows",
    "build_corpus",
    "degrade",
    "labels",
    "doc_health",
    "run_text",
    "run_table",
    "structure",
    "extract",
    "attribute",
    "run_retrieval",
    "run_read",
    "ocr",
    "render",
}


def pytest_pycollect_makemodule(module_path, parent):  # noqa: ANN001
    p = Path(module_path)
    if EXPERIMENTS in p.parents:
        exp = p.relative_to(EXPERIMENTS).parts[0]
        exp_dir = str(EXPERIMENTS / exp)
        sys.path = [exp_dir] + [x for x in sys.path if x != exp_dir]
        for name in LOCAL:
            sys.modules.pop(name, None)
    return None
