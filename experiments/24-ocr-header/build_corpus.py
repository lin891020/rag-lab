"""Build corpus/glm2 and corpus/glm2-flat from 23's saved glm-ocr replies; no model is asked.

    uv run python experiments/24-ocr-header/build_corpus.py

23's glm2-s100 replies are 06's clean 300 dpi pages read with 06's prompt (17/17 tables
byte-identical to 22's glm2-L0, which was byte-identical to 06). Each table's pages are rendered
and joined with a newline, as 23's read_sizes did. Also writes corpus/sources.json (sha256 of
every reply used) and corpus/headers.md (each multi-row header: before, flattened, text layer).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from ocr import glm_to_markdown, header_rows, norm  # noqa: E402

logger = logging.getLogger(__name__)

RAW23 = HERE.parent / "23-ocr-size/corpus/glm2-s100-raw"
CORPUS = HERE / "corpus"
ROUTES = {"glm2": False, "glm2-flat": True}  # route -> flatten


def pages(raw_dir: Path) -> dict[str, list[Path]]:
    """表號 -> its saved replies in page order."""
    out: dict[str, list[Path]] = {}
    for p in sorted(raw_dir.glob("*.html")):
        out.setdefault(p.stem.rsplit("-p", 1)[0], []).append(p)
    return out


def build(raw_dir: Path) -> dict[str, dict[str, str]]:
    """route -> 表號 -> markdown."""
    by_table = pages(raw_dir)
    return {
        route: {
            tid: "\n".join(glm_to_markdown(p.read_text("utf-8"), flatten=flat) for p in ps)
            for tid, ps in by_table.items()
        }
        for route, flat in ROUTES.items()
    }


def sources(raw_dir: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(raw_dir.glob("*.html"))
    }


def _table_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.startswith("|") and not ln.startswith("|---")]


def _cells(line: str) -> list[str]:
    return [norm(c) for c in line.strip().strip("|").split("|")]


def headers(raw_dir: Path, built: dict[str, dict[str, str]], clean_dir: Path) -> list[dict]:
    """One row per table whose first page has a <thead> of more than one row."""
    rows = []
    for tid, ps in pages(raw_dir).items():
        n = header_rows(ps[0].read_text("utf-8"))
        if n < 2:
            continue
        flat = _table_lines(built["glm2-flat"][tid])[0]
        clean = _table_lines((clean_dir / f"{tid}.txt").read_text("utf-8"))[0]
        rows.append(
            {
                "table": tid,
                "rows": n,
                "before": _table_lines(built["glm2"][tid])[:n],
                "flat": flat,
                "clean": clean,
                "same_as_clean": _cells(flat) == _cells(clean),
            }
        )
    return rows


def render_headers(rows: list[dict]) -> str:
    out = [
        "# 多層表頭：攤平前、攤平後、文字層",
        "",
        "「相同」是逐格比，NFKC、去空白之後。",
        "",
        "| 表 | 表頭列數 | 攤平後跟文字層相同 |",
        "|---|---|---|",
    ]
    out += [
        f"| {r['table']} | {r['rows']} | {'是' if r['same_as_clean'] else '否'} |" for r in rows
    ]
    for r in rows:
        out += ["", f"## {r['table']}", "", "攤平前（23 的 markdown，前幾列）：", "", "```"]
        out += r["before"]
        out += ["```", "", "攤平後：", "", "```", r["flat"], "```", "", "文字層：", "", "```"]
        out += [r["clean"], "```"]
    return "\n".join(out) + "\n"


def main() -> int:
    setup_logging("rag_lab")
    built = build(RAW23)
    for route, tables in built.items():
        d = CORPUS / route
        d.mkdir(parents=True, exist_ok=True)
        for tid, text in tables.items():
            (d / f"{tid}.txt").write_text(text, encoding="utf-8")
    (CORPUS / "sources.json").write_text(
        json.dumps(sources(RAW23), ensure_ascii=False, indent=1), "utf-8"
    )
    rows = headers(RAW23, built, CORPUS / "clean")
    (CORPUS / "headers.md").write_text(render_headers(rows), encoding="utf-8")
    changed = sorted(t for t in built["glm2"] if built["glm2"][t] != built["glm2-flat"][t])
    logger.info(
        "built",
        extra={
            "ctx": {
                "tables": len(built["glm2"]),
                "changed": changed,
                "same_as_clean": sum(r["same_as_clean"] for r in rows),
            }
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
