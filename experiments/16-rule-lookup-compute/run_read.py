"""Reading track: 6 conditions × 2 models × 160 questions against the host ollama
-> results/read/raw.jsonl (resumable, 10's done_keys). 10's skeleton with 16's evidence:

  L0         no evidence
  L1-fixed   one shared corpus text (the law + 22 clean tables as kv rows + 87 raw tables as
             their rows) cut fixed-1000/200, dense top-k within the budget
  L1-struct  the same text as one chunk per article (01's article_chunks) + one chunk per clean
             row (kv_row_chunks) + 01's raw row chunks for the other tables, dense within budget
  L1-oracle  the gold rule excerpt + the gold rows (upper bound)
  L2-*       same evidence; for number questions `value` is replaced by calc.evaluate(expression)

Budget 1,500 characters (two table rows and a rule must fit), first chunk always read
(budget_evidence as in 10). Gold is judged by chunk labels: article numbers and `<table>|<row key>`
(not by text containment: the fixed chunks hold the kv line verbatim but a window may cut it).
After the first 3 questions of a (model, condition) cell the log carries the seconds per call.

    uv run python experiments/16-rule-lookup-compute/run_read.py [--models a,b] [--limit N]
        [--conditions L0,L1-struct] [--out results/read] [--dry]   (--dry: build, embed nothing)
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from collections.abc import Iterator
from datetime import date
from pathlib import Path

from rag_lab import ollama
from rag_lab.log import setup_logging
from rag_lab.pdf import extract_text

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
ONE = HERE.parent / "01-chunking"
RESULTS = HERE / "results/read"
QUESTIONS = HERE / "questions.jsonl"
DATA = ROOT / "data/01-chunking"  # 01's embedding cache (same model, vectors keyed by text)
sys.path.insert(0, str(HERE))
from calc import BadExpression, evaluate  # noqa: E402
from chunkers import (  # noqa: E402
    Chunk,
    article_chunks,
    fixed_chunks,
    kv_row_chunks,
    kv_row_line,
    parse_articles,
    table_row_chunks,
)
from prompts import CONDITIONS, SCHEMA, SYSTEM, user_prompt  # noqa: E402
from retrievers import DenseIndex, EmbedCache  # noqa: E402
from rules import TABLES, row_key  # noqa: E402

logger = logging.getLogger(__name__)

MODELS = ["qwen3:8b", "gpt-oss:20b"]
THINK: dict[str, bool | str] = {"qwen3:8b": False, "gpt-oss:20b": "low"}  # see 02 README
OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 1024}
EMBED_MODEL = "bge-m3"
BUDGET = 1500
DEPTH = 50
DRIFT_PER_CELL = 2
FIXED = {"size": 1000, "overlap": 200}
SEPS = ["\n\n", "\n", ". ", " ", ""]  # 01's splitter separators
EXCERPT_CHARS = (
    600  # oracle: this much of the gold article around its first mention of the gold table
)
LABELS = {
    "fixed": "檢索到的段落，可能來自不同條文與附表",
    "struct": "檢索到的段落，可能來自不同條文與附表",
    "oracle": "規定與附表節錄",
}


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text("utf-8").splitlines() if ln.strip()]


def grid(models: list[str], conditions: list[str], questions: list[dict]) -> Iterator[tuple]:
    for m in models:
        for c in conditions:
            for q in questions:
                yield (m, c, q["id"])


def done_keys(raw: Path) -> set[tuple[str, str, str]]:
    if not raw.exists():
        return set()
    out = set()
    for ln in raw.read_text("utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            if not r.get("error"):  # a failed call is redone on the next run
                out.add((r["model"], r["condition"], r["qid"]))
    return out


def drift_sample(models: list[str], conditions: list[str], questions: list[dict], *, per_cell: int):
    ids = [q["id"] for q in questions[:per_cell]]
    return [(m, c, q) for m in models for c in conditions for q in ids]


# -- corpus -----------------------------------------------------------------------------------


def load_corpus() -> tuple[str, dict[str, str]]:
    """(law text, raw table texts of the 87 tables that have no clean version)."""
    law = extract_text(ONE / "corpus/law.pdf")
    raw = {
        p.stem: p.read_text("utf-8")
        for p in sorted((ONE / "corpus/tables").glob("*.txt"))
        if p.stem not in TABLES
    }
    return law, raw


def corpus_text(law: str, raw: dict[str, str]) -> tuple[str, list[tuple[int, int, str]]]:
    """One text for the fixed cut: law, then each clean table (title + kv rows), then the raw
    tables; spans label articles by number, clean rows by `<table>|<key>`, raw tables by id."""
    arts = parse_articles(law)
    spans = [(a.start, a.end, str(a.num)) for a in arts]
    parts = [law]
    pos = len(law) + 2
    for tid, t in TABLES.items():
        block = [t["title"]]
        pos_in = pos + len(t["title"]) + 1
        for r in t["rows"]:
            line = kv_row_line(t["names"], r)
            spans.append((pos_in, pos_in + len(line), f"{tid}|{row_key(tid, r)}"))
            block.append(line)
            pos_in += len(line) + 1
        if t.get("notes"):
            spans.append((pos_in, pos_in + len(t["notes"]), f"{tid}|註"))
            block.append(t["notes"])
        text = "\n".join(block)
        parts.append(text)
        pos += len(text) + 2
    for tid, text in raw.items():
        spans.append((pos, pos + len(text), tid))
        parts.append(text)
        pos += len(text) + 2
    return "\n\n".join(parts), spans


def struct_chunks(law: str, raw: dict[str, str]) -> list[Chunk]:
    return (
        article_chunks(parse_articles(law), max_len=1000)
        + kv_row_chunks(TABLES, row_key)
        + table_row_chunks(raw)
    )


def fixed_corpus_chunks(law: str, raw: dict[str, str]) -> list[Chunk]:
    text, spans = corpus_text(law, raw)
    return fixed_chunks(
        text, size=FIXED["size"], overlap=FIXED["overlap"], separators=SEPS, spans=spans
    )


# -- evidence ---------------------------------------------------------------------------------


def budget_evidence(
    ranked: list[Chunk], gold_labels: set[str], *, budget: int = BUDGET
) -> tuple[str, dict]:
    """Whole chunks in rank order while they fit; the first chunk cut when none fits (10).
    Returns (text, {rows_hit, rows_total, article_hit}) judged on labels."""
    kept: list[Chunk] = []
    used = 0
    for c in ranked:
        if used + len(c.text) > budget:
            break
        used += len(c.text)
        kept.append(c)
    if not kept and ranked:
        c = ranked[0]
        kept = [Chunk(c.text[:budget], list(c.labels), c.start, c.end, dict(c.meta))]
    labels = {lab for c in kept for lab in c.labels}
    text = "\n\n".join(f"【段 {i + 1}】\n{c.text}" for i, c in enumerate(kept))
    return text, labels


def gold_labels(q: dict) -> tuple[set[str], set[str]]:
    rows = {f"{t}|{k}" for t, k in q["gold"]["rows"]}
    arts = set(q["gold"]["articles"])
    return rows, arts


def rule_excerpt(body: str, needles: list[str], *, chars: int = EXCERPT_CHARS) -> str:
    """`chars` characters of the article around the first needle found (the whole body if short)."""
    if len(body) <= chars:
        return body
    at = min((body.find(n) for n in needles if n in body), default=0)
    lo = max(0, at - chars // 3)
    return body[lo : lo + chars]


class Evidence:
    def __init__(
        self, questions: list[dict], *, conditions: list[str], embed=None, build: bool = True
    ):
        law, raw = load_corpus()
        self.articles = {str(a.num): a for a in parse_articles(law)}
        self.ranked: dict[str, dict[str, list[Chunk]]] = {}
        sources = {
            CONDITIONS[c]["evidence"]
            for c in conditions
            if CONDITIONS[c]["evidence"] in ("fixed", "struct")
        }
        if not sources:
            return
        if embed is None:
            digest = ollama.model_digest(EMBED_MODEL)
            cache_path = DATA / "emb-cache" / f"{digest.split(':')[1][:12]}.npz"
        else:
            cache_path = sorted((DATA / "emb-cache").glob("*.npz"))[0]
        self.cache = EmbedCache(cache_path)
        self.embed = embed or (lambda texts: ollama.embed(texts, model=EMBED_MODEL))
        self.chunks = {"fixed": fixed_corpus_chunks(law, raw), "struct": struct_chunks(law, raw)}
        for src in sources:
            chunks = self.chunks[src]
            logger.info(
                "chunks",
                extra={
                    "ctx": {
                        "source": src,
                        "n": len(chunks),
                        "mean_len": round(sum(len(c.text) for c in chunks) / len(chunks)),
                    }
                },
            )
            if not build:
                continue
            index = DenseIndex.build(chunks, self.embed, self.cache)
            self.ranked[src] = {
                q["id"]: [chunks[i] for i in index.search(q["question"], DEPTH)] for q in questions
            }
            self.cache.save()
            logger.info("ranked", extra={"ctx": {"source": src, "questions": len(questions)}})

    def oracle(self, q: dict) -> str:
        parts = []
        needles = [t for t, _ in q["gold"]["rows"]]
        for num in q["gold"]["articles"]:
            a = self.articles.get(num)
            if a:
                parts.append(f"【第 {num} 條】\n{rule_excerpt(a.body, needles)}")
        for tid, key in q["gold"]["rows"]:
            t = TABLES[tid]
            row = next((r for r in t["rows"] if row_key(tid, r) == key), None)
            if key == "註":
                parts.append(f"【{t['title']}】\n{t.get('notes', '')}")
            elif row is not None:
                parts.append(f"【{t['title']}】\n{kv_row_line(t['names'], row)}")
            else:  # an off-table key (4 hp): the table's own rows around it
                parts.append(
                    f"【{t['title']}】\n"
                    + "\n".join(kv_row_line(t["names"], r) for r in t["rows"][:6])
                )
        return "\n\n".join(parts)

    def for_question(self, cond: str, q: dict) -> tuple[str | None, str, dict]:
        src = CONDITIONS[cond]["evidence"]
        rows, arts = gold_labels(q)
        if src is None:
            return None, "", {"rows_hit": 0, "rows_total": len(rows), "article_hit": False}
        if src == "oracle":
            return (
                self.oracle(q),
                LABELS["oracle"],
                {"rows_hit": len(rows), "rows_total": len(rows), "article_hit": True},
            )
        text, labels = budget_evidence(self.ranked[src][q["id"]], rows)
        return (
            text,
            LABELS[src],
            {
                "rows_hit": len(rows & labels),
                "rows_total": len(rows),
                "article_hit": bool(arts & labels),
            },
        )


# -- one call ---------------------------------------------------------------------------------


def apply_tool(content: str, q: dict) -> tuple[str, bool]:
    """L2: for number questions replace `value` by the evaluated expression. Returns (content,
    expr_ok). Ladder questions keep the model's own value; found=false is left alone."""
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return content, False
    if not isinstance(obj, dict) or obj.get("found") is not True:
        return content, False
    try:
        val = evaluate(str(obj.get("expression", "")))
    except BadExpression:
        return content, False
    if q["expect"]["kind"] != "ladder":
        obj["value"] = val
    return json.dumps(obj, ensure_ascii=False), True


def ask(model: str, condition: str, q: dict, ev: Evidence) -> dict:
    text, label, hit = ev.for_question(condition, q)
    tool = CONDITIONS[condition]["tool"]
    prompt = user_prompt(q["question"], evidence=text, label=label, tool=tool)
    row = {
        "model": model,
        "condition": condition,
        "qid": q["id"],
        "chain": q["chain"],
        "prompt_len": len(prompt),
        "evidence_chars": len(text or ""),
        "evidence_text": text or "",
        **hit,
    }
    try:
        r = ollama.chat(
            prompt, model=model, system=SYSTEM, format=SCHEMA, think=THINK[model], options=OPTIONS
        )
        content = r["content"]
        expr_ok = None
        if tool:
            content, expr_ok = apply_tool(content, q)
        row.update(
            content=content,
            raw_content=r["content"],
            expr_ok=expr_ok,
            thinking_len=len(r["thinking"]),
            eval_count=r["eval_count"],
            duration=r["duration"],
            done_reason=r["done_reason"],
            error="",
        )
    except (OSError, KeyError, ValueError) as exc:
        logger.exception(
            "call failed", extra={"ctx": {"model": model, "condition": condition, "qid": q["id"]}}
        )
        row.update(
            content="",
            raw_content="",
            expr_ok=None,
            thinking_len=0,
            eval_count=0,
            duration=0.0,
            done_reason="error",
            error=repr(exc),
        )
    return row


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conditions", default="")
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--dry", action="store_true", help="build chunks, report sizes, embed nothing")
    args = ap.parse_args(argv)
    models = args.models.split(",")
    setup_logging("rag_lab")
    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]
    byid = {q["id"]: q for q in questions}
    conditions = args.conditions.split(",") if args.conditions else list(CONDITIONS)
    if args.dry:
        Evidence(questions, conditions=conditions, embed=lambda t: [], build=False)
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.out / "raw.jsonl"
    ev = Evidence(questions, conditions=conditions)
    done = done_keys(raw)
    todo = [k for k in grid(models, conditions, questions) if k not in done]
    logger.info(
        "grid",
        extra={"ctx": {"total": len(models) * len(conditions) * len(questions), "todo": len(todo)}},
    )
    cell_secs: dict[tuple[str, str], list[float]] = {}
    with raw.open("a", encoding="utf-8") as f:
        for i, (m, c, qid) in enumerate(todo, 1):
            row = ask(m, c, byid[qid], ev)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            secs = cell_secs.setdefault((m, c), [])
            secs.append(row["duration"])
            if len(secs) == 3:
                mean = sum(secs) / 3
                logger.info(
                    "cell timing",
                    extra={
                        "ctx": {
                            "model": m,
                            "condition": c,
                            "sec_per_call": round(mean, 1),
                            "cell_hours": round(mean * len(questions) / 3600, 2),
                        }
                    },
                )
            if i % 20 == 0 or i == len(todo):
                logger.info(
                    "progress",
                    extra={"ctx": {"done": i, "todo": len(todo), "model": m, "condition": c}},
                )
    first = {
        (r["model"], r["condition"], r["qid"]): r["raw_content"]
        for r in map(json.loads, raw.read_text("utf-8").splitlines())
        if r
    }
    probes = []
    for m, c, qid in drift_sample(models, conditions, questions, per_cell=DRIFT_PER_CELL):
        again = ask(m, c, byid[qid], ev)
        probes.append(
            {
                "model": m,
                "condition": c,
                "qid": qid,
                "same": again["raw_content"] == first.get((m, c, qid)),
            }
        )
    same = sum(p["same"] for p in probes)
    logger.info("drift", extra={"ctx": {"same": same, "n": len(probes)}})
    meta = {
        "date": date.today().isoformat(),
        "machine": platform.node(),
        "ollama": ollama.version(),
        "models": {m: ollama.model_digest(m) for m in models},
        "embed_model": {EMBED_MODEL: ollama.model_digest(EMBED_MODEL)},
        "think": {m: THINK[m] for m in models},
        "options": OPTIONS,
        "budget": BUDGET,
        "fixed": FIXED,
        "conditions": conditions,
        "n_questions": len(questions),
        "drift": {"same": same, "n": len(probes), "rows": probes},
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
