"""Reading track: models x 14 conditions x 160 questions against the host ollama
-> results/read/raw.jsonl. A copy of 09's run_read.py; the evidence builder is unchanged
(tests/test_run_read.py proves it against 09's raw.jsonl).

    uv run python experiments/10-fix-ablation/run_read.py [--models qwen3:8b,gpt-oss:20b]
                                                          [--limit N] [--conditions a,b]

Resumable (06's done_keys). Evidence per condition (prompts.CONDITIONS):
  closed-book  nothing (noise floor)
  oracle       the 1,000 characters of the document around the question's quote
  the rest     the dense ranking of that retrieval cell (route / chunking / prefix), whole chunks
               while they fit in 1,000 characters (the first chunk cut to 1,000 when none fits)
Texts: base / ocr / rows from ../09-messy-real/corpus/text, ocrrows from corpus/text (extract.py).
Embeddings: 09's cache in data/09-messy-real/emb-cache (same model, same texts, same vectors).
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

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
CORPUS = HERE / "corpus"
CORPUS09 = HERE.parent / "09-messy-real" / "corpus"
RESULTS = HERE / "results/read"
DATA = ROOT / "data/09-messy-real"
ROUTES = {"base": CORPUS09, "ocr": CORPUS09, "rows": CORPUS09, "ocrrows": CORPUS}
sys.path.insert(0, str(HERE))
from chunkers import Chunk  # noqa: E402
from extract import law_name  # noqa: E402
from prompts import CONDITIONS, SCHEMAS, SYSTEM, user_prompt  # noqa: E402
from retrievers import DenseIndex, EmbedCache  # noqa: E402
from run_retrieval import DEPTH, EMBED_MODEL, build_cell, is_gold, norm  # noqa: E402

logger = logging.getLogger(__name__)

MODELS = ["qwen3:8b", "gpt-oss:20b"]
THINK: dict[str, bool | str] = {"qwen3:8b": False, "gpt-oss:20b": "low"}  # see 02 README
OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 1024}
BUDGET = 1000
DRIFT_PER_CELL = 2


def load_docs() -> list[dict]:
    return json.loads((CORPUS09 / "manifest.json").read_text("utf-8"))["docs"]


def load_questions() -> list[dict]:
    path = CORPUS09.parent / "questions.jsonl"
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
            out.add((r["model"], r["condition"], r["qid"]))
    return out


def drift_sample(models: list[str], conditions: list[str], questions: list[dict], *, per_cell: int):
    ids = [q["id"] for q in questions[:per_cell]]
    return [(m, c, q) for m in models for c in conditions for q in ids]


# -- evidence -------------------------------------------------------------------------------------


def budget_evidence(ranked: list[Chunk], gold: dict, *, budget: int = BUDGET) -> tuple[str, bool]:
    """Whole chunks in rank order while they fit; the first chunk cut when none fits.
    Returns (evidence text, whether a gold chunk made it in)."""
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
    text = "\n\n".join(f"【段 {i + 1}】\n{c.text}" for i, c in enumerate(kept))
    return text, any(is_gold(c, gold) for c in kept)


def oracle_window(text: str, quote: str, *, budget: int = BUDGET) -> str | None:
    """`budget` characters of `text` centred on the quote (matched with whitespace removed)."""
    q = norm(quote)
    folded = []  # (index in text) for every non-space char, after NFKC fold of that char
    import unicodedata

    buf = []
    for i, ch in enumerate(text):
        f = unicodedata.normalize("NFKC", ch)
        if not f.strip():
            continue
        buf.append(f)
        folded.append(i)
    joined = "".join(buf)
    at = joined.find(q)
    if at < 0:
        return None
    start_c, end_c = folded[at], folded[min(at + len(q), len(folded)) - 1] + 1
    pad = max(0, (budget - (end_c - start_c)) // 2)
    lo = max(0, start_c - pad)
    return text[lo : lo + budget]


class Evidence:
    """Rankings for the retrieval-backed conditions, built once."""

    def __init__(
        self,
        docs: list[dict],
        questions: list[dict],
        *,
        conditions: list[str] | None = None,
        embed=None,
    ):
        self.docs = docs
        self.questions = questions
        self.laws = {d["sysid"]: law_name(d["caption"]) for d in docs}
        caches = sorted((DATA / "emb-cache").glob("*.npz"))
        if embed is None:
            digest = ollama.model_digest(EMBED_MODEL)
            cache_path = DATA / "emb-cache" / f"{digest.split(':')[1][:12]}.npz"
        else:  # tests: no ollama; the one cache on disk must already hold every vector
            cache_path = caches[0]
        self.cache = EmbedCache(cache_path)
        self.embed = embed or (lambda texts: ollama.embed(texts, model=EMBED_MODEL))
        self.texts = {
            route: {
                d["sysid"]: (folder / "text" / f"{d['sysid']}.{route}.txt").read_text("utf-8")
                for d in docs
            }
            for route, folder in ROUTES.items()
        }
        self.ranked: dict[str, dict[str, list[Chunk]]] = {}
        for cond, spec in CONDITIONS.items():
            if conditions is not None and cond not in conditions:
                continue
            if not isinstance(spec, tuple):
                continue
            route, chunking, prefix = spec
            chunks = build_cell(self.texts[route], docs, chunking=chunking, prefix=prefix)
            if not chunks:
                self.ranked[cond] = {q["id"]: [] for q in questions}
                continue
            index = DenseIndex.build(chunks, self.embed, self.cache)
            self.ranked[cond] = {
                q["id"]: [chunks[i] for i in index.search(q["question"], DEPTH)] for q in questions
            }
            self.cache.save()
            logger.info("ranked", extra={"ctx": {"condition": cond, "chunks": len(chunks)}})

    def for_question(self, cond: str, q: dict) -> tuple[str | None, str, bool]:
        """(evidence, label, has_gold)."""
        spec = CONDITIONS[cond]
        gold = {"doc": q["doc"], "quote": q["quote"]}
        if spec is None:
            return None, "", False
        if spec == "oracle":
            for route in ("base", "ocr"):
                win = oracle_window(self.texts[route][q["doc"]], q["quote"])
                if win is not None:
                    return win, f"《{self.laws[q['doc']]}》公告原文節錄", True
            return None, "", False
        text, has_gold = budget_evidence(self.ranked[cond][q["id"]], gold)
        return text, "檢索到的段落，可能來自不同法規", has_gold


def ask(model: str, condition: str, q: dict, ev: Evidence) -> dict:
    text, label, has_gold = ev.for_question(condition, q)
    prompt = user_prompt(q["question"], qtype=q["type"], evidence=text, label=label)
    row = {
        "model": model,
        "condition": condition,
        "qid": q["id"],
        "type": q["type"],
        "prompt_len": len(prompt),
        "evidence_chars": len(text or ""),
        "evidence_has_gold": has_gold,
    }
    try:
        r = ollama.chat(
            prompt,
            model=model,
            system=SYSTEM,
            format=SCHEMAS[q["type"]],
            think=THINK[model],
            options=OPTIONS,
        )
        row.update(
            content=r["content"],
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
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke test)")
    ap.add_argument("--conditions", default="", help="comma-separated subset of CONDITIONS")
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args(argv)
    models = args.models.split(",")
    setup_logging("rag_lab")
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.out / "raw.jsonl"
    docs = load_docs()
    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]
    byid = {q["id"]: q for q in questions}
    conditions = args.conditions.split(",") if args.conditions else list(CONDITIONS)
    ev = Evidence(docs, questions, conditions=conditions)
    done = done_keys(raw)
    todo = [k for k in grid(models, conditions, questions) if k not in done]
    logger.info(
        "grid",
        extra={"ctx": {"total": len(models) * len(conditions) * len(questions), "todo": len(todo)}},
    )
    with raw.open("a", encoding="utf-8") as f:
        for i, (m, c, qid) in enumerate(todo, 1):
            row = ask(m, c, byid[qid], ev)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(todo):
                logger.info(
                    "progress",
                    extra={"ctx": {"done": i, "todo": len(todo), "model": m, "condition": c}},
                )
    first = {
        (r["model"], r["condition"], r["qid"]): r["content"]
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
                "same": again["content"] == first.get((m, c, qid)),
            }
        )
    same = sum(p["same"] for p in probes)
    logger.info("drift", extra={"ctx": {"same": same, "n": len(probes)}})
    meta = {
        "date": date.today().isoformat(),
        "machine": platform.node(),
        "ollama": ollama.version(),
        "models": {m: ollama.model_digest(m) for m in models},
        "think": {m: THINK[m] for m in models},
        "options": OPTIONS,
        "budget": BUDGET,
        "conditions": {k: list(v) if isinstance(v, tuple) else v for k, v in CONDITIONS.items()},
        "n_questions": len(questions),
        "drift": {"same": same, "n": len(probes), "rows": probes},
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
