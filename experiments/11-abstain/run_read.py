"""Abstention track: models x prompts x evidence kinds x questions -> results/read/raw.jsonl

    uv run python experiments/11-abstain/run_read.py [--models ...] [--prompts P0,P1]
                                                     [--evidence gold,same-doc,...] [--limit N]

A copy of 10's run_read.py. The evidence builder for `real` and `gold` is 09's; `same-doc`,
`other-doc` and the leak check are new (prompts.py says what each kind is). Each row keeps the
evidence text so score.py can check the model's quote against it without rebuilding anything.
Resumable (06's done_keys). Texts from ../09-messy-real/corpus/text and ../10-fix-ablation/corpus/
text; embeddings from 09's cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import sys
import unicodedata
from datetime import date
from pathlib import Path

from rag_lab import ollama
from rag_lab.log import setup_logging

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
CORPUS09 = HERE.parent / "09-messy-real" / "corpus"
CORPUS10 = HERE.parent / "10-fix-ablation" / "corpus"
RESULTS = HERE / "results/read"
DATA = ROOT / "data/09-messy-real"
ROUTES = {"base": CORPUS09, "ocr": CORPUS09, "rows": CORPUS09, "ocrrows": CORPUS10}
sys.path.insert(0, str(HERE))
from chunkers import Chunk, int_to_cn  # noqa: E402
from extract import law_name  # noqa: E402
from prompts import EVIDENCE, PROMPTS, REAL_CELL, SCHEMAS, SYSTEM, user_prompt  # noqa: E402
from retrievers import DenseIndex, EmbedCache  # noqa: E402
from run_retrieval import DEPTH, EMBED_MODEL, build_cell, is_gold, norm  # noqa: E402

logger = logging.getLogger(__name__)

MODELS = ["qwen3:8b", "gpt-oss:20b"]
THINK: dict[str, bool | str] = {"qwen3:8b": False, "gpt-oss:20b": "low"}  # see 02 README
OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 1024}
BUDGET = 1000
SKIP = 1500  # same-doc: how far past the quote the distractor window starts
DRIFT_PER_CELL = 2


def load_docs() -> list[dict]:
    return json.loads((CORPUS09 / "manifest.json").read_text("utf-8"))["docs"]


def load_questions() -> list[dict]:
    path = CORPUS09.parent / "questions.jsonl"
    return [json.loads(ln) for ln in path.read_text("utf-8").splitlines() if ln.strip()]


def grid(models: list[str], prompts: list[str], kinds: list[str], questions: list[dict]):
    for m in models:
        for p in prompts:
            for k in kinds:
                for q in questions:
                    yield (m, p, k, q["id"])


def done_keys(raw: Path) -> set[tuple[str, str, str, str]]:
    if not raw.exists():
        return set()
    out = set()
    for ln in raw.read_text("utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            out.add((r["model"], r["prompt"], r["evidence"], r["qid"]))
    return out


def drift_sample(models, prompts, kinds, questions, *, per_cell: int) -> list[tuple]:
    ids = [q["id"] for q in questions[:per_cell]]
    return [(m, p, k, q) for m in models for p in prompts for k in kinds for q in ids]


# -- evidence -------------------------------------------------------------------------------------


def _locate(text: str, quote: str) -> tuple[int, int] | None:
    """(start, end) of the quote in `text`, matched with whitespace removed after NFKC."""
    q = norm(quote)
    buf, folded = [], []
    for i, ch in enumerate(text):
        f = unicodedata.normalize("NFKC", ch)
        if not f.strip():
            continue
        buf.append(f)
        folded.append(i)
    at = "".join(buf).find(q)
    if at < 0:
        return None
    return folded[at], folded[min(at + len(q), len(folded)) - 1] + 1


def oracle_window(text: str, quote: str, *, budget: int = BUDGET) -> str | None:
    """`budget` characters of `text` centred on the quote (09's)."""
    loc = _locate(text, quote)
    if loc is None:
        return None
    start_c, end_c = loc
    pad = max(0, (budget - (end_c - start_c)) // 2)
    lo = max(0, start_c - pad)
    return text[lo : lo + budget]


def same_doc_window(text: str, quote: str, *, budget: int = BUDGET, skip: int = SKIP) -> str | None:
    """`budget` characters of the same document that do not contain the quote: first try
    `skip` characters past it, then from the start, then right after it; None when the document
    is too short to avoid the quote."""
    loc = _locate(text, quote)
    if loc is None:
        return None
    start_c, end_c = loc
    q = norm(quote)
    for lo in (end_c + skip, 0, end_c):
        win = text[lo : lo + budget]
        if len(win) == budget and q not in norm(win):
            return win
    return None


def other_doc_chunk(ranked: list[Chunk], doc: str, *, budget: int = BUDGET):
    """(text, doc) of the best-ranked chunk from another document, cut to `budget`."""
    for c in ranked:
        if c.meta.get("doc") != doc:
            return c.text[:budget], c.meta.get("doc")
    return None, None


def budget_evidence(ranked: list[Chunk], gold: dict, *, budget: int = BUDGET) -> tuple[str, bool]:
    """Whole chunks in rank order while they fit; the first chunk cut when none fits (09's)."""
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


def _number_forms(v: float) -> list[str]:
    forms = {f"{v:g}"}
    if v == int(v):
        forms.add(f"{int(v):,}")
        if 0 < int(v) < 1000:
            forms.add(int_to_cn(int(v)))
        if 1000 <= int(v) < 100_000_000:
            cn = _cn_large(int(v))
            forms.add(cn)
            forms.add(cn.replace("萬", "萬零"))  # 三萬六百 is also written 三萬零六百
    return [f for f in forms if f]


def _cn_large(n: int) -> str:
    """5000 -> 五千, 30600 -> 三萬六百, 60000 -> 六萬 (chunkers.int_to_cn stops at 999; the
    零 variant is added by _number_forms)."""
    out = ""
    if n >= 10_000:
        high = n // 10_000
        out += (_cn_large(high) if high >= 1000 else int_to_cn(high)) + "萬"
        n %= 10_000
    if n >= 1000:
        out += int_to_cn(n // 1000) + "千"
        n %= 1000
    if n:
        out += int_to_cn(n)
    return out


def leak(window: str | None, qtype: str, expect: dict) -> bool:
    """Does a supposedly answer-free window actually hold the answer?
    value: the gold number (either script, commas allowed) directly before its unit;
    articles: every gold 條號 written as 第N條 in the window; which: the law name."""
    if not window:
        return False
    w = norm(window)
    if qtype == "value":
        unit = norm(expect["unit"])
        return any(norm(f) + unit in w for f in _number_forms(float(expect["value"])))
    if qtype == "articles":
        return all(_article_forms(a) & _article_mentions(w) for a in expect["articles"])
    name = norm(expect["law"])
    return name in w or any(norm(a) in w for a in expect.get("alt", []))


def _article_forms(label: str) -> set[str]:
    n, _, sub = label.partition("-")
    base = {f"第{int_to_cn(int(n))}條", f"第{n}條"}
    if sub:
        base = {f"{b}之{int_to_cn(int(sub))}" for b in base} | {f"{b}之{sub}" for b in base}
    return base


_ART = re.compile(r"第[零〇一二三四五六七八九十百0-9]+條(?:之[零〇一二三四五六七八九十0-9]+)?")


def _article_mentions(w: str) -> set[str]:
    return set(_ART.findall(w))


class Evidence:
    """Rankings for `real` / `other-doc`, texts for `gold` / `same-doc`, built once."""

    def __init__(self, docs: list[dict], questions: list[dict], *, embed=None):
        self.docs = docs
        self.laws = {d["sysid"]: law_name(d["caption"]) for d in docs}
        caches = sorted((DATA / "emb-cache").glob("*.npz"))
        if embed is None:
            digest = ollama.model_digest(EMBED_MODEL)
            cache_path = DATA / "emb-cache" / f"{digest.split(':')[1][:12]}.npz"
        else:
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
        route, chunking, prefix = REAL_CELL
        chunks = build_cell(self.texts[route], docs, chunking=chunking, prefix=prefix)
        index = DenseIndex.build(chunks, self.embed, self.cache)
        self.ranked = {
            q["id"]: [chunks[i] for i in index.search(q["question"], DEPTH)] for q in questions
        }
        self.cache.save()
        logger.info("ranked", extra={"ctx": {"cell": REAL_CELL, "chunks": len(chunks)}})

    def _source_text(self, q: dict) -> str | None:
        for route in ("base", "ocr"):
            if _locate(self.texts[route][q["doc"]], q["quote"]) is not None:
                return self.texts[route][q["doc"]]
        return None

    def for_question(self, kind: str, q: dict) -> dict:
        """{text, label, has_gold, leak, source}; text None = no evidence / not constructible."""
        law = self.laws[q["doc"]]
        gold = {"doc": q["doc"], "quote": q["quote"]}
        out = {"text": None, "label": "", "has_gold": False, "leak": False, "source": ""}
        if kind == "none":
            return out
        if kind == "real":
            text, has_gold = budget_evidence(self.ranked[q["id"]], gold)
            return out | {
                "text": text,
                "label": "檢索到的段落，可能來自不同法規",
                "has_gold": has_gold,
            }
        if kind == "other-doc":
            text, doc = other_doc_chunk(self.ranked[q["id"]], q["doc"])
            if text is None:
                return out
            return out | {
                "text": text,
                "label": "檢索到的段落，可能來自不同法規",
                "leak": leak(text, q["type"], q["expect"]),
                "source": doc,
            }
        src = self._source_text(q)
        if src is None:
            return out
        if kind == "gold":
            win = oracle_window(src, q["quote"])
            return out | {"text": win, "label": f"《{law}》公告原文節錄", "has_gold": True}
        win = same_doc_window(src, q["quote"])  # same-doc
        if win is None:
            return out
        return out | {
            "text": win,
            "label": f"《{law}》公告原文節錄",
            "leak": leak(win, q["type"], q["expect"]),
            "source": q["doc"],
        }


def ask(model: str, prompt: str, kind: str, q: dict, ev: Evidence) -> dict:
    e = ev.for_question(kind, q)
    text = e["text"]
    user = user_prompt(
        q["question"], qtype=q["type"], evidence=text, label=e["label"], prompt=prompt
    )
    row = {
        "model": model,
        "prompt": prompt,
        "evidence": kind,
        "qid": q["id"],
        "type": q["type"],
        "has_gold": e["has_gold"],
        "leak": e["leak"],
        "source": e["source"],
        "evidence_chars": len(text or ""),
        "evidence_text": text or "",
        "prompt_len": len(user),
    }
    try:
        r = ollama.chat(
            user,
            model=model,
            system=SYSTEM,
            format=SCHEMAS[prompt][q["type"]],
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
            "call failed", extra={"ctx": {"model": model, "prompt": prompt, "qid": q["id"]}}
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
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--evidence", default=",".join(EVIDENCE))
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke test)")
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args(argv)
    models, prompts, kinds = (s.split(",") for s in (args.models, args.prompts, args.evidence))
    setup_logging("rag_lab")
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.out / "raw.jsonl"
    docs = load_docs()
    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]
    byid = {q["id"]: q for q in questions}
    ev = Evidence(docs, questions)
    done = done_keys(raw)
    todo = [k for k in grid(models, prompts, kinds, questions) if k not in done]
    logger.info(
        "grid",
        extra={
            "ctx": {
                "total": len(models) * len(prompts) * len(kinds) * len(questions),
                "todo": len(todo),
            }
        },
    )
    with raw.open("a", encoding="utf-8") as f:
        for i, (m, p, k, qid) in enumerate(todo, 1):
            row = ask(m, p, k, byid[qid], ev)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(todo):
                logger.info(
                    "progress",
                    extra={
                        "ctx": {"done": i, "todo": len(todo), "model": m, "prompt": p, "kind": k}
                    },
                )
    first = {
        (r["model"], r["prompt"], r["evidence"], r["qid"]): r["content"]
        for r in map(json.loads, raw.read_text("utf-8").splitlines())
        if r
    }
    probes = []
    for m, p, k, qid in drift_sample(models, prompts, kinds, questions, per_cell=DRIFT_PER_CELL):
        again = ask(m, p, k, byid[qid], ev)
        probes.append(
            {
                "model": m,
                "prompt": p,
                "evidence": k,
                "qid": qid,
                "same": again["content"] == first.get((m, p, k, qid)),
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
        "skip": SKIP,
        "real_cell": list(REAL_CELL),
        "prompts": prompts,
        "evidence": kinds,
        "n_questions": len(questions),
        "drift": {"same": same, "n": len(probes), "rows": probes},
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
