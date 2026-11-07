"""Score results/raw.jsonl -> scored.jsonl, summary.csv, diffs.csv.

    uv run python experiments/29-amendment-diff/score.py

One ruler per question kind (pre-registered, plan §4):
  T1  found false  <=> the gold is 新增 (no old article); otherwise norm_article(old_article)
      must equal the gold old number
  T2  changed must equal the gold
  T3  both values within TOLERANCE (relative 1 %, exact when the gold is 0) of the gold, in the
      question's unit; swapped (old == gold new and new == gold old) is kept as its own flag
  stopped / unparsable  wrong
Also per row: old_hit (the gold old article was in the union of what the searches returned;
recomputed here from the stored labels, agent rows only),
answer_grounded (T1: the old number the model gave was in that union; closed and oracle have no
union). Primary tests (the only verdicts): oracle − agent on T2+T3 for each model, bootstrap by
question (2000, seed 0, same resample both arms) and by chapter (resample chapters). Everything
else is described. Refuses to score while a call failed, a row is stale or a cell is missing.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))
from articles import norm_article  # noqa: E402
from run import KINDS, MODELS, RUNS, asks, last_rows, load_questions, signatures  # noqa: E402

TOLERANCE = 0.01  # relative (16's), T3 values
OPUS, HAIKU = "claude-opus-5-5", "claude-haiku-5-5"
Pair = tuple[str, str, str, str, str]  # (model A, condition A, model B, condition B, scope)
PRIMARY: list[Pair] = [(m, "oracle", m, "agent", "T2+T3") for m in (HAIKU, OPUS)]
SECONDARY: list[Pair] = [  # described, no verdict
    *[(m, "agent", m, "closed", "all") for m in (HAIKU, OPUS)],
    (OPUS, "agent", HAIKU, "agent", "all"),
    (HAIKU, "agent-2", HAIKU, "agent", "all"),
]
DIFF_METRICS = ("right", "old_hit")
METRICS = ["right", "old_hit", "answer_grounded", "found", "parse_ok", "swapped"]
_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n?(.*?)\n?\s*```\s*$", re.S)


def _parse(content: str) -> tuple[object, bool, bool]:
    fenced = False
    m = _FENCE.match(content)
    if m:
        content, fenced = m.group(1), True
    try:
        return json.loads(content), True, fenced
    except json.JSONDecodeError:
        return None, False, fenced


def _num(x: object) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.replace(",", ""))
        except ValueError:
            return None
    return None


def close(value: float | None, gold: float) -> bool:
    if value is None:
        return False
    if gold == 0:
        return value == 0
    return abs(value - gold) <= TOLERANCE * abs(gold)


def judge(content: str, q: dict) -> dict:
    """All per-reply flags for one question; `right` is the pre-registered ruler."""
    obj, parse_ok, fenced = _parse(content)
    obj = obj if isinstance(obj, dict) else {}
    found = obj.get("found") is True
    answer_num = norm_article(str(obj.get("old_article") or "")) if found else ""
    e = q["expect"]
    out = {
        "parse_ok": parse_ok,
        "fenced": fenced,
        "found": found,
        "answer_old_num": answer_num,
        "swapped": False,
        "right": False,
    }
    if q["kind"] == "T1":
        out["right"] = (
            (not found) if e["old_num"] is None else (found and answer_num == e["old_num"])
        )
    elif q["kind"] == "T2":
        out["answer_changed"] = obj.get("changed") is True
        out["right"] = found and (obj.get("changed") is True) == e["changed"]
    else:
        ov, nv = _num(obj.get("old_value")), _num(obj.get("new_value"))
        out.update(answer_old_value=ov, answer_new_value=nv, answer_unit=str(obj.get("unit") or ""))
        out["right"] = found and close(ov, e["old_value"]) and close(nv, e["new_value"])
        out["swapped"] = (
            found and close(ov, e["new_value"]) and close(nv, e["old_value"]) and not out["right"]
        )
    return out


def score_rows(raw: list[dict], questions: dict[str, dict]) -> list[dict]:
    out = []
    for r in raw:
        q = questions[r["qid"]]
        flags = judge(r.get("content", "") or "", q)
        if r.get("stopped") or not flags["parse_ok"]:
            flags.update(right=False, swapped=False, grade="stopped")
        labels = set(r.get("labels", []))
        gold = set(q["gold_labels"])
        old_hit = (gold <= labels) if gold and r["condition"].startswith("agent") else None
        grounded = (
            bool(flags["answer_old_num"]) and f"old:{flags['answer_old_num']}" in labels
            if r["condition"].startswith("agent")
            else None
        )
        out.append(
            {
                **r,
                "gold_old_num": q["old_num"],
                "old_hit": old_hit,
                "answer_grounded": grounded,
                **flags,
            }
        )
    return out


def bootstrap_ci(
    values: list[float], *, n: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff_ci(
    a: list[float], b: list[float], *, n: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    d = [float(x) - float(y) for x, y in zip(a, b, strict=True)]
    return bootstrap_ci(d, n=n, seed=seed)


def cluster_diff_ci(
    diffs: dict[str, list[float]], *, n: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Resample clusters (chapters), concatenate, mean."""
    clusters = [np.asarray(v, dtype=float) for v in diffs.values()]
    allq = np.concatenate(clusters)
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(clusters), size=len(clusters))
        means[i] = np.concatenate([clusters[j] for j in idx]).mean()
    return float(allq.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _mean_row(model: str, condition: str, kind: str, metric: str, sub: list[dict]) -> dict:
    vals = [float(bool(r[metric])) for r in sub]
    m, lo, hi = bootstrap_ci(vals)
    return {
        "model": model,
        "condition": condition,
        "kind": kind,
        "metric": metric,
        "n": len(sub),
        "mean": round(m, 4),
        "lo": round(lo, 4),
        "hi": round(hi, 4),
    }


def _scope_rows(rows: list[dict], metric: str) -> list[dict]:
    if metric in ("old_hit", "answer_grounded"):
        return [r for r in rows if r["condition"].startswith("agent") and r.get(metric) is not None]
    if metric == "swapped":
        return [r for r in rows if r["kind"] == "T3"]
    return rows


def summarize(scored: list[dict]) -> tuple[list[dict], list[dict]]:
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in scored:
        cells[(r["model"], r["condition"])].append(r)
    summary = []
    for (model, cond), rows in sorted(cells.items()):
        for kind in ["all", "T2+T3", *KINDS]:
            kinds = KINDS if kind == "all" else kind.split("+")
            sub_k = [r for r in rows if r["kind"] in kinds]
            for m in METRICS:
                sub = _scope_rows(sub_k, m)
                if sub:
                    summary.append(_mean_row(model, cond, kind, m, sub))
    pairs = [(*p, True) for p in PRIMARY] + [(*p, False) for p in SECONDARY]
    pairs += [(ma, a, mb, b, k, False) for ma, a, mb, b, _ in PRIMARY + SECONDARY for k in KINDS]
    diffs = []
    seen = set()
    for ma, a, mb, b, scope, primary in pairs:
        if (ma, a, mb, b, scope) in seen:
            continue
        seen.add((ma, a, mb, b, scope))
        ra, rb = cells.get((ma, a)), cells.get((mb, b))
        if not ra or not rb:
            continue
        kinds = KINDS if scope == "all" else scope.split("+")
        qa = {r["qid"]: r for r in ra if r["kind"] in kinds}
        qb = {r["qid"]: r for r in rb if r["kind"] in kinds}
        for metric in DIFF_METRICS:
            common = sorted(
                q
                for q in qa
                if q in qb
                and (
                    metric != "old_hit"
                    or qa[q].get("old_hit") is not None
                    and qb[q].get("old_hit") is not None
                )
            )
            if not common or (
                metric == "old_hit" and not (a.startswith("agent") and b.startswith("agent"))
            ):
                continue
            d, lo, hi = paired_diff_ci(
                [float(qa[q][metric]) for q in common], [float(qb[q][metric]) for q in common]
            )
            by_cluster: dict[str, list[float]] = defaultdict(list)
            for q in common:
                by_cluster[qa[q]["chapter"]].append(float(qa[q][metric]) - float(qb[q][metric]))
            dc, loc, hic = cluster_diff_ci(by_cluster)
            diffs.append(
                {
                    "model": ma,
                    "a": a,
                    "model_b": mb,
                    "b": b,
                    "scope": scope,
                    "primary": primary,
                    "metric": metric,
                    "n": len(common),
                    "diff": round(d, 4),
                    "lo": round(lo, 4),
                    "hi": round(hi, 4),
                    "separable": lo > 0 or hi < 0,
                    "lo_cluster": round(loc, 4),
                    "hi_cluster": round(hic, 4),
                    "separable_cluster": loc > 0 or hic < 0,
                    "clusters": len(by_cluster),
                }
            )
    return summary, diffs


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    from rag_lab.log import setup_logging

    setup_logging("rag_lab")
    questions = {q["id"]: q for q in load_questions()}
    raw = last_rows(RESULTS / "raw.jsonl")  # a retried question counts once, its final row
    sig = signatures(list(questions.values()))
    failed = [r for r in raw if r.get("error")]
    stale = [r for r in raw if r.get("prompt_sha") != sig.get((r["condition"], r["qid"]))]
    have = {(r["model"], r["condition"], r["qid"]) for r in raw}
    missing = [
        (m, c, q["id"])
        for m in MODELS
        for c in RUNS[m]
        for q in questions.values()
        if asks(c, q) and (m, c, q["id"]) not in have
    ]
    if failed or stale or missing:  # a failed call would score as wrong; a stale row asked
        # another prompt; a missing cell (a budget stop) would pair diffs on fewer questions
        logger.error(
            "not scoring",
            extra={"ctx": {"failed": len(failed), "stale": len(stale), "missing": len(missing)}},
        )
        return 1
    scored = score_rows(raw, questions)
    (RESULTS / "scored.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in scored), "utf-8"
    )
    summary, diffs = summarize(scored)
    _write_csv(RESULTS / "summary.csv", summary)
    if diffs:
        _write_csv(RESULTS / "diffs.csv", diffs)
    logger.info("scored", extra={"ctx": {"rows": len(scored)}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
