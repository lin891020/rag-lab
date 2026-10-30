"""How many questions does each "indistinguishable" pair need? Paired-proportions power
(Connor 1987, the formula behind Stata `power pairedproportions` and R `pwr_mcnemar`) applied
to the per-question paired outcomes already on disk:

    uv run python experiments/15-more-questions/power.py   -> results/power.csv, results/power.md

For a pair (A, B) scored per question as 0/1 (hit@1000 in 01/07, value_ok in 10):
  p10 = P(A right, B wrong), p01 = P(A wrong, B right), p_disc = p10 + p01, d = p10 - p01
  n(d) = ((z_a * sqrt(p_disc) + z_b * sqrt(p_disc - d^2)) / d)^2      (alpha .05 two-sided, power .8)
  MDE(n) = the d that makes n(d) == n (bisection).
The pre-registration: a pair is predicted "會" when |observed d| >= MDE at the planned n,
"收斂" when the 95% half-width at the planned n is <= .03 (a real null bounded tightly),
otherwise "不會".
"""

from __future__ import annotations

import csv
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
EXPS = HERE.parent
RESULTS = HERE / "results"
Z_A = 1.959964  # two-sided alpha .05
Z_B = 0.841621  # power .8
CONVERGED_HALF_WIDTH = 0.03

# (experiment, label, metric file, key builder) -> pairs (A, B, planned n, bucket)
PLANNED_N = {"10": 320, "01-article": 90, "01-table": 45, "07": 90}
PAIRS_10 = [
    ("+cnpre", "fixed"),
    ("+rows", "fixed"),
    ("full", "-cnpre"),
    ("full", "-rows"),
    ("oracle", "full"),
]
PAIRS_01 = [
    (("dense", "article"), ("dense", "fixed269"), "01-article"),
    (("bm25", "article"), ("bm25", "fixed1000"), "01-article"),
    (("bm25", "article"), ("bm25", "fixed269"), "01-article"),
    (("dense", "article"), ("bm25", "article"), "01-article"),
    (("dense", "table-flat-fixed"), ("dense", "table-row"), "01-table"),
    (("dense", "table-row"), ("dense", "table-whole"), "01-table"),
    (("bm25", "table-flat-fixed"), ("bm25", "table-row"), "01-table"),
    (("bm25", "table-flat-fixed"), ("bm25", "table-whole"), "01-table"),
    (("bm25", "table-row"), ("bm25", "table-whole"), "01-table"),
]
PAIRS_07 = [
    (("dense", "article+context"), ("dense", "article")),
    (("dense", "article"), ("dense", "article-noprefix")),
    (("dense", "article"), ("dense", "small-to-big")),
    (("hybrid", "article"), ("dense", "article")),
    (("dense", "article"), ("bm25", "article")),
    (("hybrid", "small-to-big"), ("bm25", "small-to-big")),
]


def n_needed(p10: float, p01: float, d: float) -> float | None:
    """Pairs needed to detect a true difference `d` given the discordant rates."""
    p_disc = p10 + p01
    if d <= 0 or p_disc <= 0 or d * d > p_disc:
        return None
    return ((Z_A * math.sqrt(p_disc) + Z_B * math.sqrt(p_disc - d * d)) / d) ** 2


def mde(p_disc: float, n: int) -> float | None:
    """Smallest difference detectable with `n` pairs at this discordant rate (bisection)."""
    if p_disc <= 0 or n <= 0:
        return None
    lo, hi = 1e-6, math.sqrt(p_disc)
    if n_needed(p_disc, 0.0, hi) is not None and n_needed(p_disc, 0.0, hi) > n:  # type: ignore[operator]
        return None  # even the largest possible difference needs more pairs
    for _ in range(100):
        mid = (lo + hi) / 2
        need = n_needed(p_disc, 0.0, mid)
        if need is None or need > n:
            lo = mid
        else:
            hi = mid
    return hi


def half_width(p_disc: float, d: float, n: int) -> float:
    """Normal-approximation 95% half-width of a paired difference at `n` pairs."""
    return Z_A * math.sqrt(max(p_disc - d * d, 0.0) / n)


def pair_stats(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, float, float]:
    """(n, p10, p01) over the questions both sides scored."""
    qids = sorted(set(a) & set(b))
    p10 = sum(a[q] and not b[q] for q in qids) / len(qids)
    p01 = sum(b[q] and not a[q] for q in qids) / len(qids)
    return len(qids), p10, p01


def predict(d: float, p_disc: float, n_new: int) -> str:
    m = mde(p_disc, n_new)
    if m is not None and abs(d) >= m:
        return "會"
    if half_width(p_disc, d, n_new) <= CONVERGED_HALF_WIDTH:
        return "收斂"
    return "不會"


def row(exp: str, model: str, a: str, b: str, bucket: str, sa: dict, sb: dict) -> dict:
    n, p10, p01 = pair_stats(sa, sb)
    d = p10 - p01
    p_disc = p10 + p01
    n_new = PLANNED_N[bucket]
    need_obs = n_needed(p10, p01, abs(d))
    need_05 = n_needed(p10, p01, 0.05)
    return {
        "exp": exp,
        "model": model,
        "pair": f"{a} − {b}",
        "n_old": n,
        "diff": round(d, 3),
        "p_disc": round(p_disc, 3),
        "mde_old": round(mde(p_disc, n) or 0, 3),
        "n_new": n_new,
        "mde_new": round(mde(p_disc, n_new) or 0, 3),
        "half_width_new": round(half_width(p_disc, d, n_new), 3),
        "n_for_observed": round(need_obs) if need_obs else "",
        "n_for_05": round(need_05) if need_05 else "",
        "prediction": predict(d, p_disc, n_new),
    }


def load_10() -> dict[tuple[str, str], dict[str, bool]]:
    cells: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    path = EXPS / "10-fix-ablation/results/read/scored.jsonl"
    for ln in path.read_text("utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            cells[(r["model"], r["condition"])][r["qid"]] = bool(r["value_ok"])
    return cells


def load_retrieval(exp: str) -> dict[tuple[str, str], dict[str, bool]]:
    cells: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    for ln in (EXPS / exp / "results/raw.jsonl").read_text("utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            cells[(r["retriever"], r["chunking"])][r["qid"]] = r["hit@1000"] >= 1
    return cells


def compute() -> list[dict]:
    out = []
    ten = load_10()
    for model in sorted({m for m, _ in ten}):
        for a, b in PAIRS_10:
            out.append(row("10", model, a, b, "10", ten[(model, a)], ten[(model, b)]))
    one = load_retrieval("01-chunking")
    for a, b, bucket in PAIRS_01:
        out.append(row("01", "bge-m3", "/".join(a), "/".join(b), bucket, one[a], one[b]))
    seven = load_retrieval("07-chunking-2")
    for a, b in PAIRS_07:
        out.append(row("07", "bge-m3", "/".join(a), "/".join(b), "07", seven[a], seven[b]))
    return out


def to_markdown(rows: list[dict]) -> str:
    head = "| 實驗 | 模型 | 對 | 舊題數 | 觀察差距 | 不一致率 | MDE@舊 | 新題數 | MDE@新 | 新區間半寬 | 偵測觀察差距要幾題 | 預測 |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    body = "".join(
        f"| {r['exp']} | {r['model']} | {r['pair']} | {r['n_old']} | {r['diff']:+.3f} | {r['p_disc']:.2f} "
        f"| {r['mde_old']:.3f} | {r['n_new']} | {r['mde_new']:.3f} | ±{r['half_width_new']:.3f} "
        f"| {r['n_for_observed']} | {r['prediction']} |\n"
        for r in rows
    )
    return head + body


def main() -> int:
    from rag_lab.log import setup_logging

    setup_logging("rag_lab")
    rows = compute()
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "power.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (RESULTS / "power.md").write_text(to_markdown(rows), encoding="utf-8")
    logger.info("power", extra={"ctx": {"pairs": len(rows)}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
