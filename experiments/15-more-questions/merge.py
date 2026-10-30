"""Old bank + new bank, per pre-registered pair: the paired difference on the old questions,
on the new ones, on both pooled (bootstrap by question, as 01/07/10 do: 2000 resamples, seed 0,
the same resampled questions on both arms) and pooled with a cluster bootstrap (resample
documents, or for the law corpus the section the gold article sits in; tables are their own
cluster). Each row carries power.py's prediction and whether it came true.

    uv run python experiments/15-more-questions/merge.py   -> results/merged/{diffs.csv,diffs.md}

Inputs (all produced by chain15.sh):
  results/rescored-old/scored.jsonl   10's raw.jsonl scored with this experiment's ruler
  results/read/scored.jsonl           the new gazette bank
  results/law01/raw.jsonl             01's run.py on 01's two banks + questions-law.jsonl
  results/law07/raw.jsonl             07's run.py on the same three files
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from power import PAIRS_01, PAIRS_07, PAIRS_10, half_width, mde  # noqa: E402

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
EXPS = HERE.parent
RESULTS = HERE / "results"
N_BOOT = 2000
SEED = 0
Outcome = dict[str, bool]  # qid -> right


def paired_diff_ci(a: list[float], b: list[float], *, n: int = N_BOOT, seed: int = SEED):
    """Mean of a−b and its bootstrap 95% interval, resampling questions (both arms together)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cluster_diff_ci(diffs_by_cluster: dict[str, list[float]], *, n: int = N_BOOT, seed: int = SEED):
    """Mean of the per-question differences and its interval when whole clusters are resampled:
    draw len(clusters) clusters with replacement, concatenate their questions, take the mean."""
    clusters = [np.asarray(v, dtype=float) for v in diffs_by_cluster.values() if len(v)]
    if not clusters:
        return 0.0, 0.0, 0.0
    allq = np.concatenate(clusters)
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(clusters), size=len(clusters))
        means[i] = np.concatenate([clusters[j] for j in pick]).mean()
    return float(allq.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verdict(lo: float, hi: float) -> str:
    return "分得出" if lo > 0 or hi < 0 else "分不出"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text("utf-8").splitlines() if ln.strip()]


def gazette_cells(path: Path) -> dict[tuple[str, str], Outcome]:
    cells: dict[tuple[str, str], Outcome] = defaultdict(dict)
    for r in _load_jsonl(path):
        cells[(r["model"], r["condition"])][r["qid"]] = bool(r["value_ok"])
    return cells


def retrieval_cells(path: Path) -> dict[tuple[str, str], Outcome]:
    cells: dict[tuple[str, str], Outcome] = defaultdict(dict)
    for r in _load_jsonl(path):
        cells[(r["retriever"], r["chunking"])][r["qid"]] = r["hit@1000"] >= 1
    return cells


def pair_rows(
    *,
    exp: str,
    model: str,
    pair: str,
    a: Outcome,
    b: Outcome,
    old_ids: set[str],
    new_ids: set[str],
    cluster_of: dict[str, str],
    n_planned: int,
) -> dict:
    """One merged row for a pair: old / new / pooled / pooled-cluster differences."""
    common = sorted(set(a) & set(b))
    groups = {
        "old": [q for q in common if q in old_ids],
        "new": [q for q in common if q in new_ids],
        "all": [q for q in common if q in old_ids or q in new_ids],
    }
    row: dict = {"exp": exp, "model": model, "pair": pair}
    for name, qids in groups.items():
        d, lo, hi = paired_diff_ci([float(a[q]) for q in qids], [float(b[q]) for q in qids])
        row.update(
            {
                f"n_{name}": len(qids),
                f"d_{name}": round(d, 3),
                f"lo_{name}": round(lo, 3),
                f"hi_{name}": round(hi, 3),
                f"v_{name}": verdict(lo, hi),
            }
        )
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for q in groups["all"]:
        by_cluster[cluster_of.get(q, q)].append(float(a[q]) - float(b[q]))
    d, lo, hi = cluster_diff_ci(by_cluster)
    row.update(
        {
            "clusters": len(by_cluster),
            "d_cluster": round(d, 3),
            "lo_cluster": round(lo, 3),
            "hi_cluster": round(hi, 3),
            "v_cluster": verdict(lo, hi),
        }
    )
    qids = groups["all"]
    if qids:
        p10 = sum(a[q] and not b[q] for q in qids) / len(qids)
        p01 = sum(b[q] and not a[q] for q in qids) / len(qids)
        pd = p10 + p01
        row["p_disc_all"] = round(pd, 3)
        row["mde_all"] = round(mde(pd, len(qids)) or 0, 3)
        row["half_width_all"] = round(half_width(pd, p10 - p01, len(qids)), 3)
    row["n_planned"] = n_planned
    return row


def outcome_vs_prediction(row: dict, prediction: str) -> str:
    """Did the pre-registered prediction come true? 會 -> pooled 分得出; 收斂 -> pooled 分不出 and
    half-width <= .03; 不會 -> pooled 分不出."""
    v = row["v_all"]
    if prediction == "會":
        ok = v == "分得出"
    elif prediction == "收斂":
        ok = v == "分不出" and row.get("half_width_all", 1) <= 0.03
    else:
        ok = v == "分不出"
    return f"{prediction}：{'如預期' if ok else '意外'}"


def _ids(path: Path) -> set[str]:
    return {json.loads(ln)["id"] for ln in path.read_text("utf-8").splitlines() if ln.strip()}


def _law_clusters() -> dict[str, str]:
    """question id -> cluster: the section (節) of the first gold article; tables cluster by id."""
    from rag_lab.pdf import extract_text

    from chunkers import parse_articles

    one = EXPS / "01-chunking"
    arts = {
        str(a.num): f"{a.chapter}/{a.section}"
        for a in parse_articles(extract_text(one / "corpus/law.pdf"))
    }
    out = {}
    for path in (
        one / "questions.jsonl",
        one / "questions-reserve.jsonl",
        HERE / "questions-law.jsonl",
    ):
        if path.exists():
            for q in _load_jsonl(path):
                g = q["gold"][0]
                out[q["id"]] = arts.get(g, g)
    return out


def _gazette_clusters() -> dict[str, str]:
    out = {}
    for path in (EXPS / "09-messy-real/questions.jsonl", HERE / "questions-gazette.jsonl"):
        if path.exists():
            for q in _load_jsonl(path):
                out[q["id"]] = q["doc"]
    return out


def compute(predictions: dict[tuple[str, str, str], str]) -> list[dict]:
    rows: list[dict] = []
    old10 = RESULTS / "rescored-old/scored.jsonl"
    new10 = RESULTS / "read/scored.jsonl"
    if old10.exists() and new10.exists():
        cells = gazette_cells(old10)
        for k, v in gazette_cells(new10).items():
            cells[k].update(v)
        old_ids = _ids(EXPS / "09-messy-real/questions.jsonl")
        new_ids = _ids(HERE / "questions-gazette.jsonl")
        clusters = _gazette_clusters()
        for model in sorted({m for m, _ in cells}):
            for a, b in PAIRS_10:
                rows.append(
                    pair_rows(
                        exp="10",
                        model=model,
                        pair=f"{a} − {b}",
                        a=cells[(model, a)],
                        b=cells[(model, b)],
                        old_ids=old_ids,
                        new_ids=new_ids,
                        cluster_of=clusters,
                        n_planned=len(old_ids) + len(new_ids),
                    )
                )
    law01 = RESULTS / "law01/raw.jsonl"
    law07 = RESULTS / "law07/raw.jsonl"
    if law01.exists() or law07.exists():
        one = EXPS / "01-chunking"
        old01 = _ids(one / "questions.jsonl")
        reserve = _ids(one / "questions-reserve.jsonl")
        new_law = (
            _ids(HERE / "questions-law.jsonl") if (HERE / "questions-law.jsonl").exists() else set()
        )
        clusters = _law_clusters()
        if law01.exists():
            cells = retrieval_cells(law01)
            for a, b, bucket in PAIRS_01:
                kind = "article" if bucket == "01-article" else "table"
                sel = {q for q in old01 if q.startswith("a" if kind == "article" else "t")}
                new_sel = {
                    q for q in new_law | reserve if q.startswith("a" if kind == "article" else "t")
                }
                rows.append(
                    pair_rows(
                        exp="01",
                        model="bge-m3",
                        pair=f"{'/'.join(a)} − {'/'.join(b)}",
                        a=cells[a],
                        b=cells[b],
                        old_ids=sel,
                        new_ids=new_sel,
                        cluster_of=clusters,
                        n_planned=len(sel) + len(new_sel),
                    )
                )
        if law07.exists():
            cells = retrieval_cells(law07)
            old07 = {q for q in old01 | reserve if q.startswith("a")}
            new07 = {q for q in new_law if q.startswith("a")}
            for a, b in PAIRS_07:
                rows.append(
                    pair_rows(
                        exp="07",
                        model="bge-m3",
                        pair=f"{'/'.join(a)} − {'/'.join(b)}",
                        a=cells[a],
                        b=cells[b],
                        old_ids=old07,
                        new_ids=new07,
                        cluster_of=clusters,
                        n_planned=len(old07) + len(new07),
                    )
                )
    for r in rows:
        pred = predictions.get((r["exp"], r["model"], r["pair"]), "")
        r["prediction"] = outcome_vs_prediction(r, pred) if pred else ""
    return rows


def to_markdown(rows: list[dict]) -> str:
    def ci(r: dict, k: str) -> str:
        return f"{r[f'd_{k}']:+.2f} [{r[f'lo_{k}']:+.2f}, {r[f'hi_{k}']:+.2f}] {r[f'v_{k}']}（{r[f'n_{k}'] if k != 'cluster' else r['clusters']}）"

    head = "| 實驗 | 模型 | 對 | 舊題 | 新題 | 合併（按題） | 合併（按群） | 合併 MDE | 事前預測 |\n|---|---|---|---|---|---|---|---|---|\n"
    return head + "".join(
        f"| {r['exp']} | {r['model']} | {r['pair']} | {ci(r, 'old')} | {ci(r, 'new')} | {ci(r, 'all')} | {ci(r, 'cluster')} | {r.get('mde_all', '')} | {r.get('prediction', '')} |\n"
        for r in rows
    )


def load_predictions() -> dict[tuple[str, str, str], str]:
    path = RESULTS / "power.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {(r["exp"], r["model"], r["pair"]): r["prediction"] for r in csv.DictReader(f)}


def main() -> int:
    from rag_lab.log import setup_logging

    setup_logging("rag_lab")
    rows = compute(load_predictions())
    if not rows:
        logger.warning("nothing to merge yet", extra={"ctx": {"results": str(RESULTS)}})
        return 1
    out = RESULTS / "merged"
    out.mkdir(exist_ok=True)
    with (out / "diffs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=sorted(
                {k for r in rows for k in r},
                key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99,
            ),
        )
        w.writeheader()
        w.writerows(rows)
    (out / "diffs.md").write_text(to_markdown(rows), encoding="utf-8")
    logger.info("merged", extra={"ctx": {"pairs": len(rows)}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
