"""Post-hoc check (added after the run): how many new questions re-ask an old one, and the
pre-registered pairs with those dropped.

    uv run python experiments/15-more-questions/reasks.py   -> results/merged/reasks.{json,md}

Gazette re-ask: same document, same type, same expected answer (any `which` question of the same
document counts), and the two quotes share at least half of the shorter one's character bigrams.
Law re-ask: the new question's gold shares an article or table with an old question's gold.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from merge import (  # noqa: E402
    EXPS,
    RESULTS,
    _load_jsonl,
    gazette_cells,
    paired_diff_ci,
    retrieval_cells,
    verdict,
)
from power import PAIRS_01, PAIRS_07, PAIRS_10  # noqa: E402

HERE = Path(__file__).parent
QUOTE_OVERLAP = 0.5


def _norm(s: str) -> str:
    return "".join(str(s).split())


def _bigrams(s: str) -> set[str]:
    return {s[i : i + 2] for i in range(len(s) - 1)}


def quote_overlap(a: str, b: str) -> float:
    x, y = _bigrams(_norm(a)), _bigrams(_norm(b))
    return len(x & y) / min(len(x), len(y)) if x and y else 0.0


def gazette_twins(old: list[dict], new: list[dict]) -> dict[str, str]:
    """new id -> the old question it re-asks."""

    def key(q: dict) -> str:
        return json.dumps(q["expect"], ensure_ascii=False, sort_keys=True)

    out = {}
    for n in new:
        best = (0.0, "")
        for o in old:
            if o["doc"] != n["doc"] or o["type"] != n["type"]:
                continue
            if n["type"] != "which" and key(o) != key(n):
                continue
            ov = quote_overlap(n["quote"], o["quote"])
            if ov > best[0]:
                best = (ov, o["id"])
        if best[0] >= QUOTE_OVERLAP:
            out[n["id"]] = best[1]
    return out


def law_reasks(old: list[dict], new: list[dict]) -> set[str]:
    seen = {str(g) for o in old for g in o["gold"]}
    return {n["id"] for n in new if any(str(g) in seen for g in n["gold"])}


def _diff(a: dict, b: dict, ids: list[str]) -> dict:
    d, lo, hi = paired_diff_ci([float(a[q]) for q in ids], [float(b[q]) for q in ids])
    return {
        "n": len(ids),
        "d": round(d, 3),
        "lo": round(lo, 3),
        "hi": round(hi, 3),
        "v": verdict(lo, hi),
    }


def main() -> int:
    out: dict = {"pairs": []}
    old_g = _load_jsonl(EXPS / "09-messy-real/questions.jsonl")
    new_g = _load_jsonl(HERE / "questions-gazette.jsonl")
    twins = gazette_twins(old_g, new_g)
    oc = gazette_cells(RESULTS / "rescored-old/scored.jsonl")
    nc = gazette_cells(RESULTS / "read/scored.jsonl")
    rng = random.Random(0)
    old_ids = [o["id"] for o in old_g]
    agree = rand = tot = 0
    for k in nc:
        if k not in oc or k[1] == "closed-book":
            continue
        for n, o in twins.items():
            tot += 1
            agree += nc[k][n] == oc[k][o]
            rand += nc[k][n] == oc[k][rng.choice(old_ids)]
    out["gazette"] = {
        "reasks": len(twins),
        "of": len(new_g),
        "agree": round(agree / tot, 3),
        "agree_random_old": round(rand / tot, 3),
        "pairs_compared": tot,
    }
    keep = {q["id"] for q in old_g} | ({q["id"] for q in new_g} - set(twins))
    cells = {k: {**oc.get(k, {}), **v} for k, v in nc.items()}
    for m in sorted({k[0] for k in cells}):
        for a, b in PAIRS_10:
            ids = sorted(q for q in cells[(m, a)] if q in cells[(m, b)] and q in keep)
            out["pairs"].append(
                {
                    "exp": "10",
                    "model": m,
                    "pair": f"{a} − {b}",
                    **_diff(cells[(m, a)], cells[(m, b)], ids),
                }
            )
    one = EXPS / "01-chunking"
    main01 = _load_jsonl(one / "questions.jsonl")
    old_law = main01 + _load_jsonl(one / "questions-reserve.jsonl")
    new_law = _load_jsonl(HERE / "questions-law.jsonl")
    re_law = law_reasks(old_law, new_law)
    out["law"] = {"reasks": len(re_law), "of": len(new_law)}
    fresh = {q["id"] for q in new_law} - re_law
    for exp, pairs, oldset in (
        ("01", PAIRS_01, {q["id"] for q in main01}),
        ("07", PAIRS_07, {q["id"] for q in old_law if q["id"].startswith("a")}),
    ):
        cells_r = retrieval_cells(RESULTS / f"law{exp}/raw.jsonl")
        for (ra, ca), (rb, cb), *_ in pairs:
            a, b = cells_r[(ra, ca)], cells_r[(rb, cb)]
            kind = "t" if "table" in ca else "a"
            ids = sorted(q for q in a if q in b and q[0] == kind and (q in oldset or q in fresh))
            out["pairs"].append(
                {
                    "exp": exp,
                    "model": "bge-m3",
                    "pair": f"{ra}/{ca} − {rb}/{cb}",
                    **_diff(a, b, ids),
                }
            )
    dest = RESULTS / "merged"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "reasks.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    g = out["gazette"]
    lines = [
        f"公報換句話問：{g['reasks']}/{g['of']}；與舊題對錯一致 {g['agree']}（隨機舊題 {g['agree_random_old']}，{g['pairs_compared']} 對）",
        f"法規同一條／同一張表：{out['law']['reasks']}/{out['law']['of']}",
        "",
        "| 實驗 | 模型 | 對 | 題數 | 差距 [95% 區間] | 判定 |",
        "|---|---|---|---|---|---|",
    ]
    for r in out["pairs"]:
        lines.append(
            f"| {r['exp']} | {r['model']} | {r['pair']} | {r['n']} | "
            f"{r['d']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] | {r['v']} |"
        )
    (dest / "reasks.md").write_text("\n".join(lines) + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
