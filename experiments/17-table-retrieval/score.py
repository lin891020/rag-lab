"""Score results/read/raw.jsonl -> scored.jsonl, summary.csv, diffs.csv. Program logic only.

    uv run python experiments/17-table-retrieval/score.py [--raw ...] [--out ...]

Per reply (10's parse + unit ruler, 11's found flag, 16's two scales):
  answered     found is true and value is a positive number
  number       |value − gold| / gold <= TOLERANCE  -> match, else error
  ladder       value on the question's ladder: == gold match; one step up acceptable; two or
               more up error; below gold CRITICAL (undersized, the unsafe side); off-ladder
               nonstandard (counted as error)
  abstain q    abstained -> right; answered -> hard answer (wrong)
  strict / lenient   unit must match / unit may be empty (10)
Summary per (model, condition, chain|all); diffs: the four pre-registered primary tests, the
secondary pairs and every primary pair per chain (description only). Bootstrap by question (2000,
seed 0, same resample both arms) and by cluster (within chain: resample (table,row) clusters).
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
RESULTS = HERE / "results/read"
QUESTIONS = HERE / "questions.jsonl"
TOLERANCE = 0.01  # relative, number questions (the assignment-style ruler uses 2 %; stricter here)
ACCEPTABLE_STEPS_ABOVE = 1  # one ladder step above the gold is still acceptable
CHAINS = ["C1", "C2", "C3", "C4", "C5"]
CONDITIONS = ["struct@1500", "struct@3000-nosub", "struct@3000", "cards", "decomp", "oracle"]
PRIMARY: list[tuple[str, str, str]] = [  # (A, B, scope): pre-registered, the only verdicts
    ("cards", "struct@3000", "all"),  # the search, same budget and tables
    ("decomp", "cards", "all"),  # splitting the question
    ("struct@3000-nosub", "struct@1500", "all"),  # the budget alone
    ("struct@3000", "struct@3000-nosub", "all"),  # the restored subtitles alone
]
SECONDARY: list[tuple[str, str, str]] = [  # described, no verdict
    ("decomp", "struct@1500", "all"),
    ("oracle", "decomp", "all"),
    ("decomp", "struct@3000", "all"),
    ("decomp@4000", "decomp", "all"),
    ("oracle", "decomp@4000", "all"),
]
METRICS = [
    "right",
    "right_lenient",
    "critical",
    "abstain_ok",
    "hard_answer",
    "rows_all_hit",
    "expr_ok",
    "parse_ok",
]

UNIT_ALIASES: dict[str, list[str]] = {
    "": ["", "無", "none", "無單位", "係數", "factor", "-", "倍", "x", "×", "ratio"],
    "倍": ["倍", "x", "×", "times", "ratio", ""],
    "%": ["%", "％", "percent", "百分比", "pct"],
    "mm": ["mm", "毫米", "公厘", "公釐", "millimeter", "millimetre", "mm."],
    "cm": ["cm", "公分", "厘米"],
    "m": ["m", "米", "公尺", "meter", "metre", "meters", "metres"],
    "mm²": [
        "mm²",
        "mm2",
        "mm^2",
        "平方毫米",
        "平方公厘",
        "平方公釐",
        "平方",
        "sqmm",
        "sq.mm",
        "mm²",
    ],
    "m²": ["m²", "m2", "m^2", "平方米", "平方公尺", "sqm"],
    "V": ["v", "伏特", "伏", "volt", "volts"],
    "VA": ["va", "v·a", "v.a", "v-a", "伏安", "volt-ampere", "voltampere", "伏特安培"],
    "VA/m²": ["va/m²", "va/m2", "va/m^2", "伏安/平方米", "伏安每平方米", "va/㎡", "va每平方米"],
    "kVA": ["kva", "千伏安"],
    "A": ["a", "安培", "安", "amp", "amps", "ampere", "amperes"],
    "kA": ["ka", "千安培", "kiloampere", "kiloamperes", "仟安培"],
    "mA": ["ma", "毫安培", "毫安", "milliampere", "milliamperes"],
    "W": ["w", "瓦", "瓦特", "watt", "watts"],
    "kW": ["kw", "千瓦", "瓩", "kilowatt", "kilowatts", "仟瓦"],
    "s": ["s", "秒", "sec", "secs", "second", "seconds"],
    "ms": ["ms", "毫秒", "millisecond", "milliseconds"],
    "Ω": ["ω", "ohm", "ohms", "歐姆", "歐", "Ω"],
    "℃": ["℃", "°c", "c", "度", "攝氏", "度c", "degc", "celsius", "攝氏度", "°"],
    # gazette units (10): synonyms only; "日" is not "工作日", "月" is not "年"
    "元": ["元", "新臺幣元", "新台幣元", "新臺幣", "新台幣", "nt$", "ntd", "twd", "塊"],
    "日": ["日", "天"],
    "工作日": ["工作日", "個工作日", "工作天"],
    "個月": ["個月", "月"],
    "年": ["年"],
    "人": ["人", "名", "位"],
    "次": ["次", "回"],
    "條": ["條"],
    "點": ["點"],
    "節": ["節"],
    "個": ["個"],
    "件": ["件"],
    "戶": ["戶"],
    "歲": ["歲"],
    "公分": ["公分", "cm", "厘米"],
    "公尺": ["公尺", "m", "米"],
    "公噸": ["公噸", "噸", "t", "ton", "tonne"],
    "hp": ["hp", "馬力", "HP", "horsepower"],
    "ppm": ["ppm"],
}


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).strip().lower().replace(" ", "")
    return s.replace("^2", "2").replace("²", "2").replace("㎡", "m2")


_CANON: dict[str, str] = {}
for _u, _al in UNIT_ALIASES.items():
    for _a in [_u, *_al]:
        _CANON.setdefault(_fold(_a), _u)


def normalize_unit(unit: str) -> str:
    """'mm^2' / '平方毫米' -> 'mm²'; unknown spellings pass through folded."""
    f = _fold(unit)
    return _CANON.get(f, f)


def _unit_matches(got: str, want: str) -> bool:
    """Both sides canonicalised, so 名 vs 人 matches whichever one the question expects."""
    if _fold(got) in {_fold(a) for a in [want, *UNIT_ALIASES.get(want, [])]}:
        return True
    return normalize_unit(got) == normalize_unit(want)


_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n?(.*?)\n?\s*```\s*$", re.S)


def _parse(content: str) -> tuple[object, bool, bool]:
    """(obj, parse_ok, fenced)."""
    fenced = False
    m = _FENCE.match(content)
    if m:
        content, fenced = m.group(1), True
    try:
        return json.loads(content), True, fenced
    except json.JSONDecodeError:
        return None, False, fenced


def _is_number(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _to_float(x: object) -> float | None:
    if _is_number(x):
        return float(x)  # type: ignore[arg-type]
    if isinstance(x, str):
        try:
            return float(_fold(x).replace(",", ""))
        except ValueError:
            return None
    return None


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


def grade_number(value: float, gold: float) -> str:
    return "match" if abs(value - gold) <= TOLERANCE * abs(gold) else "error"


def grade_ladder(value: float, gold: float, ladder: list[float]) -> str:
    if value not in ladder:
        return "nonstandard"
    steps = ladder.index(value) - ladder.index(gold)
    if steps == 0:
        return "match"
    if steps < 0:
        return "critical"
    return "acceptable_alternative" if steps <= ACCEPTABLE_STEPS_ABOVE else "error"


def judge(content: str, q: dict) -> dict:
    """All per-reply flags for one question."""
    obj, parse_ok, fenced = _parse(content)
    obj = obj if isinstance(obj, dict) else {}
    value = _to_float(obj.get("value"))
    found = obj.get("found") is True
    answered = found and value is not None and value > 0
    unit = str(obj.get("unit") or "")
    e = q["expect"]
    out = {
        "parse_ok": parse_ok,
        "fenced": fenced,
        "found": found,
        "answered": answered,
        "value": value,
        "unit": unit,
        "expression": str(obj.get("expression") or ""),
        "quote": str(obj.get("quote") or ""),
    }
    kind = e["kind"]
    if kind == "abstain":
        out.update(
            grade="abstained" if not answered else "hard_answer",
            right=not answered,
            right_lenient=not answered,
            critical=False,
            unit_ok=True,
            abstain_ok=not answered,
            hard_answer=answered,
        )
        return out
    if not answered:
        out.update(
            grade="abstained",
            right=False,
            right_lenient=False,
            critical=False,
            unit_ok=False,
            abstain_ok=False,
            hard_answer=False,
        )
        return out
    unit_ok = _unit_matches(unit, e["unit"])
    unit_lenient = unit_ok or unit.strip() == ""
    grade = (
        grade_number(value, e["value"])
        if kind == "number"
        else grade_ladder(value, e["value"], e["ladder"])
    )
    out.update(
        grade=grade,
        right=grade == "match" and unit_ok,
        right_lenient=grade == "match" and unit_lenient,
        critical=grade == "critical",
        unit_ok=unit_ok,
        abstain_ok=False,
        hard_answer=False,
    )
    return out


def latest_good(raw: list[dict]) -> list[dict]:
    """The last non-error row per (model, condition, qid); a key with only error rows is
    logged and dropped (it would otherwise score as a wrong abstention)."""
    good: dict[tuple[str, str, str], dict] = {}
    failed: set[tuple[str, str, str]] = set()
    for r in raw:
        k = (r["model"], r["condition"], r["qid"])
        if r.get("error"):
            failed.add(k)
        else:
            good[k] = r
    lost = failed - set(good)
    if lost:
        logger.error("keys with only failed calls", extra={"ctx": {"n": len(lost)}})
    return list(good.values())


def score_rows(raw: list[dict], questions: dict[str, dict]) -> list[dict]:
    out = []
    for r in latest_good(raw):
        q = questions[r["qid"]]
        flags = judge(r.get("content", "") or "", q)
        out.append(
            {
                **r,
                "kind": q["expect"]["kind"],
                "cluster": q["cluster"],
                "rows_all_hit": r.get("rows_total", 0) > 0
                and r.get("rows_hit", 0) == r.get("rows_total", 0),
                "expr_ok": bool(r.get("expr_ok")),
                **flags,
            }
        )
    return out


def _mean_row(model: str, condition: str, chain: str, metric: str, sub: list[dict]) -> dict:
    vals = [float(bool(r[metric])) for r in sub]
    m, lo, hi = bootstrap_ci(vals)
    return {
        "model": model,
        "condition": condition,
        "chain": chain,
        "metric": metric,
        "n": len(sub),
        "mean": round(m, 4),
        "lo": round(lo, 4),
        "hi": round(hi, 4),
    }


def _scope_rows(rows: list[dict], metric: str) -> list[dict]:
    """The rows a metric is defined on: abstain metrics on abstain questions, critical on
    answerable ladder questions, the rest on answerable questions."""
    if metric in ("abstain_ok", "hard_answer"):
        return [r for r in rows if r["kind"] == "abstain"]
    if metric == "critical":
        return [r for r in rows if r["kind"] == "ladder"]
    if metric in ("parse_ok", "expr_ok"):
        return rows
    if (
        metric == "rows_all_hit"
    ):  # abstain questions' gold rows are blanks or reasons: not retrievable
        return [r for r in rows if r["kind"] != "abstain"]
    return [r for r in rows if r["kind"] != "abstain"]


def cluster_diff_ci(
    diffs: dict[tuple[str, str], list[float]], *, n: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Resample clusters within each chain (stratified), concatenate, mean."""
    by_chain: dict[str, list[np.ndarray]] = defaultdict(list)
    for (chain, _), v in diffs.items():
        by_chain[chain].append(np.asarray(v, dtype=float))
    allq = np.concatenate([x for v in by_chain.values() for x in v])
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        picks = []
        for clusters in by_chain.values():
            idx = rng.integers(0, len(clusters), size=len(clusters))
            picks += [clusters[j] for j in idx]
        means[i] = np.concatenate(picks).mean()
    return float(allq.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(scored: list[dict]) -> tuple[list[dict], list[dict]]:
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in scored:
        cells[(r["model"], r["condition"])].append(r)
    summary = []
    for (model, cond), rows in sorted(cells.items()):
        for chain in ["all", *CHAINS]:
            sub_c = rows if chain == "all" else [r for r in rows if r["chain"] == chain]
            for m in METRICS:
                sub = _scope_rows(sub_c, m)
                if sub:
                    summary.append(_mean_row(model, cond, chain, m, sub))
    diffs = []
    models = sorted({m for m, _ in cells})
    for model in models:
        pairs = [(a, b, s, True) for a, b, s in PRIMARY]
        pairs += [(a, b, s, False) for a, b, s in SECONDARY]
        for a in CONDITIONS:
            for b in CONDITIONS:
                if a != b and (a, b) in {(x, y) for x, y, _ in PRIMARY}:
                    pairs += [(a, b, c, False) for c in CHAINS]
        for a, b, scope, primary in pairs:
            ra, rb = cells.get((model, a)), cells.get((model, b))
            if not ra or not rb:
                continue
            chains = CHAINS if scope == "all" else scope.split("+")
            qa = {r["qid"]: r for r in ra if r["chain"] in chains}
            qb = {r["qid"]: r for r in rb if r["chain"] in chains}
            for metric in ("right", "right_lenient", "critical"):
                common = sorted(
                    q
                    for q in qa
                    if q in qb
                    and qa[q]["kind"] != "abstain"
                    and (metric != "critical" or qa[q]["kind"] == "ladder")
                )
                if not common:
                    continue
                d, lo, hi = paired_diff_ci(
                    [float(qa[q][metric]) for q in common], [float(qb[q][metric]) for q in common]
                )
                by_cluster: dict[tuple[str, str], list[float]] = defaultdict(list)
                for q in common:
                    by_cluster[(qa[q]["chain"], qa[q]["cluster"])].append(
                        float(qa[q][metric]) - float(qb[q][metric])
                    )
                dc, loc, hic = cluster_diff_ci(by_cluster)
                diffs.append(
                    {
                        "model": model,
                        "a": a,
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


def main(argv: list[str]) -> int:
    import argparse

    from rag_lab.log import setup_logging

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RESULTS / "raw.jsonl")
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    args = ap.parse_args(argv)
    setup_logging("rag_lab")
    questions = {
        json.loads(ln)["id"]: json.loads(ln)
        for ln in args.questions.read_text("utf-8").splitlines()
        if ln.strip()
    }
    raw = [json.loads(ln) for ln in args.raw.read_text("utf-8").splitlines() if ln.strip()]
    scored = score_rows(raw, questions)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scored.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in scored), "utf-8"
    )
    summary, diffs = summarize(scored)
    _write_csv(args.out / "summary.csv", summary)
    if diffs:
        _write_csv(args.out / "diffs.csv", diffs)
    logger.info("scored", extra={"ctx": {"rows": len(scored), "out": str(args.out)}})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
