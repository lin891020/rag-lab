"""Score results/raw.jsonl -> results/summary.csv, diffs.csv. Pure program logic, no LLM judge.

    uv run python experiments/03-table-format/run.py      # produces raw.jsonl + meta.json
    uv run python experiments/03-table-format/score.py    # this file

Four flags per reply (see README section 2):
  parse_ok       json.loads succeeds after stripping a ``` fence
  schema_ok      an object with quote:str, value:number (not bool), unit:str
  value_ok       value equals the gold (or an `alt` pair) within 1e-6 AND the unit matches
  value_only_ok  value equals the gold but the unit does not (reported, not used in conclusions)
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
RESULTS = HERE / "results"
TOL = 1e-6
METRICS = ["parse_ok", "schema_ok", "value_ok", "value_only_ok"]
FORMATS = [
    "markdown",
    "html",
    "kv",
    "flat",
]  # diffs: every pair, flat last so "x − flat" reads naturally

# canonical unit -> spellings a model may use (compared after normalize_unit on both sides)
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
    return _fold(got) in {_fold(a) for a in [want, *UNIT_ALIASES.get(want, [])]}


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


def judge(content: str, expect: dict) -> dict[str, bool]:
    obj, parse_ok, fenced = _parse(content)
    out = {
        "parse_ok": parse_ok,
        "schema_ok": False,
        "value_ok": False,
        "value_only_ok": False,
        "fenced": fenced,
    }
    if not isinstance(obj, dict):
        return out
    out["schema_ok"] = (
        isinstance(obj.get("quote"), str)
        and _is_number(obj.get("value"))
        and isinstance(obj.get("unit"), str)
    )
    v = _to_float(obj.get("value"))
    if v is None:
        return out
    unit = obj.get("unit", "")
    unit = unit if isinstance(unit, str) else str(unit)
    pairs = [(expect["value"], expect["unit"])] + [
        (a["value"], a["unit"]) for a in expect.get("alt", [])
    ]
    for want_v, want_u in pairs:
        if abs(v - float(want_v)) <= TOL:
            if _unit_matches(unit, want_u):
                out["value_ok"] = True
                return out
            out["value_only_ok"] = True
    return out


def bootstrap_ci(
    values: list[float], *, n: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """(mean, lo95, hi95) by resampling the questions. Copied from 01 on purpose (no sharing)."""
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff_ci(
    a: list[float], b: list[float], *, n: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    d = [float(x) - float(y) for x, y in zip(a, b, strict=True)]
    return bootstrap_ci(d, n=n, seed=seed)


def score_rows(raw: list[dict], questions: dict[str, dict]) -> list[dict]:
    out = []
    for r in raw:
        flags = judge(r.get("content", "") or "", questions[r["qid"]]["expect"])
        out.append({**r, **flags})
    return out


def summarize(scored: list[dict]) -> tuple[list[dict], list[dict]]:
    """summary rows (model, condition, metric, mean, lo, hi, n, eval_mean, sec_mean) and paired diffs."""
    cells: dict[tuple[str, str], list[dict]] = {}
    for r in scored:
        cells.setdefault((r["model"], r["condition"]), []).append(r)
    summary, diffs = [], []
    for (model, cond), rows in sorted(cells.items()):
        rows.sort(key=lambda r: r["qid"])
        for m in METRICS:
            mean, lo, hi = bootstrap_ci([float(r[m]) for r in rows])
            summary.append(
                {
                    "model": model,
                    "condition": cond,
                    "metric": m,
                    "mean": mean,
                    "lo": lo,
                    "hi": hi,
                    "n": len(rows),
                }
            )
        summary.append(
            {
                "model": model,
                "condition": cond,
                "metric": "eval_count",
                "mean": float(np.mean([r.get("eval_count", 0) for r in rows])),
                "lo": "",
                "hi": "",
                "n": len(rows),
            }
        )
        summary.append(
            {
                "model": model,
                "condition": cond,
                "metric": "seconds",
                "mean": float(np.mean([r.get("duration", 0.0) for r in rows])),
                "lo": "",
                "hi": "",
                "n": len(rows),
            }
        )
    models = sorted({m for m, _ in cells})
    formats = [c for c in FORMATS if any((m, c) in cells for m in models)]
    for model in models:
        for i, a in enumerate(formats):
            for b in formats[i + 1 :]:
                ra, rb = cells.get((model, a)), cells.get((model, b))
                if not ra or not rb:
                    continue
                qa = {r["qid"]: r for r in ra}
                qb = {r["qid"]: r for r in rb}
                common = sorted(set(qa) & set(qb))
                for m in METRICS[:3]:
                    d, lo, hi = paired_diff_ci(
                        [float(qa[q][m]) for q in common], [float(qb[q][m]) for q in common]
                    )
                    diffs.append(
                        {
                            "model": model,
                            "metric": m,
                            "a": a,
                            "b": b,
                            "diff": d,
                            "lo": lo,
                            "hi": hi,
                            "separable": lo > 0 or hi < 0,
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
    questions = {
        json.loads(ln)["id"]: json.loads(ln)
        for ln in (HERE / "questions.jsonl").read_text("utf-8").splitlines()
        if ln.strip()
    }
    raw = [
        json.loads(ln)
        for ln in (RESULTS / "raw.jsonl").read_text("utf-8").splitlines()
        if ln.strip()
    ]
    scored = score_rows(raw, questions)
    (RESULTS / "scored.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in scored), encoding="utf-8"
    )
    summary, diffs = summarize(scored)
    _write_csv(RESULTS / "summary.csv", summary)
    _write_csv(RESULTS / "diffs.csv", diffs)
    logger.info(
        "scored", extra={"ctx": {"rows": len(scored), "cells": len(summary) // (len(METRICS) + 2)}}
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
