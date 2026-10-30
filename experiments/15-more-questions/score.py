"""Score results/read/raw.jsonl -> scored.jsonl, summary.csv, diffs.csv. Program logic only.

    uv run python experiments/09-messy-real/run_read.py
    uv run python experiments/09-messy-real/score.py

Flags per reply:
  parse_ok    json.loads succeeds after stripping a ``` fence
  schema_ok   the object has the keys its type needs, with the right JSON types
  value_ok    the answer is right:
                value     number within 1e-6 of the gold (or an alt) AND the unit matches (06's rule)
                articles  the set of 條號 equals the gold set (each item folded: 第十二條之一 -> 12-1)
                which     the law name equals the gold (or an alt) after folding, or contains it
  value_only_ok  value type only: number right, unit wrong (reported, not used)
Summary per (model, condition) over all questions and per type; paired diffs of every pair of
conditions per model (value_ok).
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

sys.path.insert(0, str(Path(__file__).parent))
from chunkers import cn_to_int  # noqa: E402

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
RESULTS = HERE / "results/read"
QUESTIONS = HERE / "questions-gazette.jsonl"
TOL = 1e-6
METRICS = ["parse_ok", "schema_ok", "value_ok", "value_lenient", "value_only_ok"]
CONDITIONS = [
    "closed-book",
    "oracle",
    "fixed",
    "+ocr",
    "+rows",
    "+prefix",
    "+cnpre",
    "full",
    "-ocr",
    "-rows",
    "-prefix",
    "-pre",
    "-cnpre",
]
# add-one: each fix − fixed; drop-one: full − each; anchors: the ladder itself
ABLATION: dict[str, tuple[str, list[str]]] = {
    "add": ("fixed", ["+ocr", "+rows", "+prefix", "+cnpre"]),
    "drop": ("full", ["-ocr", "-rows", "-prefix", "-pre", "-cnpre"]),
}
ANCHORS = [("fixed", "closed-book"), ("full", "fixed"), ("oracle", "full")]
TYPES = ["value", "articles", "which"]
NUMERALS = ["cn", "wan", "arabic"]
DIFF_METRICS = ["value_ok", "value_lenient"]

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
    # units the second bank (questions-gazette.jsonl) uses that 10's table lacked; added only
    "箱": ["箱", "box", "boxes"],
    "公升": ["公升", "升", "l", "L", "liter", "litre", "公升/日"],
    "份": ["份", "copies", "copy"],
    "小時": ["小時", "時", "hr", "h", "hour", "hours", "個小時"],
    "公頃": ["公頃", "ha", "hectare", "hectares"],
    "學分": ["學分", "credit", "credits"],
    "立方公尺": ["立方公尺", "立方米", "m³", "m3", "m^3", "cubic meter", "cubic metre"],
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


_ART = re.compile(
    r"^第?\s*([0-9]+|[零〇一二三四五六七八九十百]+)\s*條?(?:之\s*([0-9]+|[零〇一二三四五六七八九十百]+))?\s*(?:條|點)?$"
)


def norm_article(s: str) -> str:
    """'第十二條之一' / '12之1' / '12-1' / '第4點' -> '12-1' / '4'; unknown strings fold to
    themselves with spaces removed."""
    s = unicodedata.normalize("NFKC", str(s)).strip().replace(" ", "")
    if "-" in s and s.replace("-", "").isdigit():
        return s
    m = _ART.match(s)
    if not m:
        return s

    def num(x: str) -> int:
        return int(x) if x.isdigit() else cn_to_int(x)

    out = str(num(m.group(1)))
    return f"{out}-{num(m.group(2))}" if m.group(2) else out


def norm_law(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"[\s「」『』《》()（）]", "", s)


def judge_articles(obj: dict, expect: dict) -> dict[str, bool]:
    got = obj.get("articles")
    ok = isinstance(got, list) and all(isinstance(x, (str, int)) for x in got)
    schema_ok = isinstance(obj.get("quote"), str) and ok
    value_ok = ok and {norm_article(x) for x in got} == {
        norm_article(x) for x in expect["articles"]
    }
    return {
        "schema_ok": schema_ok,
        "value_ok": bool(value_ok),
        "value_lenient": bool(value_ok),
        "value_only_ok": False,
    }


def judge_which(obj: dict, expect: dict) -> dict[str, bool]:
    got = obj.get("law")
    schema_ok = isinstance(obj.get("quote"), str) and isinstance(got, str)
    names = [expect["law"], *expect.get("alt", [])]
    g = norm_law(got) if isinstance(got, str) else ""
    value_ok = bool(g) and any(norm_law(n) == g or norm_law(n) in g for n in names)
    return {
        "schema_ok": schema_ok,
        "value_ok": value_ok,
        "value_lenient": value_ok,
        "value_only_ok": False,
    }


def judge_value(obj: dict, expect: dict) -> dict[str, bool]:
    """value_ok: number and unit (synonyms allowed). value_lenient: number right and the unit
    either right or left empty (the convention of GAIA / SLMJury: units are not scored).
    value_only_ok: number right, unit wrong or empty (06's flag, kept for comparison)."""
    out = {
        "schema_ok": (
            isinstance(obj.get("quote"), str)
            and _is_number(obj.get("value"))
            and isinstance(obj.get("unit"), str)
        ),
        "value_ok": False,
        "value_lenient": False,
        "value_only_ok": False,
    }
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
                out["value_lenient"] = True
                return out
            out["value_only_ok"] = True
            if not _fold(unit):
                out["value_lenient"] = True
    return out


_CN_DIGITS = "零〇○一二三四五六七八九十百千萬"


def numeral_kind(quote: str, value: float) -> str:
    """How the gold number is written in the quote: 'arabic' when the digits (commas allowed)
    appear in it, else 'wan' when a Chinese numeral with 萬 does, else 'cn'."""
    q = unicodedata.normalize("NFKC", quote)
    v = float(value)
    forms = {f"{v:g}", f"{int(v):,}" if v == int(v) else f"{v:g}"}
    if any(f in q.replace(" ", "") for f in forms):
        return "arabic"
    runs = re.findall(f"[{_CN_DIGITS}]+", q)
    return "wan" if any("萬" in r for r in runs) else "cn"


def judge(content: str, qtype: str, expect: dict) -> dict[str, bool]:
    obj, parse_ok, fenced = _parse(content)
    out = {
        "parse_ok": parse_ok,
        "schema_ok": False,
        "value_ok": False,
        "value_lenient": False,
        "value_only_ok": False,
        "fenced": fenced,
    }
    if not isinstance(obj, dict):
        return out
    fn = {"value": judge_value, "articles": judge_articles, "which": judge_which}[qtype]
    out.update(fn(obj, expect))
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


def score_rows(raw: list[dict], questions: dict[str, dict]) -> list[dict]:
    out = []
    for r in raw:
        q = questions[r["qid"]]
        flags = judge(r.get("content", "") or "", q["type"], q["expect"])
        numeral = numeral_kind(q["quote"], q["expect"]["value"]) if q["type"] == "value" else ""
        out.append({**r, "type": q["type"], "numeral": numeral, **flags})
    return out


def summarize(scored: list[dict]) -> tuple[list[dict], list[dict]]:
    """summary rows per (model, condition, type∈{all, value, articles, which}, metric); paired
    diffs of every pair of conditions per model on value_ok (all questions)."""
    cells: dict[tuple[str, str], list[dict]] = {}
    for r in scored:
        cells.setdefault((r["model"], r["condition"]), []).append(r)
    summary, diffs = [], []
    for (model, cond), rows in sorted(cells.items()):
        rows.sort(key=lambda r: r["qid"])
        for t in ["all", *TYPES, *(f"value:{k}" for k in NUMERALS)]:
            if t == "all":
                sub = rows
            elif t.startswith("value:"):
                sub = [r for r in rows if r["type"] == "value" and r.get("numeral") == t[6:]]
            else:
                sub = [r for r in rows if r["type"] == t]
            if not sub:
                continue
            for m in METRICS:
                mean, lo, hi = bootstrap_ci([float(r[m]) for r in sub])
                summary.append(
                    {
                        "model": model,
                        "condition": cond,
                        "type": t,
                        "metric": m,
                        "mean": mean,
                        "lo": lo,
                        "hi": hi,
                        "n": len(sub),
                    }
                )
            if t == "all":
                summary.append(
                    {
                        "model": model,
                        "condition": cond,
                        "type": t,
                        "metric": "evidence_has_gold",
                        "mean": float(
                            np.mean([float(r.get("evidence_has_gold", False)) for r in sub])
                        ),
                        "lo": "",
                        "hi": "",
                        "n": len(sub),
                    }
                )
                summary.append(
                    {
                        "model": model,
                        "condition": cond,
                        "type": t,
                        "metric": "eval_count",
                        "mean": float(np.mean([r.get("eval_count", 0) for r in sub])),
                        "lo": "",
                        "hi": "",
                        "n": len(sub),
                    }
                )
                summary.append(
                    {
                        "model": model,
                        "condition": cond,
                        "type": t,
                        "metric": "seconds",
                        "mean": float(np.mean([r.get("duration", 0.0) for r in sub])),
                        "lo": "",
                        "hi": "",
                        "n": len(sub),
                    }
                )
    models = sorted({m for m, _ in cells})
    pairs: list[tuple[str, str, str]] = []  # (group, a, b) meaning a − b
    ref, fixes = ABLATION["add"]
    pairs += [("add", f, ref) for f in fixes]
    ref, drops = ABLATION["drop"]
    pairs += [("drop", ref, d) for d in drops]
    pairs += [("anchor", a, b) for a, b in ANCHORS]
    for model in models:
        for group, a, b in pairs:
            ra, rb = cells.get((model, a)), cells.get((model, b))
            if not ra or not rb:
                continue
            qa = {r["qid"]: r for r in ra}
            qb = {r["qid"]: r for r in rb}
            common = sorted(set(qa) & set(qb))
            for m in DIFF_METRICS:
                d, lo, hi = paired_diff_ci(
                    [float(qa[q][m]) for q in common], [float(qb[q][m]) for q in common]
                )
                diffs.append(
                    {
                        "model": model,
                        "group": group,
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


def main(argv: list[str]) -> int:
    """--raw/--out default to this experiment's results; point --raw at 09's raw.jsonl and --out
    at results/09-rescored to re-score 09 with this ruler."""
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
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in scored), encoding="utf-8"
    )
    summary, diffs = summarize(scored)
    _write_csv(args.out / "summary.csv", summary)
    if diffs:
        _write_csv(args.out / "diffs.csv", diffs)
    logger.info("scored", extra={"ctx": {"rows": len(scored), "out": str(args.out)}})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
