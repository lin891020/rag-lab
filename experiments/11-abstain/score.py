"""Score results/read/raw.jsonl -> scored.jsonl, summary.csv, diffs.csv. Program logic only.

    uv run python experiments/11-abstain/run_read.py
    uv run python experiments/11-abstain/score.py

10's judge (strict value_ok, lenient value_lenient) plus, per reply:
  answered    the model gave an answer (not abstained):
                P1  found is true
                P0  value not 0 / null, articles not empty, law not empty, and the quote is
                    not a refusal phrase (無明確規定、找不到…; a lone 無 is not one)
  grounded    the model's quote (≥ 4 chars, whitespace removed, NFKC) occurs in the evidence
  answered_f / value_ok_f   after the program filter: an answer whose quote is not grounded
                            is turned into an abstention
  lucky       right answer without the gold in the evidence (knowledge or guess, not reading)
Rows are grouped: answerable = gold, or real with has_gold; unanswerable = same-doc, other-doc,
none, or real without has_gold. Leaks (an answer-free window that does hold the answer) are
dropped from the summary and counted.
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
QUESTIONS = HERE.parent / "09-messy-real" / "questions.jsonl"
TOL = 1e-6
METRICS = ["parse_ok", "schema_ok", "value_ok", "value_lenient", "value_only_ok"]
TYPES = ["value", "articles", "which"]
PROMPTS = ["P0", "P1"]
KINDS = ["gold", "same-doc", "other-doc", "none", "real"]
GROUP_METRICS = {
    "unanswerable": ["answered", "answered_f", "lucky", "grounded"],
    "answerable": ["answered", "value_ok", "value_lenient", "value_ok_f", "grounded"],
}
# whole refusal phrases only: a lone 無／未 is everywhere in legal text (無工作能力, 未滿四十歲)
_REFUSE = re.compile(
    r"無明確|無相關|無資料|無此|無規定|沒有(?:提|規定|說明|相關|答案|資料|明確)|未(?:提及|載明|規定|說明)"
    r"|找不到|不適用|無法(?:判斷|確定|找到|回答|得知)|資料不足|依據(?:中|裡)?(?:沒有|無)"
)

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


def abstained(obj: dict, qtype: str, prompt: str) -> bool:
    """P1: the found flag. P0 (no flag): an empty / zero answer or a refusing quote."""
    if prompt == "P1":
        return obj.get("found") is not True
    quote = obj.get("quote")
    if isinstance(quote, str) and _REFUSE.search(quote):
        return True
    if qtype == "value":
        v = obj.get("value")
        return v is None or (_is_number(v) and float(v) == 0)
    if qtype == "articles":
        return not obj.get("articles")
    return not (isinstance(obj.get("law"), str) and obj["law"].strip())


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))


def grounded(quote: object, evidence: str | None) -> bool:
    if not evidence or not isinstance(quote, str):
        return False
    q = _norm(quote)
    return len(q) >= 4 and q in _norm(evidence)


def group_of(evidence: str, has_gold: bool) -> str:
    if evidence == "gold" or (evidence == "real" and has_gold):
        return "answerable"
    return "unanswerable"


def score_rows(raw: list[dict], questions: dict[str, dict]) -> list[dict]:
    out = []
    for r in raw:
        q = questions[r["qid"]]
        content = r.get("content", "") or ""
        flags = judge(content, q["type"], q["expect"])
        obj, _, _ = _parse(content)
        obj = obj if isinstance(obj, dict) else {}
        answered = bool(obj) and not abstained(obj, q["type"], r["prompt"])
        g = grounded(obj.get("quote"), r.get("evidence_text"))
        group = group_of(r["evidence"], bool(r.get("has_gold")))
        value_ok = flags["value_ok"] and answered
        value_lenient = flags["value_lenient"] and answered
        out.append(
            {
                **r,
                "type": q["type"],
                "group": group,
                **flags,
                "value_ok": value_ok,
                "value_lenient": value_lenient,
                "answered": answered,
                "grounded": g,
                "answered_f": answered and g,
                "value_ok_f": value_ok and g,
                "lucky": value_ok and group == "unanswerable",
            }
        )
    return out


def _mean_row(model, prompt, group, kind, t, metric, sub) -> dict:
    mean, lo, hi = bootstrap_ci([float(r[metric]) for r in sub])
    return {
        "model": model,
        "prompt": prompt,
        "group": group,
        "evidence": kind,
        "type": t,
        "metric": metric,
        "mean": mean,
        "lo": lo,
        "hi": hi,
        "n": len(sub),
    }


def summarize(scored: list[dict]) -> tuple[list[dict], list[dict]]:
    """summary: per (model, prompt, group, evidence|all, type|all, metric); leaks dropped.
    diffs: P1 − P0 per (group, metric); filtered − unfiltered per (prompt, group)."""
    kept = [r for r in scored if not r.get("leak")]
    cells: dict[tuple, list[dict]] = {}
    for r in kept:
        cells.setdefault((r["model"], r["prompt"], r["group"]), []).append(r)
    summary, diffs = [], []
    for (model, prompt, group), rows in sorted(cells.items()):
        kinds = ["all", *[k for k in KINDS if any(r["evidence"] == k for r in rows)]]
        for kind in kinds:
            sub_k = rows if kind == "all" else [r for r in rows if r["evidence"] == kind]
            for t in ["all", *TYPES]:
                sub = sub_k if t == "all" else [r for r in sub_k if r["type"] == t]
                if not sub:
                    continue
                for m in GROUP_METRICS[group]:
                    summary.append(_mean_row(model, prompt, group, kind, t, m, sub))
    models = sorted({m for m, _, _ in cells})
    for model in models:
        for group, metrics in GROUP_METRICS.items():
            a, b = cells.get((model, "P1", group)), cells.get((model, "P0", group))
            if a and b:
                qa = {(r["evidence"], r["qid"]): r for r in a}
                qb = {(r["evidence"], r["qid"]): r for r in b}
                common = sorted(set(qa) & set(qb))
                for m in metrics:
                    if m.endswith("_f"):
                        continue
                    d, lo, hi = paired_diff_ci(
                        [float(qa[k][m]) for k in common], [float(qb[k][m]) for k in common]
                    )
                    diffs.append(_diff_row(model, group, m, "P1", "P0", d, lo, hi, len(common)))
            for prompt in PROMPTS:
                rows = cells.get((model, prompt, group))
                if not rows:
                    continue
                for m in metrics:
                    if not m.endswith("_f"):
                        continue
                    base = m[:-2]
                    d, lo, hi = paired_diff_ci(
                        [float(r[m]) for r in rows], [float(r[base]) for r in rows]
                    )
                    diffs.append(
                        _diff_row(
                            model, group, m, "filtered", "unfiltered", d, lo, hi, len(rows), prompt
                        )
                    )
    return summary, diffs


def _diff_row(model, group, metric, a, b, d, lo, hi, n, prompt="") -> dict:
    return {
        "model": model,
        "prompt": prompt,
        "group": group,
        "metric": metric,
        "a": a,
        "b": b,
        "diff": d,
        "lo": lo,
        "hi": hi,
        "n": n,
        "separable": lo > 0 or hi < 0,
    }


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
    args = ap.parse_args(argv)
    setup_logging("rag_lab")
    questions = {
        json.loads(ln)["id"]: json.loads(ln)
        for ln in QUESTIONS.read_text("utf-8").splitlines()
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
    leaks = sum(1 for r in scored if r.get("leak"))
    logger.info("scored", extra={"ctx": {"rows": len(scored), "leaks": leaks}})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
