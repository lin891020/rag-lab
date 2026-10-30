"""Templates × seeded parameters -> questions.jsonl (160: 32 per chain, about a fifth of each
chain's questions have "abstain" as the right answer because the cell is blank, the key is off
the table, or a precondition fails). Every question carries the chain's intermediate values and
gold rows so score.py / attribute.py need no re-computation.

    uv run python experiments/16-rule-lookup-compute/build_questions.py            -> questions.jsonl
    uv run python experiments/16-rule-lookup-compute/build_questions.py --rewrite FILE
        FILE is the rewriter agent's jsonl ({id, question}); the wording is swapped in only when
        every number the question gives (`given`) still appears and no table / article id leaked.

Rules for the wording: give the situation only (conduit, insulation ℃, conductor kind and size,
conductor count, ambient ℃, motor hp / phase / volts, load W and place, OCPD A); never a table
or article number; one question per line; no answer or intermediate value in the text.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
OUT = HERE / "questions.jsonl"
sys.path.insert(0, str(HERE))
from rules import (  # noqa: E402
    AMPACITY_TABLES,
    TABLES,
    Answer,
    ampacity,
    corrected_ampacity,
    dwelling_other_demand,
    ground_conductor,
    lighting_demand,
    motor_branch_conductor,
    receptacle_demand,
)

logger = logging.getLogger(__name__)

SEED = 16
PER_CHAIN = 32
ABSTAIN_PER_CHAIN = 6
MAX_PER_ROW = 2  # cluster = the chain's first gold row (C4: the tier the load reaches)
MAX_PER_ROW_C4 = 4  # 表五二/五三/六○ have 2–3 tiers per place: four loads per tier
MAX_PER_ROW_ABSTAIN = 3  # abstain questions share a reason; three of a kind is the cap
CONDUIT_TEXT = {"金屬管": "金屬導線管", "PVC管": "PVC 導線管"}
PLACE_TEXT = {
    "住宅": "一棟住宅",
    "醫院註": "一家醫院（非手術室等全部照明同時使用的區域）",
    "飯店、旅館及汽車旅館，包括不提供房客烹飪用電器具之公寓式房屋註": "一家旅館（不含房客烹飪用電器具）",
    "大賣場(倉儲)": "一家倉儲型大賣場",
    "其他": "一間辦公大樓",
}
_LEAK = re.compile(r"表[一二三四五六七八九十○〇～]+|第\s*[一二三四五六七八九十百\d]+\s*條")
# a clause the independent checker asked for: 表五二 reads as tiered ("…負載部分"), but a flat
# reading (whole load × top band) is a second plausible one; the question says which
CLAUSES = {
    "lighting": "需量因數按負載分段累進適用"
}  # not on hospital-na: there the right answer is 不得適用


def _kind_text(kind: str, size: str) -> str:
    return f"絞線、標稱截面積 {size} mm²" if kind == "絞線" else f"單線、直徑 {size} mm"


def with_clause(text: str, variant: str) -> str:
    clause = CLAUSES.get(variant)
    if not clause or clause in text:
        return text
    return text.rstrip("？?") + f"（{clause}）？"


def _q(qid: str, chain: str, variant: str, text: str, given: dict, a: Answer) -> dict:
    text = with_clause(text, variant)
    return {
        "id": qid,
        "chain": chain,
        "variant": variant,
        "question": text,
        "given": given,
        "expect": {"kind": a.kind, "value": a.value, "unit": a.unit, "ladder": a.ladder},
        "gold": {"articles": a.gold_articles, "rows": a.gold_rows},
        "inputs": a.inputs,
        "abstain_reason": a.reason,
        "cluster": f"{a.gold_rows[-1][0]}|{a.gold_rows[-1][1]}"
        if chain == "C4" and a.gold_rows
        else (f"{a.gold_rows[0][0]}|{a.gold_rows[0][1]}" if a.gold_rows else chain),
    }


def gen_c1(rng: random.Random) -> list[dict]:
    out = []
    for (conduit, ins), table in AMPACITY_TABLES.items():
        for r in TABLES[table]["rows"]:
            kind, size = r[0], r[1] or r[2]
            for n in (3, 4, 5, 6, 7, 8, 9):
                a = ampacity(conduit=conduit, insulation=ins, kind=kind, size=size, conductors=n)
                text = f"{CONDUIT_TEXT[conduit]}配線，導線絕緣物最高容許溫度 {ins}℃，{_kind_text(kind, size)}，同一管內 {n} 條載流導線，周圍溫度 35℃。這條導線的安培容量是多少 A？"
                out.append(
                    _q(
                        "",
                        "C1",
                        "cell" if a.kind != "abstain" else "blank",
                        text,
                        {"insulation": ins, "size": float(size), "conductors": n},
                        a,
                    )
                )
    for conduit, ins in AMPACITY_TABLES:  # sizes the table does not list
        for size in ("24", "45", "70", "180"):
            a = ampacity(conduit=conduit, insulation=ins, kind="絞線", size=size, conductors=3)
            text = f"{CONDUIT_TEXT[conduit]}配線，導線絕緣物最高容許溫度 {ins}℃，絞線、標稱截面積 {size} mm²，同一管內 3 條載流導線，周圍溫度 35℃。這條導線的安培容量是多少 A？"
            out.append(
                _q(
                    "",
                    "C1",
                    "offtable",
                    text,
                    {"insulation": ins, "size": float(size), "conductors": 3},
                    a,
                )
            )
    return out


def gen_c2(rng: random.Random) -> list[dict]:
    out = []
    ambients = [8, 12, 18, 22, 28, 38, 42, 48, 53, 58, 63, 68, 72, 78, 83, 88]
    for (conduit, ins), table in AMPACITY_TABLES.items():
        for r in TABLES[table]["rows"]:
            if r[0] != "絞線":
                continue
            kind, size = r[0], r[1]
            for amb in ambients:
                a = corrected_ampacity(
                    conduit=conduit, insulation=ins, kind=kind, size=size, conductors=3, ambient=amb
                )
                text = f"{CONDUIT_TEXT[conduit]}配線，導線絕緣物最高容許溫度 {ins}℃，{_kind_text(kind, size)}，同一管內 3 條載流導線，但周圍溫度是 {amb}℃。修正後這條導線的安培容量是多少 A？"
                out.append(
                    _q(
                        "",
                        "C2",
                        "temp" if a.kind != "abstain" else "temp-blank",
                        text,
                        {"insulation": ins, "size": float(size), "conductors": 3, "ambient": amb},
                        a,
                    )
                )
            for n in (10, 12, 15, 20, 24, 30, 36, 45):
                a = corrected_ampacity(
                    conduit=conduit, insulation=ins, kind=kind, size=size, conductors=n
                )
                text = f"{CONDUIT_TEXT[conduit]}配線，導線絕緣物最高容許溫度 {ins}℃，{_kind_text(kind, size)}，同一管內有 {n} 條載流導線，周圍溫度 35℃。修正後這條導線的安培容量是多少 A？"
                out.append(
                    _q(
                        "",
                        "C2",
                        "count",
                        text,
                        {"insulation": ins, "size": float(size), "conductors": n},
                        a,
                    )
                )
    return out


def gen_c3(rng: random.Random) -> list[dict]:
    out = []
    hps = [int(r[0]) for r in TABLES["表二五八～三"]["rows"]] + [4, 6, 8, 12, 35, 45]
    for hp in hps:
        for volts in (115, 200, 208, 220, 380, 440):
            for n in (3, 4, 6):
                a = motor_branch_conductor(hp=hp, volts=volts, conductors=n)
                text = f"一台三相鼠籠型感應電動機 {hp} hp、額定電壓 {volts} V，連續責務，分路採 PVC 導線管配線、導線絕緣物最高容許溫度 60℃、管內 {n} 條載流導線、周圍溫度 35℃。分路導線的最小絞線截面積是多少 mm²？"
                variant = (
                    "size" if a.kind != "abstain" else ("dash" if "－" in a.reason else "offtable")
                )
                out.append(
                    _q("", "C3", variant, text, {"hp": hp, "volts": volts, "conductors": n}, a)
                )
    return out


def gen_c4(rng: random.Random) -> list[dict]:
    out = []
    for place, ptext in PLACE_TEXT.items():
        for watts in (
            1200,
            2500,
            5000,
            8000,
            15000,
            20000,
            30000,
            40000,
            50000,
            60000,
            80000,
            100000,
            120000,
            150000,
            200000,
            250000,
        ):
            a = lighting_demand(place=place, watts=watts)
            text = f"{ptext}的照明負載合計 {watts:,} W。適用照明負載的幹線需量因數後，幹線的照明計算負載是多少 W？"
            out.append(_q("", "C4", "lighting", text, {"watts": watts}, a))
    for kva in (3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 45, 60, 80, 100):
        a = receptacle_demand(kva=kva)
        text = f"一處非住宅場所的插座負載合計 {kva} kVA。單獨適用插座負載的幹線需量因數（不併入照明負載），幹線的插座計算負載是多少 kVA？"
        out.append(_q("", "C4", "receptacle", text, {"kva": kva}, a))
    for kw in (4, 6, 8, 10, 12, 14, 18, 20, 25, 30, 32, 40, 45, 55, 60):
        a = dwelling_other_demand(kw=kw)
        text = f"一戶獨棟住宅，單相三線 120/240 V 供電，進屋線計算負載 100 A 以上。空調、電暖以外的其他負載合計 {kw} kW，適用獨棟住宅負載需量因數後，這部分的計算負載是多少 kW？"
        out.append(_q("", "C4", "dwelling", text, {"kw": kw}, a))
    for kw, why in (
        (25, "三相四線 220/380 V 供電"),
        (32, "進屋線計算負載只有 60 A"),
        (18, "三相 220 V 供電"),
        (40, "進屋線計算負載 80 A"),
    ):
        a = dwelling_other_demand(kw=kw)
        a.kind, a.value, a.reason = (
            "abstain",
            None,
            f"表六○ 的需量因數只適用於單相三線、計算負載 ≥ 100 A 的獨棟住宅（第 60 條）；本題{why}，不適用",
        )
        a.gold_rows = [
            ("表六○", "所有其他負載之首10 kW")
        ]  # the row a careless reader would use; the precondition is 第60條
        text = f"一戶獨棟住宅，{why}。空調、電暖以外的其他負載合計 {kw} kW，適用獨棟住宅負載需量因數後，這部分的計算負載是多少 kW？"
        out.append(_q("", "C4", "dwelling-na", text, {"kw": kw}, a))
    for watts in (
        12000,
        30000,
        45000,
        70000,
    ):  # 表五二 note: areas where all lighting may be on at once
        a = lighting_demand(place="醫院註", watts=watts)
        a.kind, a.value, a.reason = (
            "abstain",
            None,
            "表五二 註：醫院全部照明可能同時使用的區域（手術室等）不得適用需量因數",
        )
        a.gold_rows = [("表五二", "註")]  # the note is a chunk of its own in both indexes
        text = f"一家醫院的手術室區域（全部照明可能同時使用），照明負載合計 {watts:,} W。適用照明負載的幹線需量因數後，幹線的照明計算負載是多少 W？"
        out.append(_q("", "C4", "hospital-na", text, {"watts": watts}, a))
    return out


def gen_c5(rng: random.Random) -> list[dict]:
    out = []
    for amps in (
        15,
        20,
        25,
        30,
        40,
        50,
        60,
        75,
        100,
        125,
        150,
        175,
        200,
        225,
        250,
        300,
        350,
        400,
        450,
        500,
        550,
        600,
        650,
        700,
        800,
        900,
        1000,
        1100,
        1200,
        1400,
        1600,
        1800,
        2000,
        2200,
        2500,
        2800,
        3000,
        3500,
        4000,
        4500,
        5000,
        5500,
        6300,
        7000,
        8000,
    ):
        for kind in ("絞線", "單線"):
            a = ground_conductor(ocpd_a=amps, kind=kind)
            unit = "mm²" if kind == "絞線" else "mm"
            text = f"一組用電設備單獨接地，設備接地導線用銅{kind}，電路的過電流保護裝置額定 {amps:,} A。設備接地導線的最小線徑是多少 {unit}？"
            variant = "size" if a.kind != "abstain" else ("dash" if "-" in a.reason else "offtable")
            out.append(_q("", "C5", variant, text, {"ocpd": amps}, a))
    return out


def pick(pool: list[dict], rng: random.Random, *, n: int, n_abstain: int) -> list[dict]:
    """n questions: n_abstain with kind abstain, the rest answerable; at most MAX_PER_ROW per cluster."""
    rng.shuffle(pool)
    chosen: list[dict] = []
    per_cluster: Counter[str] = Counter()
    for want_abstain, k in ((True, n_abstain), (False, n - n_abstain)):
        got = 0
        for q in pool:
            if got >= k:
                break
            cap = (
                MAX_PER_ROW_ABSTAIN
                if want_abstain
                else (MAX_PER_ROW_C4 if q["chain"] == "C4" else MAX_PER_ROW)
            )
            if (
                (q["expect"]["kind"] == "abstain") != want_abstain
                or per_cluster[q["cluster"]] >= cap
                or q in chosen
            ):
                continue
            chosen.append(q)
            per_cluster[q["cluster"]] += 1
            got += 1
        assert got == k, f"pool too small: wanted {k} abstain={want_abstain}, got {got}"
    return chosen


def build() -> list[dict]:
    rng = random.Random(SEED)
    out: list[dict] = []
    for chain, gen in (
        ("C1", gen_c1),
        ("C2", gen_c2),
        ("C3", gen_c3),
        ("C4", gen_c4),
        ("C5", gen_c5),
    ):
        qs = pick(gen(rng), rng, n=PER_CHAIN, n_abstain=ABSTAIN_PER_CHAIN)
        for i, q in enumerate(qs, 1):
            q["id"] = f"{chain.lower()}-{i:02d}"
        out += qs
    return out


def rewrite(questions: list[dict], path: Path) -> tuple[list[dict], list[str]]:
    """Swap in the rewriter's wording where it keeps every given number and leaks nothing."""
    new = {
        json.loads(ln)["id"]: json.loads(ln)["question"]
        for ln in path.read_text("utf-8").splitlines()
        if ln.strip()
    }
    rejected = []
    for q in questions:
        text = new.get(q["id"])
        if text is None:
            continue
        ok = all(_number_present(text, v) for v in q["given"].values()) and not _LEAK.search(text)
        if ok:
            q["question"] = with_clause(text, q["variant"])
        else:
            rejected.append(q["id"])
    return questions, rejected


def _number_present(text: str, v: float) -> bool:
    forms = {f"{v:g}", f"{v:,}", f"{int(v):,}" if v == int(v) else "", str(v)}
    return any(f and f in text.replace(" ", "") for f in forms)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrite", type=Path)
    args = ap.parse_args(argv)
    setup_logging("rag_lab")
    if args.rewrite:
        questions = [json.loads(ln) for ln in OUT.read_text("utf-8").splitlines() if ln.strip()]
        questions, rejected = rewrite(questions, args.rewrite)
        logger.info("rewritten", extra={"ctx": {"rejected": rejected}})
    else:
        questions = build()
    OUT.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions), "utf-8")
    kinds = Counter((q["chain"], q["expect"]["kind"]) for q in questions)
    logger.info(
        "questions",
        extra={
            "ctx": {
                "n": len(questions),
                "kinds": {f"{c}/{k}": v for (c, k), v in sorted(kinds.items())},
            }
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
