"""The five chains, computed from the clean tables, with every intermediate value kept so the
scorer can grade on a ladder and the attribution can tell "wrong row" from "wrong value" from
"wrong arithmetic". Program logic only; the rules are 《用戶用電設備裝置規則》:

  C1  ampacity       第25條   表二五～二/三/四 (金屬管 60/75/90℃), 表二五～五 (PVC 管 60℃)
  C2  corrected      第25條   C1 × 表二五～七 (ambient temperature) or, with >= 10 current-carrying
                              conductors, the 3以下 column × 表二五～六
  C3  motor_branch   第214條  表二五八～三 induction FLC × 1.25 -> smallest stranded size in
                              表二五～五 whose ampacity >= that, never below 3.5 mm² (第19條二)
  C4  feeder_demand  第52/53/60條  表五二 (tiered by place), 表五三 (receptacles, applied alone),
                              表六○ (single dwelling "other loads": first 10 kW 100 %, rest 40 %)
  C5  ground_conductor 第93條 表九三～二: OCPD rating -> stranded mm² (range rows; the solid column
                              is "-" above 30 A)

Every function returns an Answer. kind "number" is graded with a 1 % tolerance, "ladder" on the
table's own size ladder (match / one step up acceptable / two up error / below critical), and
"abstain" means the correct reply is found=false (the cell is blank or the key is off the table).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).parent
TABLES = json.loads((HERE / "corpus" / "tables_clean.json").read_text("utf-8"))

MOTOR_FACTOR = 1.25  # 第214條: conductor ampacity >= 1.25 x full-load current
MIN_STRANDED_MM2 = 3.5  # 第19條二 / 第214條六: stranded branch conductors never below 3.5 mm²
COUNT_FACTOR_FROM = (
    10  # 第25條: 表二五～六 applies from 10 current-carrying conductors (4–9 exempt)
)
AMPACITY_TABLES = {  # (conduit, insulation ℃) -> table
    ("金屬管", 60): "表二五～二",
    ("金屬管", 75): "表二五～三",
    ("金屬管", 90): "表二五～四",
    ("PVC管", 60): "表二五～五",
}
COUNT_COLUMNS = {
    3: 3,
    4: 4,
    5: 5,
    6: 5,
    7: 6,
    8: 6,
    9: 6,
}  # conductors -> column index in 表二五～x
INDUCTION_VOLTS = {
    "115": 1,
    "200": 2,
    "208": 3,
    "220": 4,
    "380": 5,
    "440": 6,
}  # 表二五八～三 columns
TEMP_COLUMNS = {60: 1, 75: 2, 90: 3}  # 表二五～七
Place = str


@dataclass
class Answer:
    chain: str
    kind: str  # number | ladder | abstain
    value: float | None
    unit: str
    inputs: dict = field(default_factory=dict)  # intermediate values the model should produce
    gold_articles: list[str] = field(default_factory=list)
    gold_rows: list[tuple[str, str]] = field(default_factory=list)  # (table, row key)
    ladder: list[float] | None = None
    reason: str = ""  # why abstain


def _num(cell: str) -> float | None:
    cell = cell.replace(",", "").strip()
    try:
        return float(cell)
    except ValueError:
        return None


def row_key(table: str, row: list[str]) -> str:
    """The cells that identify a row: 線別+截面積/直徑 for the ampacity tables, else the first cell
    (plus the place for 表五二)."""
    if table.startswith("表二五～") and table[-1] in "二三四五":
        return f"{row[0]} {row[1] or row[2]}"
    if table == "表五二":
        return f"{row[0]} {row[1]}"
    return row[0]


def _in_range(key: str, x: float) -> bool:
    """'10以下', '超過10', '11 - 15', '41以上', '3,001至120,000', '1 - 2', '6以上'."""
    k = key.replace(",", "").replace(" ", "")
    if k.endswith("以下"):
        return x <= float(k[:-2])
    if k.startswith("超過"):
        return x > float(k[2:])
    if k.endswith("以上"):
        return x >= float(k[:-2])
    m = re.fullmatch(r"([\d.]+)[-至]([\d.]+)", k)
    if m:
        return float(m.group(1)) <= x <= float(m.group(2))
    return k == f"{x:g}"


def ampacity_row(table: str, kind: str, size: str) -> list[str] | None:
    for r in TABLES[table]["rows"]:
        if r[0] == kind and (r[1] or r[2]) == size:
            return r
    return None


def stranded_ladder(table: str) -> list[float]:
    return [float(r[1]) for r in TABLES[table]["rows"] if r[0] == "絞線"]


# -- C1 ---------------------------------------------------------------------------------------


def ampacity(*, conduit: str, insulation: int, kind: str, size: str, conductors: int) -> Answer:
    table = AMPACITY_TABLES[(conduit, insulation)]
    row = ampacity_row(table, kind, size)
    base = Answer(
        "C1", "number", None, "A", gold_articles=["25"], gold_rows=[(table, f"{kind} {size}")]
    )
    if row is None:
        return _abstain(base, f"{table} 沒有 {kind} {size} 這一列")
    if conductors not in COUNT_COLUMNS:
        return _abstain(base, f"{conductors} 條載流導線不在表的欄位內（≥10 條要用第 25 條的修正）")
    cell = _num(row[COUNT_COLUMNS[conductors]])
    if cell is None:
        return _abstain(base, f"{table} {kind} {size} 在 {conductors} 條那一欄是空白")
    base.value, base.inputs = cell, {"ampacity": cell}
    return base


# -- C2 ---------------------------------------------------------------------------------------


def corrected_ampacity(
    *, conduit: str, insulation: int, kind: str, size: str, conductors: int, ambient: int = 35
) -> Answer:
    """第25條: ambient != 35℃ -> × 表二五～七; >= 10 conductors -> 3以下 column × 表二五～六."""
    table = AMPACITY_TABLES[(conduit, insulation)]
    row = ampacity_row(table, kind, size)
    base = Answer(
        "C2", "number", None, "A", gold_articles=["25"], gold_rows=[(table, f"{kind} {size}")]
    )
    if row is None:
        return _abstain(base, f"{table} 沒有 {kind} {size} 這一列")
    if conductors >= COUNT_FACTOR_FROM:
        cell = _num(row[3])
        if cell is None:
            return _abstain(base, f"{table} {kind} {size} 的 3以下 欄是空白")
        count_row = next(
            (r for r in TABLES["表二五～六"]["rows"] if _in_range(r[0], conductors)), None
        )
        if count_row is None:
            return _abstain(base, f"表二五～六 沒有 {conductors} 條的列")
        factor = _num(count_row[1]) / 100
        base.gold_rows.append(("表二五～六", count_row[0]))
        base.inputs = {"ampacity": cell, "count_factor": factor}
        base.value = round(cell * factor, 1)
        return base
    if conductors not in COUNT_COLUMNS:
        return _abstain(base, f"{conductors} 條載流導線不在表的欄位內")
    cell = _num(row[COUNT_COLUMNS[conductors]])
    if cell is None:
        return _abstain(base, f"{table} {kind} {size} 在 {conductors} 條那一欄是空白")
    temp_row = next((r for r in TABLES["表二五～七"]["rows"] if _in_range(r[0], ambient)), None)
    if temp_row is None:
        return _abstain(base, f"表二五～七 沒有周溫 {ambient}℃ 的列")
    factor = _num(temp_row[TEMP_COLUMNS[insulation]])
    base.gold_rows.append(("表二五～七", temp_row[0]))
    if factor is None:
        return _abstain(base, f"表二五～七 周溫 {ambient}℃ 在 {insulation}℃ 欄是空白")
    base.inputs = {"ampacity": cell, "temp_factor": factor}
    base.value = round(cell * factor, 1)
    return base


# -- C3 ---------------------------------------------------------------------------------------


def motor_branch_conductor(*, hp: int, volts: int, conductors: int = 3) -> Answer:
    """第214條 one: 1.25 × 表二五八～三 FLC (induction columns) -> smallest stranded size in 表二五～五
    (PVC conduit, 60℃) with ampacity >= that in the `conductors` column; floor 3.5 mm²."""
    base = Answer(
        "C3",
        "ladder",
        None,
        "mm²",
        gold_articles=["214", "19"],
        ladder=stranded_ladder("表二五～五"),
    )
    flc_row = next((r for r in TABLES["表二五八～三"]["rows"] if r[0] == str(hp)), None)
    base.gold_rows.append(("表二五八～三", str(hp)))
    if flc_row is None:
        return _abstain(base, f"表二五八～三 沒有 {hp} hp 這一列")
    col = INDUCTION_VOLTS.get(str(volts))
    if col is None:
        return _abstain(base, f"表二五八～三 沒有 {volts} V 這一欄")
    flc = _num(flc_row[col])
    if flc is None:
        return _abstain(base, f"表二五八～三 {hp} hp 在 {volts} V 欄是「－」")
    if conductors not in COUNT_COLUMNS:
        return _abstain(base, f"{conductors} 條載流導線不在表的欄位內")
    required = round(flc * MOTOR_FACTOR, 3)
    col_a = COUNT_COLUMNS[conductors]
    chosen = None
    for r in TABLES["表二五～五"]["rows"]:
        if r[0] != "絞線":
            continue
        a = _num(r[col_a])
        if a is not None and a >= required and float(r[1]) >= MIN_STRANDED_MM2:
            chosen = r
            break
    if chosen is None:
        return _abstain(base, f"表二五～五 沒有載流量 ≥ {required} A 的絞線列")
    base.gold_rows.append(("表二五～五", f"絞線 {chosen[1]}"))
    base.inputs = {
        "flc": flc,
        "factor": MOTOR_FACTOR,
        "required": required,
        "ampacity": _num(chosen[col_a]),
    }
    base.value = float(chosen[1])
    return base


# -- C4 ---------------------------------------------------------------------------------------


def lighting_demand(*, place: Place, watts: float) -> Answer:
    """第52條 表五二, tiered: each band's share × its factor, summed."""
    base = Answer("C4", "number", None, "W", gold_articles=["52"])
    rows = [r for r in TABLES["表五二"]["rows"] if r[0] == place]
    if not rows:  # every other place is the "其他 / 總伏安 100 %" row
        place = "其他"
        rows = [r for r in TABLES["表五二"]["rows"] if r[0] == place]
    total, remaining, lo, parts = 0.0, watts, 0.0, []
    for r in rows:
        band, pct = r[1].replace(",", ""), _num(r[2]) / 100
        if band == "總伏安":
            share = remaining
        elif band.endswith("以下"):
            hi = float(band[:-2])
            share = min(remaining, hi - lo)
            lo = hi
        elif "至" in band:
            a, b = (float(x) for x in band.split("至"))
            share = min(remaining, b - lo)
            lo = b
        else:  # 超過X
            share = remaining
        if share <= 0:
            break
        total += share * pct
        remaining -= share
        parts.append((share, pct))
        base.gold_rows.append(("表五二", f"{place} {r[1]}"))
        if remaining <= 0:
            break
    base.inputs = {f"band{i + 1}": s for i, (s, _) in enumerate(parts)} | {
        f"pct{i + 1}": p for i, (_, p) in enumerate(parts)
    }
    base.value = round(total, 1)
    return base


def receptacle_demand(*, kva: float) -> Answer:
    """第53條 表五三 applied alone: first 10 kVA at 100 %, the rest at 50 %."""
    base = Answer(
        "C4",
        "number",
        None,
        "kVA",
        gold_articles=["53"],
        gold_rows=[("表五三", "10以下"), ("表五三", "超過10")],
    )
    first = min(kva, 10.0)
    rest = max(kva - 10.0, 0.0)
    base.inputs = {"first": first, "rest": rest, "pct_rest": 0.5}
    base.value = round(first + rest * 0.5, 2)
    return base


def dwelling_other_demand(*, kw: float) -> Answer:
    """第60條 表六○ "所有其他負載": first 10 kW at 100 %, the remainder at 40 %."""
    base = Answer(
        "C4",
        "number",
        None,
        "kW",
        gold_articles=["60"],
        gold_rows=[("表六○", "所有其他負載之首10 kW"), ("表六○", "其他負載之剩餘部分")],
    )
    first = min(kw, 10.0)
    rest = max(kw - 10.0, 0.0)
    base.inputs = {"first": first, "rest": rest, "pct_rest": 0.4}
    base.value = round(first + rest * 0.4, 2)
    return base


# -- C5 ---------------------------------------------------------------------------------------


def ground_conductor(*, ocpd_a: float, kind: str = "絞線") -> Answer:
    """第93條四(三) 表九三～二: the row whose range holds the OCPD rating; stranded column mm², solid
    column mm (blank "-" above 30 A -> abstain)."""
    col = 2 if kind == "絞線" else 1
    ladder = sorted(
        {float(r[col]) for r in TABLES["表九三～二"]["rows"] if _num(r[col]) is not None}
    )
    base = Answer(
        "C5", "ladder", None, "mm²" if kind == "絞線" else "mm", gold_articles=["93"], ladder=ladder
    )
    row = next((r for r in TABLES["表九三～二"]["rows"] if _in_range(r[0], ocpd_a)), None)
    if row is None:
        return _abstain(base, f"表九三～二 沒有 {ocpd_a:g} A 的列（最大 6,300 A）")
    base.gold_rows.append(("表九三～二", row[0]))
    cell = _num(row[col])
    if cell is None:
        return _abstain(base, f"表九三～二 {row[0]} 的{kind}欄是「-」")
    base.inputs = {"table_value": cell}
    base.value = cell
    return base


def _abstain(a: Answer, reason: str) -> Answer:
    a.kind, a.value, a.reason = "abstain", None, reason
    return a
