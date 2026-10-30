"""22 clean tables -> corpus/tables_clean.json. 17 come straight from 04's tables.json (cell
geometry from the attachment PDFs, checked against the page images in 03/04); 5 more are
extracted here with the same loader (tables.py, 04's copy) and patched where the text layer
is wrong: 表六○ prints "l00" / "l0" with a lowercase L, and 表二五八～三 loses the fraction
bars of 1/2, 3/4, 1-1/2, 7-1/2 hp (those rows are dropped: the chains use integer hp only).

    uv run python experiments/16-rule-lookup-compute/build_tables.py

Every table carries `checked`: who verified it against the PDF page and when. The 5 new ones
were checked cell by cell by an independent agent on 2026-10-30 (5/5 all cells right); if a
row changes, re-run that check before trusting the stamp.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
ATTACH = ROOT / "data/01-chunking/attachments"
FOUR = HERE.parent / "04-table-lookup" / "corpus" / "tables.json"
OUT = HERE / "corpus" / "tables_clean.json"
sys.path.insert(0, str(HERE))
from tables import load_table, table_to_json  # noqa: E402

logger = logging.getLogger(__name__)

NEW_TABLES: dict[str, str] = {  # 表號 -> attachment id (first line of the PDF names the table)
    "表二五～三": "0000395919",
    "表五三": "0000395927",
    "表五八": "0000395930",
    "表六○": "0000395931",
    "表二五八～三": "0000395949",
}
INTEGER_HP = {
    "1",
    "2",
    "3",
    "5",
    "10",
    "15",
    "20",
    "25",
    "30",
    "40",
    "50",
    "60",
    "75",
    "100",
    "125",
    "150",
    "200",
}
CHECKED_17 = "04 build_corpus + 03/04 page-image check (2026-10-29)"
CHECKED_5 = "independent agent, cell by cell against the attachment PDF pages, 5/5 tables all cells right (2026-10-30)"


def _fix_lowercase_l(rows: list[list[str]]) -> list[list[str]]:
    """'l00' -> '100', '首l0 kW' -> '首10 kW' (text-layer OCR-like slips in 表六○)."""
    out = []
    for r in rows:
        out.append([c.replace("l00", "100").replace("l0 kW", "10 kW") for c in r])
    return out


def _integer_hp_rows(rows: list[list[str]]) -> list[list[str]]:
    return [r for r in rows if r[0] in INTEGER_HP]


def build() -> dict[str, dict]:
    blob = json.loads(FOUR.read_text("utf-8"))
    for tid in blob:
        blob[tid]["checked"] = CHECKED_17
    for tid, fid in NEW_TABLES.items():
        j = table_to_json(load_table(ATTACH / f"{fid}.pdf"))
        if tid == "表六○":
            j["rows"] = _fix_lowercase_l(j["rows"])
        if tid == "表二五八～三":
            j["rows"] = _integer_hp_rows(j["rows"])
            j["names"][0] = "馬力(hp)"
        if tid == "表二五～三":  # the subtitle carries the conditions the chains key on
            j["notes"] = "（導線絕緣物最高容許溫度75℃，周圍溫度35℃）" + (j.get("notes") or "")
        if tid == "表二五八～三":
            j["notes"] = (
                "(本表所列數值為附有皮帶電動機及正常轉矩特性之電動機，於通常速率運轉時之典型滿載電流值。表列電壓為電動機額定電壓。) "
                + (j.get("notes") or "")
            )
        j["checked"] = CHECKED_5
        blob[tid] = j
    return blob


def main() -> int:
    setup_logging("rag_lab")
    blob = build()
    OUT.write_text(json.dumps(blob, ensure_ascii=False, indent=1), "utf-8")
    logger.info(
        "tables built",
        extra={
            "ctx": {
                "tables": len(blob),
                "unchecked": sum(1 for t in blob.values() if not t["checked"]),
            }
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
