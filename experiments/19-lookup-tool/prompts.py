"""19 adds the lookup tool's first step (LOOKUP_SCHEMA, lookup_prompt): the model reads the table
catalog and names the cells it needs; the second step is user_prompt below, unchanged, with the
looked-up cells as evidence. The rest is 16/17's text:

One prompt family for the six conditions. The schema always has `found` first (11's P1) and
an `expression` field: in L1 the model fills `value` itself and the expression is only read for
attribution; in L2 the program evaluates the expression and overwrites `value` for number
questions (the "calculation tool"). Sizing (ladder) questions keep the model's own `value` in
every condition: the tool does arithmetic, not table lookups.
"""

from __future__ import annotations

import json

SYSTEM = "你是台灣的電機技師助理，熟悉屋內線路裝置規則與其附表。只回一個 JSON 物件，不要其他文字。"

CONDITIONS: dict[str, dict] = {  # name -> evidence source, tool
    "L0": {"evidence": None, "tool": False},
    "L1-fixed": {"evidence": "fixed", "tool": False},
    "L1-struct": {"evidence": "struct", "tool": False},
    "L1-oracle": {"evidence": "oracle", "tool": False},
    "L2-struct": {"evidence": "struct", "tool": True},
    "L2-oracle": {"evidence": "oracle", "tool": True},
}

SCHEMA: dict = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "quote": {"type": "string"},
        "expression": {"type": "string"},
        "value": {"type": "number"},
        "unit": {"type": "string"},
    },
    "required": ["found", "quote", "expression", "value", "unit"],
}

_RULES = """{"found": true或false, "quote": "依據的那一句或那一列（照抄）", "expression": "算式", "value": 數字, "unit": "單位"}

規則：
- 先判斷依據裡有沒有這題需要的規定與表格列：有就 found 填 true；**依據裡找不到、表格沒有這一格、或規定不適用於題目的情況，found 填 false、quote 填空字串、expression 填空字串、value 填 0、unit 填空字串，不要用你自己知道的數值補。**
- quote 照抄依據裡你用到的那一句規定或那一列表格（含表格的欄名與數值）。
- expression 寫出你怎麼算的，只能用數字和 + − × ÷ ( )，例如 "23.2 × 1.25" 或 "60 × 0.77"；直接查表不用算的題，expression 填那個查到的數字，例如 "60"。
- value 必須是 JSON 數字：要換算就換算（倍數、百分比、分級），單位照題目問的填（A、mm²、mm、W、kVA、kW）。
- 選線徑的題，value 填選出來的線徑（表裡有的規格），不是算出來的電流。"""

_TOOL_RULE = """
- 本題的 value 會由程式照你的 expression 算，所以 expression 要完整（含所有倍數與係數）；value 仍要填。"""

_HEAD = """請回答下面的問題，只回一個 JSON 物件，鍵名與順序固定：
{rules}

JSON schema：
{schema}

{evidence}

問題：{question}"""


def user_prompt(question: str, *, evidence: str | None, label: str, tool: bool) -> str:
    if evidence is None:
        ev = "依據：本題不提供原文，請依你對屋內線路裝置規則與附表的知識回答；quote 填你記得的那句或那一列。"
    else:
        ev = f"依據（{label}）：\n<<<\n{evidence}\n>>>"
    rules = _RULES + (_TOOL_RULE if tool else "")
    return _HEAD.format(
        rules=rules, schema=json.dumps(SCHEMA, ensure_ascii=False), evidence=ev, question=question
    )


MAX_LOOKUPS = 6
LOOKUP_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "lookups": {
            "type": "array",
            "maxItems": MAX_LOOKUPS,
            "items": {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "row": {"type": "string"},
                    "column": {"type": "string"},
                },
                "required": ["table", "row", "column"],
            },
        }
    },
    "required": ["lookups"],
}

_LOOKUP_V1 = """下面是《用戶用電設備裝置規則》附表的目錄：每張表列出表號與表名、欄名、列名（不含表內數值）。
要回答問題，你需要查哪幾格？每一格寫：
- table：表號，例如「表九三～二」；
- row：列名，照目錄抄；範圍列（例如「800以下」）可以直接寫題目裡的數字，程式會找它落在哪一列；
- column：欄名，照目錄抄，或只寫能分辨的那一段；範圍欄（例如導線數「7-9」）也可以直接寫數字。
需要幾格就列幾格，最多 {n} 格。程式會把查到的格子交給你作答。只回一個 JSON 物件：
{{"lookups": [{{"table": "...", "row": "...", "column": "..."}}]}}

目錄：
<<<
{catalog}
>>>

問題：{question}"""

_LOOKUP = """下面是《用戶用電設備裝置規則》附表的目錄：每張表列出表號與表名、欄名、列名（不含表內數值）。
要回答問題，你需要查哪幾格？每一格寫：
- table：表號，照目錄抄；
- row：列名，照目錄抄；範圍列可以直接寫題目裡的數字，程式會找它落在哪一列；
- column：欄名，照目錄抄，或只寫能分辨的那一段；範圍欄也可以直接寫數字。
需要幾格就列幾格，最多 {n} 格。程式會把查到的格子交給你作答。只回一個 JSON 物件：
{{"lookups": [{{"table": "...", "row": "...", "column": "..."}}]}}

目錄：
<<<
{catalog}
>>>

問題：{question}"""
# The fix cells' text: _LOOKUP_V1 with its three examples taken out (qwen3 copied 「800以下」 into
# 10 C5 lookups in the first run); the table line says 「照目錄抄」 instead. Nothing else changed.

LOOKUP_LABEL = "查表結果：程式依你指定的表、列、欄查到的格子"


def lookup_prompt(question: str, *, catalog: str, fix: bool) -> str:
    text = _LOOKUP if fix else _LOOKUP_V1
    return text.format(n=MAX_LOOKUPS, catalog=catalog, question=question)
