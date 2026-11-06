"""28's prompt: 27's, with the search pointed at a library of six laws instead of 用戶用電設備裝置
規則 alone. The library is listed in pcode order (D…, J…, N…), so the law that applies is neither
first nor singled out. 17's rules word for word; the schema has `found` first (11's P1); the model
fills `value` itself (no calculation tool, as 26 and 27).
"""

from __future__ import annotations

import json

LIBRARY = (  # pcode order: D0070117 D0120029 J0030011 J0030018 J0030019 N0060009
    "建築技術規則建築設備編",
    "各類場所消防安全設備設置標準",
    "電業法",
    "用戶用電設備裝置規則",
    "輸配電設備裝置規則",
    "職業安全衛生設施規則",
)
_NAMES = "".join(f"《{n}》" for n in LIBRARY)

SYSTEM = "你是台灣的電機技師助理，熟悉屋內線路裝置規則與其附表。只回一個 JSON 物件，不要其他文字。"

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

AGENT_LINE = (
    f"依據：本題不附原文。請用 search 工具查法規庫（{_NAMES}六部）的條文與附表，"
    "每次回傳約 3,000 字，最多查 5 次；查到足夠的依據再回答。"
    "上面規則裡的「依據」，指 search 查回來的文字。"
)

TOOL: dict = {
    "name": "search",
    "description": (
        f"在法規庫（{_NAMES}六部）的條文與附表裡做語意搜尋，回傳最相關的條文段落與整張附表，"
        "合計約 3,000 字。換不同的說法可以查到不同的條文或表。"
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "要查的內容：條文主題、表名或關鍵詞"}
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

_HEAD = """請回答下面的問題，只回一個 JSON 物件，鍵名與順序固定：
{rules}

JSON schema：
{schema}

{evidence}

問題：{question}"""


def agent_prompt(question: str) -> str:
    return _HEAD.format(
        rules=_RULES,
        schema=json.dumps(SCHEMA, ensure_ascii=False),
        evidence=AGENT_LINE,
        question=question,
    )
