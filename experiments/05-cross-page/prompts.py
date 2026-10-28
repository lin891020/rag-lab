"""Same prompt and schema as 02/03/04. Only the evidence block changes: the table merged across
its pages (merged), every page as its own block in order (pages), or only the page that holds
the answer row (page); closed-book gets none.
"""

from __future__ import annotations

import json

SYSTEM = "你是台灣的電機技師，熟悉《用戶用電設備裝置規則》。只回一個 JSON 物件，不要其他文字。"

SCHEMA: dict = {
    "type": "object",
    "properties": {
        "quote": {"type": "string"},
        "value": {"type": "number"},
        "unit": {"type": "string"},
    },
    "required": ["quote", "value", "unit"],
}

# condition -> how the evidence is built; None = closed-book
CONDITIONS: dict[str, str | None] = {
    "closed-book": None,  # noise floor
    "page": "page",  # the page with the answer row, read on its own (page-chunked pipeline)
    "pages": "pages",  # every page of the table as its own block, in order (all chunks retrieved)
    "merged": "merged",  # continuation detected and merged: one table with its header
}

_HEAD = """請回答下面的問題，只回一個 JSON 物件，鍵名與順序固定：
{{"quote": "依據的原文那一句（照抄）", "value": 數字, "unit": "單位字串"}}

規則：
- value 必須是 JSON 數字（不是字串），照原文的數值填，不要換算、不要四捨五入。
- unit 填原文的單位，例如 "A"、"mm²"、"V"、"%"、"倍"、"區"；沒有單位就填空字串 ""。
- 問題只問一個數字；原文有多個數字時，挑問題問的那一個。

JSON schema：
{schema}

{evidence}

問題：{question}"""


def user_prompt(question: str, *, evidence: str | None, label: str) -> str:
    if evidence is None:
        ev = "依據：本題不提供條文或表格，請依你對這部規則的知識回答；quote 填你記得的那句。"
    else:
        ev = f"依據（{label}）：\n<<<\n{evidence}\n>>>"
    return _HEAD.format(
        schema=json.dumps(SCHEMA, ensure_ascii=False), evidence=ev, question=question
    )
