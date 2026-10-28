"""Same prompt and schema as 02 (02 showed the forcing method makes no difference, so every
condition here uses schema-constrained decoding). Only the evidence block changes: the same
table rendered as flat text (01's text layer), markdown, html or kv; closed-book gets none.
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

# condition -> which rendering of the table goes into the prompt (None = closed-book)
CONDITIONS: dict[str, str | None] = {
    "closed-book": None,  # noise floor
    "flat": "flat",  # blind baseline: 01's corpus/tables/*.txt
    "markdown": "markdown",
    "html": "html",
    "kv": "kv",
}

_HEAD = """請回答下面的問題，只回一個 JSON 物件，鍵名與順序固定：
{{"quote": "依據的原文那一句（照抄）", "value": 數字, "unit": "單位字串"}}

規則：
- value 必須是 JSON 數字（不是字串），照原文的數值填，不要換算、不要四捨五入。
- unit 填原文的單位，例如 "A"、"mm²"、"V"、"%"、"倍"；沒有單位就填空字串 ""。
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
