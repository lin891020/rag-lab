"""Same prompt and schema as 02/03/04/05/06. Only the evidence block changes: the table as text
(the PDF text layer, or glm-ocr's reply rendered as 06 did) or as page images attached before the
prompt (06's clean page, or 22's worst scan of it).
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

# condition -> corpus route (both copied from 24: clean = 04's markdown of the text layer,
# glm2 = glm-ocr's reply for 06's page rendered as 06 did, byte for byte 23's glm2-s100)
TEXT: dict[str, str] = {"clean": "clean", "glm2": "glm2"}
# condition -> where its pages come from (run.pages_for)
IMAGES: dict[str, str] = {
    "image": "06's clean 300 dpi page, PNG",
    "image-L2": "22's worst scan: 150 dpi, 1.5°, blur, noise, JPEG q30",
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


def image_prompt(question: str, *, label: str, pages: int) -> str:
    """06's prompt with the evidence block naming the page images sent before it."""
    ev = f"依據（{label}）：附在這段文字前面的頁圖，共 {pages} 張。"
    return _HEAD.format(
        schema=json.dumps(SCHEMA, ensure_ascii=False), evidence=ev, question=question
    )


def user_prompt(question: str, *, evidence: str | None, label: str) -> str:
    if evidence is None:
        ev = "依據：本題不提供條文或表格，請依你對這部規則的知識回答；quote 填你記得的那句。"
    else:
        ev = f"依據（{label}）：\n<<<\n{evidence}\n>>>"
    return _HEAD.format(
        schema=json.dumps(SCHEMA, ensure_ascii=False), evidence=ev, question=question
    )
