"""Two prompts and five kinds of evidence.

P0   09's prompt word for word (no way to say "not here"); the schema forces a number
P1   the same prompt with a `found` flag first in the schema and one rule: when the evidence
     does not hold the answer, found=false and the other fields empty / 0; do not fill in from
     memory

gold       09's oracle window (1,000 chars around the quote)         answerable
same-doc   1,000 chars of the same document that do not hold the quote  unanswerable (hardest)
other-doc  the best-ranked chunk of another document                   unanswerable
none       no evidence                                                  unanswerable
real       10's `full` retrieval cell, whole chunks under 1,000 chars   mixed (has_gold splits)
"""

from __future__ import annotations

import json

SYSTEM = "你是台灣的法制作業助理，熟悉政府公報裡的法規修正公告。只回一個 JSON 物件，不要其他文字。"

PROMPTS = ["P0", "P1"]
EVIDENCE = ["gold", "same-doc", "other-doc", "none", "real"]
REAL_CELL = ("ocrrows", "article-cn+pre", True)  # 10's `full`

_P0_SCHEMAS: dict[str, dict] = {
    "value": {
        "type": "object",
        "properties": {
            "quote": {"type": "string"},
            "value": {"type": "number"},
            "unit": {"type": "string"},
        },
        "required": ["quote", "value", "unit"],
    },
    "articles": {
        "type": "object",
        "properties": {
            "quote": {"type": "string"},
            "articles": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["quote", "articles"],
    },
    "which": {
        "type": "object",
        "properties": {"quote": {"type": "string"}, "law": {"type": "string"}},
        "required": ["quote", "law"],
    },
}


def _with_found(schema: dict) -> dict:
    return {
        "type": "object",
        "properties": {"found": {"type": "boolean"}, **schema["properties"]},
        "required": ["found", *schema["required"]],
    }


SCHEMAS: dict[str, dict[str, dict]] = {
    "P0": _P0_SCHEMAS,
    "P1": {t: _with_found(s) for t, s in _P0_SCHEMAS.items()},
}

_RULES = {
    "value": """{{"quote": "依據的原文那一句（照抄）", "value": 數字, "unit": "單位字串"}}

規則：
- value 必須是 JSON 數字（不是字串），照原文的數值填；原文寫中文數字（如「三萬元」「○．四」）要換成阿拉伯數字（30000、0.4）。
- unit 填題目問的單位，例如 "元"、"日"、"人"、"%"；沒有單位就填空字串 ""。
- 問題問「修正後」時，以修正條文為準，不要抄現行條文的數字。""",
    "articles": """{{"quote": "依據的原文那一句（照抄）", "articles": ["條號", "條號"]}}

規則：
- articles 是字串陣列，每個元素一個條（點）號，用阿拉伯數字，例如 "5"；「第十二條之一」寫成 "12-1"；要點的「第四點」寫成 "4"。
- 只列這次公告修正、增訂或刪除的條（點），不要列引用到的其他條。""",
    "which": """{{"quote": "依據的原文那一句（照抄）", "law": "法規全名"}}

規則：
- law 填法規的完整名稱，不加「」，不加「修正草案」等字。
- 若依據裡有好幾個法規，挑規定了問題那件事的那一個。""",
}

_P1_EMPTY = {
    "value": "quote 填空字串、value 填 0、unit 填空字串",
    "articles": "quote 填空字串、articles 填空陣列 []",
    "which": "quote 填空字串、law 填空字串",
}

_P1_RULE = """
- 先判斷依據裡有沒有這題的答案：有就 found 填 true；**依據裡找不到答案就 found 填 false，{empty}，不要用你自己知道的補。**"""

_HEAD = """請回答下面的問題，只回一個 JSON 物件，鍵名與順序固定：
{rules}

JSON schema：
{schema}

{evidence}

問題：{question}"""


def _rules(qtype: str, prompt: str) -> str:
    rules = _RULES[qtype]
    if prompt == "P1":
        first, rest = rules.split("\n", 1)
        rules = '{{"found": true或false, ' + first[2:] + "\n" + rest
        rules += _P1_RULE.format(empty=_P1_EMPTY[qtype])
    return rules


def user_prompt(
    question: str, *, qtype: str, evidence: str | None, label: str, prompt: str = "P0"
) -> str:
    if evidence is None:
        ev = "依據：本題不提供原文，請依你對台灣法規的知識回答；quote 填你記得的那句。"
    else:
        ev = f"依據（{label}）：\n<<<\n{evidence}\n>>>"
    return _HEAD.format(
        rules=_rules(qtype, prompt),
        schema=json.dumps(SCHEMAS[prompt][qtype], ensure_ascii=False),
        evidence=ev,
        question=question,
    )
