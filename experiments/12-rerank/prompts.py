"""Reading prompt (09's P0, word for word) and the listwise reranking prompt.

Prompt and schema per question type. The value prompt is 02–06's; the two new types keep the
same shape (quote first, then the answer) so the models are asked the same way.

  value     {"quote", "value", "unit"}      a number read from the amended text
  articles  {"quote", "articles"}            which 條 (or 點) this notice amends, as strings
  which     {"quote", "law"}                 the name of the regulation that says so
"""

from __future__ import annotations

import json

SYSTEM = "你是台灣的法制作業助理，熟悉政府公報裡的法規修正公告。只回一個 JSON 物件，不要其他文字。"

SCHEMAS: dict[str, dict] = {
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

# condition -> (route, chunking, prefix) of the retrieval cell whose dense ranking feeds the
# reader; None = closed-book; "oracle" = the passage around the quote itself.
CONDITIONS: dict[str, tuple[str, str, bool] | str | None] = {
    "closed-book": None,
    "oracle": "oracle",
    "base": ("base", "article", False),
    "fixed": ("base", "fixed-1000", False),
    "all": ("rows", "article-cn+pre", True),
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

_HEAD = """請回答下面的問題，只回一個 JSON 物件，鍵名與順序固定：
{rules}

JSON schema：
{schema}

{evidence}

問題：{question}"""


def user_prompt(question: str, *, qtype: str, evidence: str | None, label: str) -> str:
    if evidence is None:
        ev = "依據：本題不提供原文，請依你對台灣法規的知識回答；quote 填你記得的那句。"
    else:
        ev = f"依據（{label}）：\n<<<\n{evidence}\n>>>"
    return _HEAD.format(
        rules=_RULES[qtype],
        schema=json.dumps(SCHEMAS[qtype], ensure_ascii=False),
        evidence=ev,
        question=question,
    )


# -- listwise reranking (RankGPT style) --------------------------------------------------------------

LISTWISE_SYSTEM = "你是檢索結果的重排器。只回一個 JSON 物件，不要其他文字。"

LISTWISE_SCHEMA: dict = {
    "type": "object",
    "properties": {"order": {"type": "array", "items": {"type": "integer"}}},
    "required": ["order"],
}

_LISTWISE_HEAD = """下面有 {n} 段文字，編號 [1] 到 [{n}]。請依「哪一段最可能含有這個問題的答案」由高到低排序，回 JSON：{{"order": [編號, 編號, …]}}，每個編號出現一次、全部都要列。

JSON schema：
{schema}

問題：{question}

{passages}"""


def listwise_prompt(question: str, passages: list[str]) -> str:
    body = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return _LISTWISE_HEAD.format(
        n=len(passages),
        schema=json.dumps(LISTWISE_SCHEMA, ensure_ascii=False),
        question=question,
        passages=body,
    )
