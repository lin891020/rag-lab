"""29's prompts: a new system line (two versions, the tool reads the old one), one JSON schema per
question kind, the evidence line for each of the three conditions, and the search tool.

27's rules are not reused: those are for the numeric lookup questions. Here the reply is a
version-comparison judgement, so each kind has its own small schema; `found` first (11's P1).
T1  found false means 新增 (no old article); old_article is the old number as written
T2  changed is the judgement (any wording change counts; renumbering, references, numeral style
    and punctuation do not); both quotes show what was compared
T3  the two values as numbers, in the unit the question names
The closed condition's line keeps a single instruction ("依你的記憶回答，不確定也要猜") so 26's two
competing instructions (do not fill in from memory / answer from memory) never meet.
"""

from __future__ import annotations

import json

NEW_DATE, OLD_DATE = "114 年 5 月 9 日", "110 年 3 月 17 日"

SYSTEM = (
    f"你是台灣的電機技師助理，熟悉《用戶用電設備裝置規則》。這部規則在民國 {NEW_DATE}全文修正，"
    f"條號大多重排；修正前的版本是 {OLD_DATE}修正的，條號不同。題目會給你新版某一條的全文，"
    "問它跟舊版的對應關係。只回一個 JSON 物件，不要其他文字。"
)

SCHEMAS: dict[str, dict] = {
    "T1": {
        "type": "object",
        "properties": {
            "found": {"type": "boolean"},
            "old_article": {"type": "string"},
            "quote": {"type": "string"},
        },
        "required": ["found", "old_article", "quote"],
    },
    "T2": {
        "type": "object",
        "properties": {
            "found": {"type": "boolean"},
            "old_article": {"type": "string"},
            "changed": {"type": "boolean"},
            "quote_new": {"type": "string"},
            "quote_old": {"type": "string"},
        },
        "required": ["found", "old_article", "changed", "quote_new", "quote_old"],
    },
    "T3": {
        "type": "object",
        "properties": {
            "found": {"type": "boolean"},
            "old_article": {"type": "string"},
            "old_value": {"type": "number"},
            "new_value": {"type": "number"},
            "unit": {"type": "string"},
            "quote_old": {"type": "string"},
            "quote_new": {"type": "string"},
        },
        "required": [
            "found",
            "old_article",
            "old_value",
            "new_value",
            "unit",
            "quote_old",
            "quote_new",
        ],
    },
}

RULES: dict[str, str] = {
    "T1": """{"found": true或false, "old_article": "舊版條號", "quote": "舊版那一條開頭的一句（照抄）"}

規則：
- 在舊版裡找到跟新版這一條對應（內容相同或由它修正而來）的條文，found 填 true，old_article 填它的條號，寫成「16」或「13-1」這種形式（第十三條之一寫成 13-1），quote 照抄那一條開頭的一句。
- 舊版沒有對應的條文（新版這一條是新增的），found 填 false、old_article 填空字串、quote 填空字串。
- 條號不同不代表沒有對應：這次修正把條號大多重排了，要靠內容比對。""",
    "T2": """{"found": true或false, "old_article": "舊版條號", "changed": true或false, "quote_new": "新版有差異或對應的一句（照抄）", "quote_old": "舊版有差異或對應的一句（照抄）"}

規則：
- 先在舊版裡找到跟新版這一條對應的條文，found 填 true，old_article 填它的條號（「16」或「13-1」）。找不到就 found 填 false，其餘填空字串、changed 填 false。
- 比較兩條的文字：用詞、句子、規定的事項、數值、條件、適用範圍有任何改動，changed 填 true（例如「選定」改成「有」、「損害」改成「損壞」、但書改寫成另一項，都算）；只有條號不同、因條次變更而改的引據條號與表號、數字寫法不同（六○○伏與六百伏特）、標點或分項分款的編號不同，changed 填 false。
- quote_new、quote_old 各照抄一句：有修改就抄差異處，沒修改就抄兩邊對應的同一句。""",
    "T3": """{"found": true或false, "old_article": "舊版條號", "old_value": 數字, "new_value": 數字, "unit": "單位", "quote_old": "舊版寫這個數值的那一句（照抄）", "quote_new": "新版寫這個數值的那一句（照抄）"}

規則：
- 先在舊版裡找到跟新版這一條對應的條文，found 填 true，old_article 填它的條號（「16」或「13-1」）。找不到就 found 填 false，數值填 0、其餘填空字串。
- old_value、new_value 必須是 JSON 數字，依題目指定的單位換算（二‧五公尺以毫米計是 2500；六○○伏是 600）。
- quote_old、quote_new 照抄兩版寫這個數值的那一句。""",
}

LINES: dict[str, str] = {
    "agent": (
        "依據：本題只附新版條文，不附舊版。請用 search_old 工具查舊版（110 年 3 月 17 日修正）的條文，"
        "每次回傳約 3,000 字，最多查 5 次；找到對應的舊條文再回答。"
    ),
    "oracle": "依據：下面附上舊版（110 年 3 月 17 日修正）對應的那一條全文，請直接比對後回答。",
    "closed": "依據：本題不附舊版條文，也沒有工具。請依你的記憶回答，不確定也要猜，不要棄答。",
}

TOOL: dict = {
    "name": "search_old",
    "description": (
        "在《用戶用電設備裝置規則》舊版（民國 110 年 3 月 17 日修正）的條文裡做語意搜尋，"
        "回傳最相關的條文段落，合計約 3,000 字。用條文的內容或關鍵詞去查，不要用新版的條號查。"
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "要查的內容：條文主題、關鍵詞或一句原文"}
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

_ORACLE_BLOCK = "\n\n舊版對應條文（第 {old_num} 條）：\n{old_text}"


def prompt_for(q: dict, cond: str, old_text: str | None = None) -> str:
    """The user turn for one question in one condition. oracle appends the old article."""
    kind = q["kind"]
    evidence = LINES[cond]
    if cond == "oracle":
        if old_text is None:
            raise ValueError("oracle needs the old article's text")
        evidence += _ORACLE_BLOCK.format(old_num=q["old_num"], old_text=old_text)
    return _HEAD.format(
        rules=RULES[kind],
        schema=json.dumps(SCHEMAS[kind], ensure_ascii=False),
        evidence=evidence,
        question=q["question"],
    )
