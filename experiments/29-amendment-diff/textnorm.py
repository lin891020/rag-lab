"""Make two spellings of one article read as one, and read numbers as values.

The 110 version writes numerals by position (六○○伏, 一五○, 二‧五 or 二．五); the 114 version
by place words (六百伏特, 一百五十, 二‧五). References move with the renumbering (第十六條 ->
第二十五條, 表一六～三 -> 表二五～二) and the question says they do not count as a change. So:
  cn_value   one numeral token -> float (both styles, decimals, ASCII digits with commas)
  numbers_of the values in a text, references and list markers (一、(一)) masked first
  canon      the text with the 「第N條」 header dropped, references masked, every numeral
             replaced by its value, spacing and punctuation dropped: equal canon => the article
             did not change in substance (T2's 沒改), unequal => changed
"""

from __future__ import annotations

import re
import unicodedata

_DIGITS = {ch: i for i, ch in enumerate("零一二三四五六七八九")} | {"〇": 0, "○": 0}
_PLACE = {"十": 10, "百": 100, "千": 1000}
_CN = "〇○零一二三四五六七八九十百千萬"
_D = "〇○零一二三四五六七八九"
_DEC = "‧．.·"
NUM_RE = re.compile(rf"[0-9][0-9,]*(?:\.[0-9]+)?|[{_CN}]+(?:[{_DEC}][{_D}]+)?")
_REFS = [
    rf"第[{_CN}0-9]+條(?:之[{_CN}0-9]+)?",  # 第十六條, 第二百五十二條之三
    rf"[表圖][{_CN}0-9]+(?:(?:[～~]|之)[{_CN}0-9]+)*",  # 表二五～二, 圖二四～一, 表二九之三一 (old style)
    rf"第[{_CN}0-9]+[章節款項目]",  # 第一章, 第三款, 第二項
    rf"[（(][{_CN}0-9]+[)）]",  # (一), （二）
    rf"[{_CN}]+、",  # 一、 二、 (list markers; a value is followed by its unit, not 、)
    r"[0-9０-９]+[.．]",  # 1. 2. sub-item markers
]
REF_RE = re.compile("|".join(_REFS))
HEAD_RE = re.compile(rf"^第[{_CN}]+條(?:之[{_CN}]+)?\s*")
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[\W_]+")


def _cn_int(s: str) -> int:
    """Chinese numerals, both styles: positional (六○○, 一五○) and place words (一百五十,
    三十五千 = 35,000, 二萬五千 = 25,000, 一千零一十四)."""
    if not any(ch in _PLACE or ch == "萬" for ch in s):
        return int("".join(str(_DIGITS[ch]) for ch in s))  # positional
    total = 0
    for part in s.split("萬"):  # 二萬五千 -> (二) * 10000 + (五千)
        section = num = 0
        max_unit = 0
        for ch in part:
            if ch in _PLACE:
                unit = _PLACE[ch]
                if unit > max_unit:  # 三十五千: the whole 三十五 is multiplied
                    section = (section + num or 1) * unit if section + num else unit
                else:
                    section += (num or 1) * unit
                max_unit = max(max_unit, unit)
                num = 0
            else:
                num = _DIGITS[ch]
        total = total * 10000 + section + num
    return total


def cn_value(tok: str) -> float:
    """「六○○」「六百」 -> 600.0; 「二‧五」 -> 2.5; "1,500" -> 1500.0."""
    tok = unicodedata.normalize("NFKC", tok).replace(",", "")
    if tok[0].isdigit():
        return float(tok)
    m = re.fullmatch(rf"([{_CN}]*)(?:[{_DEC}]([{_D}]+))?", tok)
    if not m:
        raise ValueError(f"not a numeral: {tok!r}")
    whole, frac = m.group(1), m.group(2)
    value = float(_cn_int(whole)) if whole else 0.0
    if frac:
        value += float("0." + "".join(str(_DIGITS[ch]) for ch in frac))
    return value


def _fmt(v: float) -> str:
    return str(int(v)) if v == int(v) else str(v)


def numbers_of(text: str) -> list[float]:
    """Values in a text. A lone Chinese digit (每一, 同一, 之一, 其一) is a word more often than a
    value and is skipped; T3's candidates are found on the rest."""
    s = REF_RE.sub("＃", _WS.sub("", text))
    return [
        cn_value(m.group(0))
        for m in NUM_RE.finditer(s)
        if not (len(m.group(0)) == 1 and m.group(0) in _D)
    ]


def canon(text: str) -> str:
    s = HEAD_RE.sub("", _WS.sub("", unicodedata.normalize("NFKC", text)))
    s = REF_RE.sub("＃", s)
    s = NUM_RE.sub(lambda m: f"#{_fmt(cn_value(m.group(0)))}#", s)
    return _PUNCT.sub("", s)
