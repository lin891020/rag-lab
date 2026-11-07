"""Article numbers in any spelling -> one key: 「第十六條」「第 16 條」「16」 -> "16";
「第十三條之一」「第 13-1 條」「13-1」 -> "13-1". 15's `norm_article` + `cn_to_int`, copied, with
two patches: 千 (this law has 1,014 articles; 15's table stopped at 百) and the database page's
「第 13-1 條」 spelling of a branch."""

from __future__ import annotations

import re
import unicodedata

_CN_VALUE = {ch: i for i, ch in enumerate("零一二三四五六七八九")} | {"〇": 0}
_CN = "零〇一二三四五六七八九十百千"
_ART = re.compile(rf"^第?\s*([0-9]+|[{_CN}]+)\s*條?(?:之\s*([0-9]+|[{_CN}]+))?\s*(?:條|點)?$")
_DASH = re.compile(r"^第?\s*([0-9]+)\s*-\s*([0-9]+)\s*條?$")


def cn_to_int(s: str) -> int:
    """Chinese numeral (零 to 九千九百九十九; 十, 百, 千 as place words) to int."""
    s = s.replace(" ", "")
    total = 0
    num = 0
    for ch in s:
        if ch == "千":
            total += (num or 1) * 1000
            num = 0
        elif ch == "百":
            total += (num or 1) * 100
            num = 0
        elif ch == "十":
            total += (num or 1) * 10
            num = 0
        else:
            num = _CN_VALUE[ch]
    return total + num


def norm_article(s: str) -> str:
    """'第十二條之一' / '12之1' / '12-1' / '第 12-1 條' / '第4點' -> '12-1' / '4'; unknown strings
    fold to themselves with spaces removed."""
    s = unicodedata.normalize("NFKC", str(s)).strip().replace(" ", "")
    m = _DASH.match(s)
    if m:
        return f"{int(m.group(1))}-{int(m.group(2))}"
    m = _ART.match(s)
    if not m:
        return s

    def num(x: str) -> int:
        return int(x) if x.isdigit() else cn_to_int(x)

    out = str(num(m.group(1)))
    return f"{out}-{num(m.group(2))}" if m.group(2) else out
