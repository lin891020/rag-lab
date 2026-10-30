"""Checks a question bank must pass before it is run (tests/test_questions_gazette.py uses
them, and the writer agents' drafts are bounced on them). Program logic only.

  number_in_quote   the gold number occurs in the quote in some written form: 5000 / 5,000 /
                    五千 / 五○○○; 三萬 / 三萬零六百 for 萬; ○．五 / 零點五 for decimals;
                    百分之三十 for %  (11's _number_forms, copied)
  starts_with_article_header   the quote opens with 第<numeral>條: article-cn chunking drops
                    that header from the chunk text, so such a question can never hit there
                    (counted, not rejected)
  occurrences       how many times the quote occurs in a text (whitespace removed); repeats are
                    counted, not rejected: is_gold takes any chunk holding the quote
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from chunkers import int_to_cn  # noqa: E402

_DIGIT_CN = str.maketrans("0123456789", "○一二三四五六七八九")
_ARTICLE_HEAD = re.compile(r"^第\s*[零〇一二三四五六七八九十百0-9]+\s*[條點]")


def norm(s: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))


def _cn_large(n: int) -> str:
    out = ""
    if n >= 10_000:
        high = n // 10_000
        out += (_cn_large(high) if high >= 1000 else int_to_cn(high)) + "萬"
        n %= 10_000
    if n >= 1000:
        out += int_to_cn(n // 1000) + "千"
        n %= 1000
    if n:
        out += int_to_cn(n)
    return out


def number_forms(v: float) -> list[str]:
    forms = {f"{v:g}", f"{v:f}".rstrip("0").rstrip(".")}
    if v == int(v):
        n = int(v)
        forms |= {str(n), f"{n:,}", str(n).translate(_DIGIT_CN)}
        if 0 < n < 1000:
            cn = int_to_cn(n)
            forms.add(cn)
            if n >= 100 and (n // 10) % 10 == 1:
                forms.add(cn.replace("一十", "十"))
        if 1000 <= n < 100_000_000:
            cn = _cn_large(n)
            forms.add(cn)
            forms.add(cn.replace("萬", "萬零"))
            if n % 10_000 == 0:
                forms.add(f"{n // 10_000}萬")  # 1萬 (mixed, common in tables)
    else:
        ip, fp = f"{v:f}".rstrip("0").split(".")
        if ip.isdigit() and fp:
            cn_i = "零" if ip == "0" else int_to_cn(int(ip)) if int(ip) < 1000 else ""
            if cn_i:
                forms.add(cn_i + "點" + fp.translate(_DIGIT_CN).replace("○", "零"))
                forms.add(("○" if ip == "0" else cn_i) + "．" + fp.translate(_DIGIT_CN))
                forms.add(("○" if ip == "0" else cn_i) + "." + fp.translate(_DIGIT_CN))
    return [f for f in forms if f]


def number_in_quote(quote: str, value: float, unit: str = "") -> bool:
    q = norm(quote)
    forms = [norm(f) for f in number_forms(float(value))]
    if norm(unit) == "%":
        forms += ["百分之" + f for f in forms]
    return any(f in q for f in forms)


def starts_with_article_header(quote: str) -> bool:
    return _ARTICLE_HEAD.match(norm(quote)) is not None


def occurrences(text: str, quote: str) -> int:
    return norm(text).count(norm(quote))


def route_texts(corpus_text_dirs: dict[str, Path], doc: str) -> dict[str, str]:
    """route -> text for one document, over the routes whose file exists."""
    out = {}
    for route, folder in corpus_text_dirs.items():
        p = folder / f"{doc}.{route}.txt"
        if p.exists():
            out[route] = p.read_text("utf-8")
    return out
