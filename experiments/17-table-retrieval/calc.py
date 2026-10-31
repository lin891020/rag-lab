"""The "calculation tool": evaluate the model's arithmetic expression safely. Numbers, + - * /,
parentheses, unary minus; nothing else (no names, calls, powers). The text is NFKC-folded and the
fullwidth / typographic operators the prompt allows (× ÷ （ ） －) are mapped first."""

from __future__ import annotations

import ast
import re
import unicodedata

MAX_LEN = 200
_MAP = str.maketrans({"×": "*", "÷": "/", "−": "-", "－": "-", "（": "(", "）": ")", "，": ""})
_BIN = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}


class BadExpression(ValueError):
    pass


_SUPERSCRIPT = set("⁰¹²³⁴⁵⁶⁷⁸⁹")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)%")


def normalize(expr: str) -> str:
    """Cut at the first = / ≈ (models like to append the result), 35% -> (35/100), fold widths,
    map × ÷ − （ ）. Superscript digits (2×10³) would fold to plain digits and change the value,
    so they are rejected by evaluate()."""
    if any(ch in _SUPERSCRIPT for ch in expr):
        raise BadExpression("superscript")
    head = re.split(r"[=≈＝]", expr, maxsplit=1)[0]
    s = unicodedata.normalize("NFKC", head).translate(_MAP).replace(",", "").replace(" ", "")
    return _PERCENT.sub(r"(\1/100)", s)


def evaluate(expr: str) -> float:
    """Value of the expression, or BadExpression (empty, too long, forbidden node, division by zero)."""
    s = normalize(expr or "")
    if not s:
        raise BadExpression("empty")
    if len(s) > MAX_LEN:
        raise BadExpression("too long")
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError as exc:
        raise BadExpression(f"syntax: {exc.msg}") from exc
    try:
        return float(_eval(tree.body))
    except ZeroDivisionError as exc:
        raise BadExpression("division by zero") from exc


def _eval(node: ast.AST) -> float:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](_eval(node.left), _eval(node.right))
    raise BadExpression(f"forbidden: {type(node).__name__}")
