"""Shrinking keeps everything but the size: same mode, same aspect, bytes reproducible."""

import io

from PIL import Image, ImageDraw
from shrink import SCALES, shrink


def _page():
    im = Image.new("RGB", (2481, 3508), "white")
    ImageDraw.Draw(im).rectangle((200, 300, 2200, 900), outline="black", width=4)
    return im


def test_scales_are_the_planned_ones():
    assert SCALES == {"s83": 0.83, "s67": 0.67, "s50": 0.5, "s33": 0.33}


def test_shrink_scales_both_sides_and_keeps_the_mode():
    out = Image.open(io.BytesIO(shrink(_page(), 0.5)))
    assert out.size == (1240, 1754) and out.mode == "RGB" and out.format == "PNG"


def test_shrink_is_reproducible_byte_for_byte():
    assert shrink(_page(), 0.33) == shrink(_page(), 0.33)
