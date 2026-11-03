"""The synthetic scans: fixed parameters, byte-identical on every run and machine, white fill
where the rotation exposes the background, and the noise seeded per page (not per process)."""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from degrade import LEVELS, render, scan, seed_for

ATTACH = Path(__file__).parents[3] / "data/01-chunking/attachments"


def _page() -> Image.Image:
    img = Image.new("RGB", (800, 1100), "white")
    d = ImageDraw.Draw(img)
    for y in range(100, 1000, 60):
        d.line([(80, y), (720, y)], fill="black", width=2)
        d.text((100, y + 15), "表五二 100 80 70", fill="black")
    return img


def _args(level: str) -> dict:
    return {k: v for k, v in LEVELS[level].items() if k != "dpi"}


def test_levels_are_the_planned_ones():
    assert LEVELS == {
        "L1": {"dpi": 200, "angle": 0.5, "blur": 0.0, "noise": 0.0, "quality": 50},
        "L2": {"dpi": 150, "angle": 1.5, "blur": 0.8, "noise": 8.0, "quality": 30},
    }


def test_scan_is_byte_identical_for_the_same_page():
    for level in LEVELS:
        s = seed_for(level, "表五二-p0")
        assert scan(_page(), **_args(level), seed=s) == scan(_page(), **_args(level), seed=s)


def test_seed_is_stable_and_differs_by_page():
    assert seed_for("L2", "表五二-p0") == seed_for("L2", "表五二-p0")  # not hash(): no salt
    assert seed_for("L2", "表五二-p0") != seed_for("L2", "表五三-p0")
    a = scan(_page(), **_args("L2"), seed=seed_for("L2", "a"))
    b = scan(_page(), **_args("L2"), seed=seed_for("L2", "b"))
    assert a != b


def test_rotation_fills_the_corners_white():
    for level in LEVELS:
        img = Image.open(io.BytesIO(scan(_page(), **_args(level), seed=1)))
        x = np.asarray(img, dtype=np.float64)
        for patch in (x[:8, :8], x[:8, -8:], x[-8:, :8], x[-8:, -8:]):
            assert patch.mean() > 230, level


def test_scan_is_a_grayscale_jpeg_rotated_with_expand():
    raw = scan(_page(), **_args("L2"), seed=1)
    img = Image.open(io.BytesIO(raw))
    assert img.format == "JPEG" and img.mode == "L"
    assert img.size[0] > 800 and img.size[1] > 1100  # expand=True keeps the corners of the page


def test_l2_is_worse_than_l1():
    """Smaller file at the same size means less detail survived (blur, noise, q30)."""
    page = _page()
    l1 = Image.open(io.BytesIO(scan(page, **_args("L1"), seed=1)))
    l2 = Image.open(io.BytesIO(scan(page, **_args("L2"), seed=1)))
    sharp = np.abs(np.diff(np.asarray(l1, dtype=np.float64), axis=1)).max()
    soft = np.abs(np.diff(np.asarray(l2.resize(l1.size), dtype=np.float64), axis=1)).max()
    assert soft < sharp


@pytest.mark.skipif(not (ATTACH / "0000395915.pdf").exists(), reason="data/ is not in git")
def test_render_uses_the_level_dpi():
    import pdfplumber

    with pdfplumber.open(ATTACH / "0000395915.pdf") as pdf:
        page = pdf.pages[0]
        img = render(page, 150)
        assert abs(img.size[0] - page.width * 150 / 72) <= 1
