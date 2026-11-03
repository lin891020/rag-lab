"""06's clean 300 dpi page PNGs, shrunk before OCR -> data/23-size/<scale>/<頁名>.png.

    uv run python experiments/23-ocr-size/shrink.py

Only the size changes: same RGB mode, same aspect, PIL LANCZOS, saved as PNG (lossless).
s100 is 06's PNG itself and is not written again. sha256 per image -> results/size_manifest.json.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import sys
from pathlib import Path

import PIL
from PIL import Image

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
PNG06 = ROOT / "data/06-ocr/png"
SHRUNK = ROOT / "data/23-size"
SCALES = {"s83": 0.83, "s67": 0.67, "s50": 0.5, "s33": 0.33}  # about 250/200/150/100 dpi

logger = logging.getLogger(__name__)


def shrink(img: Image.Image, scale: float) -> bytes:
    size = (round(img.width * scale), round(img.height * scale))
    buf = io.BytesIO()
    img.resize(size, Image.Resampling.LANCZOS).save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    setup_logging("rag_lab")
    pages = sorted(PNG06.glob("*.png"))
    if not pages:
        raise SystemExit(f"no page images in {PNG06}: run 06's render.py first")
    manifest: dict[str, dict] = {}
    for name, scale in SCALES.items():
        out = SHRUNK / name
        out.mkdir(parents=True, exist_ok=True)
        for p in pages:
            data = shrink(Image.open(p), scale)
            (out / p.name).write_bytes(data)
            manifest.setdefault(name, {})[p.stem] = hashlib.sha256(data).hexdigest()
        logger.info("scale done", extra={"ctx": {"scale": name, "pages": len(pages)}})
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/size_manifest.json").write_text(
        json.dumps(
            {"pillow": PIL.__version__, "scales": SCALES, "sha256": manifest},
            ensure_ascii=False,
            indent=1,
        ),
        "utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
