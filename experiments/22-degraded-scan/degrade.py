"""06's 17 tables as poor scans: each PDF page rendered again at a lower dpi, then rotated,
blurred, noised and saved as a low-quality grayscale JPEG -> data/22-scan/<level>/<表號>-p<N>.jpg
plus data/22-scan/manifest.json (sha256 per image; a copy goes to results/ for git).

    uv run python experiments/22-degraded-scan/degrade.py

Synthetic, not real scans; the parameters were fixed before the run (README section 2).
The noise seed is crc32 of "<level>/<page>", so every machine draws the same noise.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import logging
import sys
import zlib
from pathlib import Path

import numpy as np
import pdfplumber
from PIL import Image, ImageFilter

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
ATTACH = ROOT / "data/01-chunking/attachments"
SCAN = ROOT / "data/22-scan"

logger = logging.getLogger(__name__)


def _tables() -> dict[str, str]:
    """04's 表號 -> FileId (06's 17 tables), loaded by path: 22 has its own build_corpus."""
    spec = importlib.util.spec_from_file_location(
        "build_corpus_04", HERE.parent / "04-table-lookup/build_corpus.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TABLES


LEVELS = {
    "L1": {"dpi": 200, "angle": 0.5, "blur": 0.0, "noise": 0.0, "quality": 50},  # an ordinary scan
    "L2": {"dpi": 150, "angle": 1.5, "blur": 0.8, "noise": 8.0, "quality": 30},  # a poor one
}


def seed_for(level: str, page: str) -> int:
    return zlib.crc32(f"{level}/{page}".encode())


def render(page, dpi: int) -> Image.Image:
    """The same call as 06's render.py, at another resolution."""
    return page.to_image(resolution=dpi, antialias=True).original


def scan(
    img: Image.Image, *, angle: float, blur: float, noise: float, quality: int, seed: int
) -> bytes:
    g = img.convert("L")
    g = g.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
    if blur:
        g = g.filter(ImageFilter.GaussianBlur(blur))
    if noise:
        x = np.asarray(g, dtype=np.float64)
        x = x + np.random.default_rng(seed).normal(0.0, noise, size=x.shape)
        g = Image.fromarray(np.clip(np.rint(x), 0, 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    g.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def main() -> int:
    setup_logging("rag_lab")
    manifest: dict[str, dict] = {}
    for tid, fid in _tables().items():
        with pdfplumber.open(ATTACH / f"{fid}.pdf") as pdf:
            for i, page in enumerate(pdf.pages):
                name = f"{tid}-p{i}"
                for level, p in LEVELS.items():
                    args = {k: v for k, v in p.items() if k != "dpi"}
                    raw = scan(render(page, p["dpi"]), **args, seed=seed_for(level, name))
                    out = SCAN / level / f"{name}.jpg"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(raw)
                    w, h = Image.open(io.BytesIO(raw)).size
                    manifest[f"{level}/{name}.jpg"] = {
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": len(raw),
                        "size": [w, h],
                    }
    text = json.dumps({"levels": LEVELS, "images": manifest}, ensure_ascii=False, indent=1)
    (SCAN / "manifest.json").write_text(text, "utf-8")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/scan_manifest.json").write_text(text, "utf-8")
    logger.info("scanned", extra={"ctx": {"images": len(manifest), "levels": list(LEVELS)}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
