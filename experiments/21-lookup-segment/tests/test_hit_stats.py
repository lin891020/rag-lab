"""A gold row whose band holds nothing is not needed (c4-16: 6 kVA of receptacles, only the
10以下 band): hit_stats does not count it, so looking up the one band is a full hit (19's review)."""

import json
from pathlib import Path

from evidence import hit_stats

Q = {
    q["id"]: q
    for q in map(
        json.loads, (Path(__file__).parents[1] / "questions.jsonl").read_text("utf-8").splitlines()
    )
}


def test_unused_band_is_not_counted():
    h = hit_stats({"表五三|10以下"}, Q["c4-16"])
    assert h["rows_hit"] == h["rows_total"] == 1


def test_two_bands_still_need_both():
    h = hit_stats({"表五二|大賣場(倉儲) 12,500以下"}, Q["c4-20"])
    assert (h["rows_hit"], h["rows_total"]) == (1, 2)
