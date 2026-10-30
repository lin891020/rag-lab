"""The generated bank: situation only (no table or article ids), every given number present,
a fifth abstain, clusters capped, ids in order, expectations consistent with rules.py."""

import json
import re
from collections import Counter
from pathlib import Path

from build_questions import MAX_PER_ROW, MAX_PER_ROW_ABSTAIN, MAX_PER_ROW_C4, _number_present

HERE = Path(__file__).parents[1]
QS = [
    json.loads(ln)
    for ln in (HERE / "questions.jsonl").read_text("utf-8").splitlines()
    if ln.strip()
]
LEAK = re.compile(r"表[一二三四五六七八九十○〇～]+|第\s*[一二三四五六七八九十百\d]+\s*條")


def test_shape_ids_and_abstain_share():
    assert len(QS) == 160
    per = Counter(q["chain"] for q in QS)
    assert per == {"C1": 32, "C2": 32, "C3": 32, "C4": 32, "C5": 32}
    for chain in per:
        ids = [q["id"] for q in QS if q["chain"] == chain]
        assert ids == [f"{chain.lower()}-{i:02d}" for i in range(1, 33)]
        assert sum(q["expect"]["kind"] == "abstain" for q in QS if q["chain"] == chain) == 6


def test_wording_gives_the_situation_only():
    for q in QS:
        assert not LEAK.search(q["question"]), q["id"]
        for v in q["given"].values():
            assert _number_present(q["question"], v), (q["id"], v)
        if q["expect"]["kind"] != "abstain":  # the answer (value + its unit) is not in the text
            v, u = q["expect"]["value"], q["expect"]["unit"]
            if v not in q["given"].values():  # a 100 % demand factor legitimately echoes the load
                assert f"{v:g}{u}" not in q["question"].replace(",", "").replace(" ", ""), q["id"]


def test_expectations_are_well_formed():
    for q in QS:
        e = q["expect"]
        if e["kind"] == "abstain":
            assert e["value"] is None and q["abstain_reason"], q["id"]
        else:
            assert isinstance(e["value"], (int, float)) and e["value"] > 0, q["id"]
            assert q["inputs"], q["id"]
        if e["kind"] == "ladder":
            assert e["ladder"] and e["value"] in e["ladder"], q["id"]
        assert q["gold"]["articles"] and q["gold"]["rows"], q["id"]


def test_clusters_are_capped():
    for abstain in (True, False):
        c = Counter(q["cluster"] for q in QS if (q["expect"]["kind"] == "abstain") == abstain)
        for cluster, n in c.items():
            chain = next(q["chain"] for q in QS if q["cluster"] == cluster)
            cap = (
                MAX_PER_ROW_ABSTAIN
                if abstain
                else (MAX_PER_ROW_C4 if chain == "C4" else MAX_PER_ROW)
            )
            assert n <= cap, (cluster, n)


def test_lighting_questions_say_the_factor_is_tiered():
    for q in QS:
        if q["variant"] == "lighting":
            assert "分段累進" in q["question"], q["id"]


def test_hospital_questions_do_not_tell_the_model_to_apply_the_factor():
    for q in QS:
        if q["variant"] == "hospital-na":
            assert "分段累進" not in q["question"] and q["gold"]["rows"] == [["表五二", "註"]], q[
                "id"
            ]
