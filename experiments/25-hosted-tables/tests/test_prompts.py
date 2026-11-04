"""06's prompt for every condition; only the evidence block changes. clean and glm2 must be the
prompts 24 sent; the image conditions name the attached pages instead of quoting a table."""

from pathlib import Path

from prompts import IMAGES, SCHEMA, SYSTEM, TEXT, image_prompt, user_prompt


def test_conditions():
    assert list(TEXT) == ["clean", "glm2"] and list(IMAGES) == ["image", "image-L2"]
    assert list(SCHEMA["properties"]) == ["quote", "value", "unit"] and SYSTEM


def _outside(p):
    return p.split("依據（")[0], p.split("問題：")[1]


def test_image_prompt_differs_only_in_evidence():
    a = user_prompt("幾安培？", evidence="| a | b |", label="表一")
    b = image_prompt("幾安培？", label="表一", pages=2)
    assert _outside(a) == _outside(b)
    assert "2 張" in b and "| a | b |" not in b


def test_clean_and_glm2_are_the_prompts_24_sent():
    from run import last_rows, load_corpus, load_questions, text_prompt

    raw24 = Path(__file__).parents[2] / "24-ocr-header/results/raw.jsonl"
    sent = {(r["condition"], r["qid"]): r["prompt_len"] for r in last_rows(raw24)}
    corpus = load_corpus()
    for q in load_questions():
        for c in ("clean", "glm2"):
            assert len(text_prompt(c, q, corpus)) == sent[(c, q["id"])], (c, q["id"])
