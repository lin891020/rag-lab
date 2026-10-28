"""The MOJ page of the distractor regulation -> lines shaped like law.pdf's text layer."""

from build_corpus import moj_lines

HTML = """
<div class="h3 char-2">   第 一 章 總則</div>
<div class="h3 char-3">      第 一 節 通則</div>
<div class="row"><div class="col-no"> <a href="x" name="1">第 1 條</a></div><div class="col-data"><div class="law-article">
<div class="line-0000">本規則依電業法第二十五條第三項規定訂定之。</div></div>
</div></div>
<div class="row"><div class="col-no"> <a href="x" name="2">第 2 條</a></div><div class="col-data"><div class="law-article">
<div class="line-0000 show-number">本規則目的為提供安全防護。</div><div class="line-0000 show-number">本規則包括基本規定。</div></div>
</div></div>
<div class="row"><div class="col-no"> <a href="x" name="3">第 3 條</a></div><div class="col-data"><div class="law-article">
<div class="line-0000 show-number">本規則適用範圍如下：</div><div class="line-0004">一、電業之供電線路。</div><div class="line-0004">二、接戶點之設施。</div></div>
</div></div>
"""


def test_moj_lines_mimic_the_pdf_text_layer():
    lines = moj_lines(HTML)
    assert lines == [
        "   第 一 章 總則",
        "      第 一 節 通則",
        "第 1 條",
        "本規則依電業法第二十五條第三項規定訂定之。",
        "第 2 條",
        "1   本規則目的為提供安全防護。",
        "2   本規則包括基本規定。",
        "第 3 條",
        "本規則適用範圍如下：",
        "一、電業之供電線路。",
        "二、接戶點之設施。",
    ]
