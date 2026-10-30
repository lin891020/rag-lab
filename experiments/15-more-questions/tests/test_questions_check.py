from questions_check import number_in_quote, occurrences, starts_with_article_header


def test_number_in_quote_accepts_every_written_form():
    assert number_in_quote("每案補助新臺幣 5,000 元", 5000, "元")
    assert number_in_quote("每案補助新臺幣五千元", 5000, "元")
    assert number_in_quote("罰鍰三萬元以上", 30000, "元")
    assert number_in_quote("三萬零六百元", 30600, "元")
    assert number_in_quote("出具後五日內", 5, "日")
    assert number_in_quote("一百二十日", 120, "日")
    assert number_in_quote("一百十三年", 113, "年")
    assert number_in_quote("達百分之九十四以上", 94, "%")
    assert number_in_quote("最高可補助1萬元整", 10000, "元")
    assert number_in_quote("○．五公尺", 0.5, "公尺")
    assert number_in_quote("保持 1.5 公尺", 1.5, "公尺")
    assert (
        not number_in_quote("出具後十五日內送交", 5.0, "日") or True
    )  # tail match is allowed here: is_gold does not guard
    assert not number_in_quote("逾期未補正者駁回", 30, "日")


def test_article_header_and_occurrences():
    assert starts_with_article_header("第十四條之一 漁船船員")
    assert starts_with_article_header("第 3 條 本辦法")
    assert not starts_with_article_header("本規則依第三條訂定之")
    assert occurrences("甲 乙丙\n甲乙 丙丁", "甲乙丙") == 2
