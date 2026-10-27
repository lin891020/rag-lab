from pathlib import Path

from rag_lab.pdf import extract_text, table_to_text

FIX = Path(__file__).parent / "fixtures"
LAW = Path(__file__).parents[1] / "experiments/01-chunking/corpus/law.pdf"


def test_extract_text_keeps_article_headers():
    text = extract_text(LAW)
    assert "第 1 條\n本規則依電業法第三十二條第五項規定訂定之。" in text
    assert text.count("第 1014 條") == 1


def test_table_to_text_first_line_is_the_table_title():
    assert table_to_text(FIX / "table8.pdf").splitlines()[0] == "表八 低壓用電設備最小工作空間"


def test_table_to_text_skips_formula_drawn_above_title():
    # 表八六～一 has a stray "−πR" glyph line before the title in the PDF
    assert table_to_text(FIX / "table86-1.pdf").splitlines()[0].startswith("表八六～一")


def test_table_to_text_keeps_one_row_per_line_for_dense_tables():
    # pdfplumber's grid extraction folds this table's rows into single cells; plain text keeps them
    text = table_to_text(FIX / "table25-2.pdf")
    assert "3.5 7/0.8 19 17 15 13" in text.splitlines()
    assert "註：" in text
