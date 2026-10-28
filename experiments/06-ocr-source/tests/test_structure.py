from structure import compare, ocr_rows


def test_compare_counts_cells_rows_and_text():
    gt = {"rows": [["38 - 50", "14"], ["60 - 80", "22"]]}
    md = "表\n| 38-50 | 14 |\n|---|---|\n| 60 - 80 | 2Z |\n註：22"
    r = compare(gt, md)
    assert r["n_cells"] == 4 and r["n_nums"] == 4
    assert r["cell_recall"] == 0.75 and r["row_recall"] == 0.5
    assert r["text_recall"] == 1.0  # "22" is still somewhere in the text


def test_ocr_rows_skips_separator_and_prose():
    assert ocr_rows("title\n| a | b |\n|---|---|\n| 1 | 2 |\nnote") == [["a", "b"], ["1", "2"]]
