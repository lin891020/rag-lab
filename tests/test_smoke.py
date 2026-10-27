import logging

from rag_lab.log import setup_logging


def test_logging_boots(tmp_path, caplog):
    setup_logging("rag_lab", log_dir=tmp_path)
    logging.getLogger("rag_lab.smoke").info("alive")
    assert "alive" in caplog.text
