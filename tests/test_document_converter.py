"""文档转换（任意格式 → Markdown）单元测试。"""
import builtins

import pytest

from app.config import config
from app.services import document_converter as dc


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_txt_read_directly(tmp_path):
    assert dc.load_markdown(_write(tmp_path, "a.txt", "纯文本内容")) == "纯文本内容"


def test_md_read_directly(tmp_path):
    assert dc.load_markdown(_write(tmp_path, "a.md", "# 标题\n正文")) == "# 标题\n正文"


def test_html_converted_to_markdown(tmp_path):
    p = _write(tmp_path, "a.html", "<html><body><h1>标题</h1><p>正文 hello</p></body></html>")
    md = dc.load_markdown(p)
    assert "标题" in md
    assert "正文 hello" in md


def test_unsupported_extension_raises(tmp_path):
    with pytest.raises(RuntimeError, match="不支持的文件格式"):
        dc.load_markdown(_write(tmp_path, "a.exe", "x"))


def test_conversion_disabled_blocks_non_text(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "doc_conversion_enabled", False)
    with pytest.raises(RuntimeError, match="DOC_CONVERSION_ENABLED"):
        dc.load_markdown(_write(tmp_path, "a.pdf", "x"))


def test_conversion_disabled_still_allows_text(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "doc_conversion_enabled", False)
    assert dc.load_markdown(_write(tmp_path, "a.txt", "文本")) == "文本"


def test_missing_markitdown_raises_clear_error(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "markitdown":
            raise ImportError("no markitdown")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(RuntimeError, match="markitdown"):
        dc.load_markdown(_write(tmp_path, "a.pdf", "x"))


def test_extension_sets_are_consistent():
    assert dc.SUPPORTED_EXTENSIONS == dc.TEXT_EXTENSIONS | dc.CONVERT_EXTENSIONS
    assert dc.is_supported("a.pdf")
    assert dc.is_supported("a.docx")
    assert dc.is_supported("a.md")
    assert not dc.is_supported("a.exe")
