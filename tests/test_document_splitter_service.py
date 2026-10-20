"""文档分割（Markdown 结构切块 + 表格原子保护）单元测试。"""
import re

from app.services.document_splitter_service import (
    _find_table_blocks,
    _protect_tables,
    document_splitter_service,
)

_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_SEP_ROW = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")


def test_find_table_blocks():
    # _find_table_blocks 返回 0 索引的 [start, end) 区间：表格行是第 2~4 行
    text = "# 标题\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\n正文"
    assert _find_table_blocks(text) == [(2, 5)]


def test_protect_tables_replaces_with_placeholder():
    text = "前文\n\n| A |\n| --- |\n| 1 |\n\n后文"
    reduced, tables = _protect_tables(text)
    assert "[[TABLE_0]]" in reduced
    assert "| A |" not in reduced
    assert "| A |" in tables["[[TABLE_0]]"]


def test_chunk_table_reattaches_header_when_oversized():
    header = "| 名称 | 值 |\n| --- | --- |"
    rows = [f"| 行{i} | 值{i} |" for i in range(100)]
    table = header + "\n" + "\n".join(rows)
    chunks = document_splitter_service._chunk_table(table, max_size=200)
    assert len(chunks) > 1
    for c in chunks:
        lines = c.split("\n")
        assert lines[0].startswith("| 名称")  # 每个片段重挂表头行
        assert _SEP_ROW.match(lines[1])       # 表头行紧跟分隔行


def test_split_markdown_keeps_table_rows_intact():
    # 一个标题下挂一张超长表格（> chunk_size*2=1600），验证不把表头/行切散、不丢上下文
    header = "| 名称 | 值 |\n| --- | --- |"
    rows = [f"| 服务{i} | 阈值说明文字{i} |" for i in range(120)]
    content = "# 参数表\n\n" + header + "\n" + "\n".join(rows)

    docs = document_splitter_service.split_markdown(content, "params.md")

    table_chunks = []
    for d in docs:
        lines = [ln for ln in d.page_content.split("\n") if _TABLE_ROW.match(ln)]
        if lines:
            table_chunks.append((d.page_content, lines))

    assert table_chunks  # 确实切出了含表格的 chunk
    for _content_text, lines in table_chunks:
        assert lines[0].startswith("| 名称")  # 含表格的 chunk 必带表头，不丢上下文


def test_split_markdown_preserves_original_extension():
    docs = document_splitter_service.split_markdown("# 标题\n\n正文内容", "runbook.pdf")
    assert docs
    for d in docs:
        assert d.metadata["_extension"] == ".pdf"
        assert d.metadata["_file_name"] == "runbook.pdf"
