"""文档分割服务模块 - 基于 LangChain 的智能文档分割"""
import re
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from loguru import logger

from app.config import config
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Markdown 表格行：以 | 开头、以 | 结尾（含表头行与 |---|---| 分隔行）
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
# 表格占位符：[[TABLE_0]] 形式，_protect_tables 写入、_restore_tables 还原
_PLACEHOLDER_RE = re.compile(r"\[\[TABLE_\d+\]\]")


def _find_table_blocks(text: str) -> List[Tuple[int, int]]:
    """标记连续表格行的 [start, end) 区间。"""
    lines = text.split("\n")
    blocks: List[Tuple[int, int]] = []
    i = 0
    n = len(lines)
    while i < n:
        if _TABLE_ROW_RE.match(lines[i]):
            j = i
            while j < n and _TABLE_ROW_RE.match(lines[j]):
                j += 1
            blocks.append((i, j))
            i = j
        else:
            i += 1
    return blocks


def _protect_tables(text: str) -> Tuple[str, dict]:
    """把表格块替换成带空行的占位符，返回 (缩减文本, {占位符: 表格原文})。"""
    blocks = _find_table_blocks(text)
    if not blocks:
        return text, {}
    lines = text.split("\n")
    tables: dict = {}
    out: List[str] = []
    prev = 0
    for idx, (start, end) in enumerate(blocks):
        out.extend(lines[prev:start])
        placeholder = f"[[TABLE_{idx}]]"
        tables[placeholder] = "\n".join(lines[start:end])
        # 用空行包住占位符，让递归切块把它当独立段落，切完单独成 chunk
        out.append("")
        out.append(placeholder)
        out.append("")
        prev = end
    out.extend(lines[prev:])
    return "\n".join(out), tables


class DocumentSplitterService:
    """文档分割服务 - 使用 LangChain 的分割器"""

    def __init__(self):
        # 初始化文档分割服务
        self.chunk_size = config.chunk_max_size
        self.chunk_overlap = config.chunk_overlap

        # Markdown 标题分割器（只按照一级和二级标题分割，减少分片数）
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#","h1"),
                ("##","h2"),
                # 不在按三级标题分割，避免过度碎片化
            ],
            strip_headers=False,# 保留标题在内容中
        )

        # 递归字符分割器（用于二次分割，使用更大的chunk_size）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size * 2, # 加倍chunk_size，减少分片数
            chunk_overlap = self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        logger.info(
            f"文档分割服务初始化完成，chunk_size = {self.chunk_size}, "
            f"secondary_chunk_szie = {self.chunk_size * 2},"
            f"overlap = {self.chunk_overlap},"
        )

    def split_markdown(self,content:str,file_path:str = "") -> List[Document]:
        """
        分割 Markdown 文档 (两阶段分割 + 合并小片段)

        Args:
            content: Markdown 内容
            file_path: 文件路径 (用于元数据)

        Returns:
            List[Document]: 文档分片列表
        """
        if not content or not content.strip():
            logger.warning(f"Markdown 文档内容为空：{file_path}")
            return []

        try:
            # 第一阶段：按标题分割
            md_docs = self.markdown_splitter.split_text(content)

            # 第二阶段：按大小进一步分割（表格原子保护：不把表格拦腰切断）
            docs_after_split: List[Document] = []
            for md_doc in md_docs:
                docs_after_split.extend(self._split_section_with_tables(md_doc))

            # 第三阶段：合并太小的分片（<300字符）
            final_docs = self._merge_small_chunks(docs_after_split, min_size=300)

            # 添加文件路径元数据（保留原始扩展名，便于追溯与权威度规则）
            for doc in final_docs:
                doc.metadata["_source"] = file_path
                doc.metadata["_extension"] = Path(file_path).suffix
                doc.metadata["_file_name"] = Path(file_path).name

            logger.info(f"Markdown 分割完成：{file_path} -> {len(final_docs)} 个分片")
            return final_docs

        except Exception as e:
            logger.error(f"Markdown 分割失败：{file_path},错误：{e}")
            raise

    def split_text(self,content:str,file_path:str = "") -> List[Document]:
        """
          分割普通文本文档

          Args:
              content: 文本内容
              file_path: 文件路径 (用于元数据)

          Returns:
              List[Document]: 文档分片列表
        """
        if not content or not content.strip():
            logger.warning(f"文本内容为空：{file_path}")
            return []

        try:
            # 直接使用递归字符分割器
            docs = self.text_splitter.create_documents(
                texts = [content],
                metadatas = [
                    {
                        "_source": file_path,
                        "_extension": Path(file_path).suffix,
                        "_file_name": Path(file_path).name,
                    }
                ],
            )

            logger.info(f"文本分割完成：{file_path} ->{len(docs)}个分片")
            return docs

        except Exception as e:
            logger.error(f"文本分割失败：{file_path},错误:{e}")
            raise

    def split_document(self,content:str,file_path:str = "") -> List[Document]:
        """
        智能分割文档 (根据文件类型选择分割器)

        Args:
            content: 文档内容
            file_path: 文件路径

        Returns:
            List[Document]: 文档分片列表
        """
        if file_path.endswith(".md"):
            return self.split_markdown(content,file_path)
        else:
            return self.split_text(content,file_path)

    def _split_section_with_tables(self, section: Document) -> List[Document]:
        """对单个标题段做「表格原子保护」的二次切块。

        无表格段直接走递归切块；含表格段先把表格替换成占位符再切，切完把表格整块拼回，
        超长表格按行拆、重挂表头，避免 RecursiveCharacterTextSplitter 把表格拦腰切断。
        """
        text = section.page_content
        reduced, tables = _protect_tables(text)
        if not tables:
            return self.text_splitter.split_documents([section])

        split_docs = self.text_splitter.split_documents(
            [Document(page_content=reduced, metadata=section.metadata)]
        )
        out: List[Document] = []
        for doc in split_docs:
            out.extend(self._restore_tables(doc.page_content, doc.metadata, tables))
        return out

    def _restore_tables(self, content: str, metadata: dict, tables: dict) -> List[Document]:
        """把切块里的表格占位符还原成真实表格。

        递归切块可能把占位符和前后文本合成一块（比如「标题 + 超长表格」整体没超长时不会
        拆开），所以不能只在整块恰等于占位符时才还原。这里按占位符边界切成段落：占位符
        展开成表格（超长按行拆、重挂表头），紧邻的文本段拼到第一块表格上，不丢上下文。
        """
        segments = _PLACEHOLDER_RE.split(content)
        placeholders = _PLACEHOLDER_RE.findall(content)
        docs: List[Document] = []
        for i, seg in enumerate(segments):
            seg = seg.strip()
            if i < len(placeholders):
                table = tables.get(placeholders[i], placeholders[i])
                parts = self._chunk_table(table, self.chunk_size * 2)
                if seg:
                    parts[0] = seg + "\n\n" + parts[0]
                docs.extend(Document(page_content=p, metadata=metadata) for p in parts)
            elif seg:
                docs.append(Document(page_content=seg, metadata=metadata))
        return docs

    @staticmethod
    def _chunk_table(table_text: str, max_size: int) -> List[str]:
        """超长表格按行拆分，每片重挂表头行 + 分隔行；未超长则原样返回。"""
        lines = table_text.split("\n")
        if len(table_text) <= max_size or len(lines) < 3:
            return [table_text]
        header = lines[:2]  # 表头行 + |---|---| 分隔行
        body = lines[2:]
        chunks: List[str] = []
        cur = list(header)
        cur_len = sum(len(line) for line in header)
        for row in body:
            if cur_len + len(row) + 1 > max_size and len(cur) > len(header):
                chunks.append("\n".join(cur))
                cur = list(header)
                cur_len = sum(len(line) for line in header)
            cur.append(row)
            cur_len += len(row) + 1
        if cur:
            chunks.append("\n".join(cur))
        return chunks

    def _merge_small_chunks(
            self,
            documents:List[Document],
            min_size:int = 300
    ) -> List[Document]:
        """
        合并太小的分片

        Args:
            documents: 文档列表
            min_size: 最小分片大小 (字符数)

        Returns:
            List[Document]: 合并后的文档列表
        """
        if not documents:
            return []

        merged_docs = []
        current_doc = None

        for doc in documents:
            doc_size = len(doc.page_content)

            if current_doc is None:
                # 第一个文档
                current_doc = doc
            elif doc_size < min_size and len(current_doc.page_content) + doc_size <= self.chunk_size * 2:
                # 当前文档太小，且合并后不会超过上限，则合并
                current_doc.page_content += "\n\n" + doc.page_content
                # 保留主文档的元数据
            else:
                # 保存当前文档，开始新文档
                merged_docs.append(current_doc)
                current_doc = doc

        # 添加最后一个文档
        if current_doc is not None:
            merged_docs.append(current_doc)

        return merged_docs

# 全局单例
document_splitter_service = DocumentSplitterService()