"""应用层 BM25 关键词索引（稀疏召回路）。

为什么不直接用 Milvus 原生 BM25/稀疏向量：那需要改 collection schema 并全量重建
（重新向量化所有已入库文档），对已部署系统风险大。这里改为纯 Python 的 Okapi BM25，
数据源仍是 Milvus（唯一真相），用「读全量 chunk → 内存建索引」的方式零迁移实现
关键词召回；语料是内部 runbook，体量小，内存索引瞬时、完全够用。

索引在首次 search 时惰性构建，之后缓存；语料变更（写入/删除）时调用 mark_dirty()
失效，下次 search 重建。任何异常都 fail-open（返回空结果），不影响稠密召回主路。
"""
import json
import math
import re
import threading
from typing import List, Tuple

from langchain_core.documents import Document
from loguru import logger

from app.core.milvus_client import milvus_manager

# 关键词索引分页读取的每批条数（远小于 Milvus 单次 query 上限 16384）
_BATCH = 1000

_ASCII_RUN = re.compile(r"[a-z0-9_]+")


def _is_cjk(ch: str) -> bool:
    """判断单字符是否属于 CJK 表意文字（中日韩统一表意 + 扩展 A + 兼容区）。"""
    cp = ord(ch)
    return 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0xF900 <= cp <= 0xFAFF


def tokenize(text: str) -> List[str]:
    """把文本切成索引词元。

    规则：
    - ASCII 字母/数字/下划线连续段整段保留（`cpu`、`e500`、`data_sync` 这类精确词）；
    - 中文按「单字 + 相邻二字 bigram」切（中文无空格，bigram 能命中短语/专名片段）。
    """
    text = (text or "").lower()
    tokens: List[str] = []
    ascii_buf: List[str] = []
    cjk_buf: List[str] = []

    def flush_ascii() -> None:
        if ascii_buf:
            tokens.append("".join(ascii_buf))
            ascii_buf.clear()

    def flush_cjk() -> None:
        if not cjk_buf:
            return
        tokens.extend(cjk_buf)  # 单字
        for a, b in zip(cjk_buf, cjk_buf[1:]):  # 相邻二字
            tokens.append(a + b)
        cjk_buf.clear()

    for ch in text:
        if ch.isascii() and (ch.isalnum() or ch == "_"):
            flush_cjk()
            ascii_buf.append(ch)
        elif _is_cjk(ch):
            flush_ascii()
            cjk_buf.append(ch)
        else:
            flush_ascii()
            flush_cjk()

    flush_ascii()
    flush_cjk()
    return tokens


class BM25:
    """标准 Okapi BM25（k1=1.5, b=0.75）。docs 为已分词词元列表。"""

    def __init__(self, docs: List[List[str]]):
        self.k1 = 1.5
        self.b = 0.75
        self.docs = docs
        self.n = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0

        # 文档频率 df[term]，用于算 IDF
        self.df: dict = {}
        for d in docs:
            for term in set(d):
                self.df[term] = self.df.get(term, 0) + 1

        # IDF：Robertson-Sparck Jones 平滑
        self.idf: dict = {}
        for term, f in self.df.items():
            self.idf[term] = math.log(1.0 + (self.n - f + 0.5) / (f + 0.5))

    def score(self, query_tokens: List[str], doc_index: int) -> float:
        doc = self.docs[doc_index]
        dl = self.doc_len[doc_index]
        tf: dict = {}
        for t in doc:
            tf[t] = tf.get(t, 0) + 1

        s = 0.0
        for q in query_tokens:
            if q not in self.idf:
                continue
            f = tf.get(q, 0)
            if f == 0:
                continue
            denom = f + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
            s += self.idf[q] * f * (self.k1 + 1.0) / denom
        return s


class KeywordIndex:
    """关键词索引门面：惰性构建 + 脏标记失效 + 线程安全。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._docs: List[Document] = []
        self._tokens: List[List[str]] = []
        self._bm25: BM25 | None = None
        self._dirty = True

    def mark_dirty(self) -> None:
        """语料变更后调用，下次 search 时重建索引。"""
        with self._lock:
            self._dirty = True

    def _ensure_built(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            self._docs, self._tokens = self._load_corpus()
            self._bm25 = BM25(self._tokens)
            self._dirty = False

    def _load_corpus(self) -> Tuple[List[Document], List[List[str]]]:
        """从 Milvus 分页读全量 chunk 构建语料；失败返回空（fail-open）。"""
        try:
            collection = milvus_manager.get_collection()
        except Exception as e:  # noqa: BLE001 —— Milvus 未连接/不可用时索引为空，回退稠密路
            logger.debug(f"Milvus 不可用，关键词索引为空：{e}")
            return [], []

        docs: List[Document] = []
        tokens: List[List[str]] = []
        try:
            offset = 0
            while True:
                rows = collection.query(
                    expr="",
                    output_fields=["content", "metadata"],
                    offset=offset,
                    limit=_BATCH,
                )
                if not rows:
                    break
                for row in rows:
                    content = row.get("content") or ""
                    metadata = row.get("metadata") or {}
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata) if metadata else {}
                    docs.append(Document(page_content=content, metadata=metadata))
                    tokens.append(tokenize(content))
                offset += len(rows)
                if len(rows) < _BATCH:
                    break
        except Exception as e:  # noqa: BLE001
            logger.warning(f"关键词索引构建失败，退化为仅向量召回：{e}")
            return [], []

        logger.info(f"关键词索引构建完成：{len(docs)} 个 chunk")
        return docs, tokens

    def search(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        """返回 [(doc, bm25_score), ...]，按分数降序，最多 k 条（跳过 0 分）。"""
        self._ensure_built()
        if not self._docs or self._bm25 is None:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scored = [(self._bm25.score(q_tokens, i), i) for i in range(self._bm25.n)]
        scored.sort(key=lambda x: x[0], reverse=True)

        out = []
        for s, i in scored[:k]:
            if s <= 0.0:
                continue
            out.append((self._docs[i], s))
        return out


# 全局单例（跨检索请求共享缓存）
keyword_index = KeywordIndex()
