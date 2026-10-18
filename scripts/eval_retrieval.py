"""
RAG 检索质量量化评估脚本（golden 集 + recall@k / precision@k / hit@k + 参数扫描）。

针对「检索环节」单独量化（不经过 LLM，纯向量检索），回答两个问题：
1. 当前 rag_top_k=3 / rag_relevance_threshold=0.5 的真实召回水平是多少？
2. 有没有更优的 (top_k, threshold) 组合？

原理：
- 读 golden 集（scripts/eval/retrieval_golden.json），每条是「自然语言查询 → 预期命中的
  知识库文件名（runbook）」。
- 对每条 query 调用 vector_store.similarity_search_with_score(query, k=10)，拿到带
  L2 距离的排序结果；距离经 l2_distance_to_relevance 映射为相关性分数（与线上
  retrieve_knowledge 完全一致）。
- 以「文件」为相关性单位：某 chunk 命中 = 其 metadata._file_name == golden 的
  relevant_file。据此计算：
    hit@k      —— 前 k 个结果里是否出现了正确文件（= 单文件场景下的 recall@k）
    precision@k—— 前 k 个结果里来自正确文件的 chunk 占比
- 最后扫描 (top_k × relevance_threshold) 网格，输出每个组合下的命中率，据此判断
  当前 magic number 是否合理、要不要调。

用法：
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --golden scripts/eval/retrieval_golden.json
    python scripts/eval_retrieval.py --max-k 20

注意：依赖本地 Milvus（localhost:19530）已连接并已把 aiops-docs/ 索引进去，且 .env
里有 DASHSCOPE_API_KEY（embedding 需要真实调用）。非 pytest，不进 CI。
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

# 让 `python scripts/eval_retrieval.py` 也能从项目根导入 app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import config  # noqa: E402
from app.core.milvus_client import milvus_manager  # noqa: E402
from app.eval_metrics import hit_k, load_retrieval_golden, precision_k  # noqa: E402
from app.services.vector_store_manager import vector_store_manager  # noqa: E402
from app.tools.knowledge_tool import l2_distance_to_relevance  # noqa: E402


def run_retrieval(golden_path: str, max_k: int = 10, top_k: int | None = None, threshold: float | None = None) -> dict:
    """跑 golden 集，返回线上口径指标（top_k/threshold 过滤后的 hit@3 / precision@3）。

    供 run_eval.py 统一入口复用；top_k/threshold 默认取线上 config（rag_top_k /
    rag_relevance_threshold），保证评测口径与线上一致。
    """
    if top_k is None:
        top_k = config.rag_top_k
    if threshold is None:
        threshold = config.rag_relevance_threshold

    golden = load_retrieval_golden(golden_path)
    milvus_manager.connect()

    ranked_all = [retrieve(item["query"], k=max_k) for item in golden]
    n = len(golden)
    hits = 0
    precs = 0.0
    for r, g in zip(ranked_all, golden):
        filtered = [(fn, s) for fn, s in r if s >= threshold][:top_k]
        hits += int(hit_k(filtered, g["relevant_file"], top_k))
        precs += precision_k(filtered, g["relevant_file"], top_k)

    return {
        "n": n,
        "top_k": top_k,
        "threshold": threshold,
        "hit_at_3": hits / n if n else 0.0,
        "precision_at_3": precs / n if n else 0.0,
    }


def retrieve(query: str, k: int) -> list:
    """返回 [(file_name, relevance_score), ...]，按分数降序，最多 k 条。"""
    scored = vector_store_manager.get_vector_store().similarity_search_with_score(query, k=k)
    out = []
    for doc, distance in scored:
        file_name = (doc.metadata or {}).get("_file_name", "未知来源")
        out.append((file_name, l2_distance_to_relevance(distance)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索质量量化评估")
    parser.add_argument("--golden", default=str(Path(__file__).resolve().parent / "eval" / "retrieval_golden.json"))
    parser.add_argument("--max-k", type=int, default=10, help="检索时向 Milvus 请求的最大 k")
    args = parser.parse_args()

    golden = load_retrieval_golden(args.golden)
    max_k = args.max_k

    # 连接 Milvus（复用生产唯一连接；索引数据已在 aiops-docs 导入）
    milvus_manager.connect()

    ranked_all = []
    for item in golden:
        ranked = retrieve(item["query"], k=max_k)
        ranked_all.append(ranked)
        print(f"\n【{item['topic']}】{item['query']}")
        print(f"    预期命中：{item['relevant_file']}")
        for i, (fn, score) in enumerate(ranked, 1):
            mark = "  <-- 命中" if fn == item["relevant_file"] else ""
            print(f"    #{i} {fn:24s} score={score:.3f}{mark}")

    # ---------- 命中率 / 精确率 @k ----------
    ks = [1, 2, 3, 5, 10]
    print("\n" + "=" * 72)
    print("按 k 的指标（文件为相关性单位，N=%d）" % len(golden))
    print("=" * 72)
    header = f"{'k':>4} | {'hit@k':>8} | {'precision@k':>12}"
    print(header)
    print("-" * len(header))
    for k in ks:
        if k > max_k:
            continue
        hits = sum(hit_k(r, g["relevant_file"], k) for r, g in zip(ranked_all, golden))
        prec = sum(precision_k(r, g["relevant_file"], k) for r, g in zip(ranked_all, golden)) / len(golden)
        print(f"{k:>4} | {hits/len(golden):>7.1%} | {prec:>11.1%}")

    # ---------- 相关性分数分布（判断 threshold 该卡在哪） ----------
    print("\n" + "=" * 72)
    print("top-1 命中/未命中时的相关性分数分布（判断 threshold 是否误伤）")
    print("=" * 72)
    hit_scores, miss_scores = [], []
    for r, g in zip(ranked_all, golden):
        if not r:
            continue
        # top-1 是否是正确文件
        if r[0][0] == g["relevant_file"]:
            hit_scores.append(r[0][1])
        else:
            miss_scores.append(r[0][1])
    def _dist(scores, label):
        if not scores:
            print(f"  {label}: （无样本）")
            return
        lo, hi = min(scores), max(scores)
        avg = sum(scores) / len(scores)
        print(f"  {label}: n={len(scores)}  min={lo:.3f}  avg={avg:.3f}  max={hi:.3f}")
    _dist(hit_scores, "top-1 命中正确文件 ")
    _dist(miss_scores, "top-1 未命中         ")

    # ---------- (top_k × threshold) 扫描 ----------
    thresholds = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]
    print("\n" + "=" * 72)
    print("参数扫描：不同 (top_k, relevance_threshold) 下的 hit@k（正确文件是否被喂给 LLM）")
    print("（当前线上配置 = top_k=3, threshold=0.5）")
    print("=" * 72)
    print(f"{'threshold':>9} | " + " | ".join(f"k={k:>2}" for k in ks if k <= max_k))
    print("-" * (9 + 4 + sum(7 for k in ks if k <= max_k)))
    for th in thresholds:
        cells = []
        for k in ks:
            if k > max_k:
                continue
            hits = 0
            for r, g in zip(ranked_all, golden):
                # 先按阈值过滤，再取前 k
                filtered = [(fn, s) for fn, s in r if s >= th][:k]
                hits += any(fn == g["relevant_file"] for fn, _ in filtered)
            cells.append(f"{hits/len(golden):>5.1%}")
        print(f"{th:>9.3f} | " + " | ".join(cells))

    print("\n结论提示：")
    print("  - hit@k 反映「正确 runbook 有没有被检索到」；precision@k 反映「喂进去的噪声占比」。")
    print("  - 若某 threshold 下 hit@k 骤降，说明该阈值把正确 chunk 也过滤掉了，需下调。")
    print("  - 优先选 hit@k 高且 threshold 能压低噪声的 (top_k, threshold) 组合。")


if __name__ == "__main__":
    main()
