"""评测纯函数：检索/诊断打分、报告汇总、基线回归对比。

只依赖标准库，不 import Milvus / LLM 等运行时组件，因此可被 tests 直接
确定性测试；scripts/ 下的评测脚本与统一入口 run_eval.py 复用这里的打分逻辑，
保证「评测口径」与「线上口径」一致。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def load_retrieval_golden(path: str) -> list:
    """读检索 golden 集（JSON 数组，每项含 query / relevant_file）。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"golden 集应为 JSON 数组：{path}")
    for item in data:
        for key in ("query", "relevant_file"):
            if key not in item:
                raise ValueError(f"golden 条目缺少字段 {key!r}：{item}")
    return data


def load_diagnosis_scenarios(path: str) -> list:
    """读诊断场景：可以是单个 JSON 文件或目录（按文件名排序读取 *.json）。"""
    p = Path(path)
    if p.is_file():
        return [json.loads(p.read_text(encoding="utf-8"))]
    files = sorted(p.glob("*.json"))
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def hit_k(ranked: list, relevant_file: str, k: int) -> bool:
    """前 k 个结果里是否出现正确文件。ranked 形如 [(file_name, score), ...]。"""
    return any(fn == relevant_file for fn, _ in ranked[:k])


def precision_k(ranked: list, relevant_file: str, k: int) -> float:
    """前 k 个结果里来自正确文件的 chunk 占比。"""
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for fn, _ in top if fn == relevant_file) / len(top)


def score_response(response: str, expected: dict) -> tuple[bool, list]:
    """按根因关键词打分：命中 >= min_keyword_hits 即通过。返回 (是否通过, 命中词)。"""
    keywords = expected.get("root_cause_keywords", [])
    min_hits = expected.get("min_keyword_hits", 1)
    low = (response or "").lower()
    matched = [k for k in keywords if k.lower() in low]
    return len(matched) >= min_hits, matched


# 回归对比容差：命中率是正确性硬门槛，跌一点即退化；LLM 非确定，准确率给 10% 缓冲。
DEFAULT_TOLERANCES = {
    "retrieval.hit@3": 0.0,
    "retrieval.precision@3": 0.10,
    "diagnosis.accuracy": 0.10,
}


def build_report(retrieval_metrics: dict, diagnosis_metrics: dict) -> dict:
    """汇总成统一报告结构。"""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "retrieval": retrieval_metrics,
        "diagnosis": diagnosis_metrics,
    }


def scalar_metrics(report: dict) -> dict[str, float]:
    """提取报告中用于回归对比的标量指标（只含实际跑过的 stage）。"""
    r = report.get("retrieval") or {}
    d = report.get("diagnosis") or {}
    out: dict[str, float] = {}
    if "hit_at_3" in r:
        out["retrieval.hit@3"] = r["hit_at_3"]
    if "precision_at_3" in r:
        out["retrieval.precision@3"] = r["precision_at_3"]
    if "accuracy" in d:
        out["diagnosis.accuracy"] = d["accuracy"]
    return out


def check_regression(current: dict, baseline: dict, tolerances: dict | None = None) -> list[str]:
    """对比当前报告与基线，返回退化项列表（空 = 无退化）。

    只对比两边都存在的指标（单 stage 运行时缺失的 stage 不算退化）。
    """
    tolerances = tolerances or DEFAULT_TOLERANCES
    cur = scalar_metrics(current)
    base = scalar_metrics(baseline)
    regressions: list[str] = []
    for name, cur_v in cur.items():
        if name not in base:
            continue
        tol = tolerances.get(name, 0.0)
        if cur_v < base[name] - tol:
            regressions.append(f"{name} 退化：{base[name]:.3f} → {cur_v:.3f}（容差 {tol:.3f}）")
    return regressions
