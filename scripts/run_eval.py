"""统一评测入口：检索 + 诊断，汇总报告，支持基线回归对比。

把原先两个独立脚本（eval_retrieval / eval_diagnosis）收口到一个命令，产出
同一份 JSON 报告；配合基线文件实现「每次改动跑评测、退化即报错」。

用法：
    python scripts/run_eval.py                     # 全量评测并打印报告
    python scripts/run_eval.py --stage retrieval   # 只跑检索（需 Milvus + embedding）
    python scripts/run_eval.py --stage diagnosis   # 只跑诊断（真实 LLM，有 API 成本）
    python scripts/run_eval.py --save-baseline     # 跑完写基线 scripts/eval/baseline.json
    python scripts/run_eval.py --check             # 跑完与基线对比，退化则非零退出
    python scripts/run_eval.py --json report.json  # 报告落盘

注意：检索依赖本地 Milvus（localhost:19530）+ .env 的 DASHSCOPE_API_KEY；
诊断依赖真实 LLM（有 API 成本、结果非确定）。非 pytest，不进 CI。
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# 评测不落会话：任何 app 导入前禁用 SQLite checkpointer，用内存 MemorySaver
os.environ["SESSION_STORE_ENABLED"] = "false"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 使 import eval_retrieval / eval_diagnosis 可用

from app.eval_metrics import build_report, check_regression  # noqa: E402

DEFAULT_BASELINE = Path(__file__).resolve().parent / "eval" / "baseline.json"
DEFAULT_GOLDEN = Path(__file__).resolve().parent / "eval" / "retrieval_golden.json"
DEFAULT_SCENARIOS = Path(__file__).resolve().parent / "eval" / "scenarios"


def _run_retrieval(golden_path: str, max_k: int) -> dict:
    import eval_retrieval

    return eval_retrieval.run_retrieval(golden_path, max_k=max_k)


def _run_diagnosis(scenarios_path: str, limit: int) -> dict:
    import eval_diagnosis

    return asyncio.run(eval_diagnosis.run_diagnosis(scenarios_path, limit=limit))


def _print_report(report: dict) -> None:
    r = report.get("retrieval") or {}
    d = report.get("diagnosis") or {}
    print("\n" + "=" * 60)
    print("评测汇总")
    print("=" * 60)
    if r:
        print(
            f"检索：N={r.get('n')}  top_k={r.get('top_k')}  threshold={r.get('threshold')}  "
            f"hit@3={r.get('hit_at_3', 0):.1%}  precision@3={r.get('precision_at_3', 0):.1%}"
        )
    if d:
        print(f"诊断：N={d.get('n')}  passed={d.get('passed')}  accuracy={d.get('accuracy', 0):.1%}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="统一评测入口（检索 + 诊断 + 基线回归）")
    parser.add_argument("--stage", choices=["all", "retrieval", "diagnosis"], default="all")
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    parser.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS))
    parser.add_argument("--max-k", type=int, default=10, help="检索时向 Milvus 请求的最大 k")
    parser.add_argument("--limit", type=int, default=0, help="诊断只跑前 N 个场景（0=全部）")
    parser.add_argument("--json", default="", help="报告落盘路径")
    parser.add_argument("--save-baseline", action="store_true", help="跑完写基线")
    parser.add_argument("--check", action="store_true", help="与基线对比，退化则非零退出")
    args = parser.parse_args()

    report = build_report({}, {})
    if args.stage in ("all", "retrieval"):
        report["retrieval"] = _run_retrieval(args.golden, args.max_k)
    if args.stage in ("all", "diagnosis"):
        report["diagnosis"] = _run_diagnosis(args.scenarios, args.limit)

    _print_report(report)

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"报告已写入：{args.json}")

    if args.check:
        if not DEFAULT_BASELINE.exists():
            print(f"未找到基线 {DEFAULT_BASELINE}，请先 `--save-baseline` 生成基线")
            sys.exit(2)
        baseline = json.loads(DEFAULT_BASELINE.read_text(encoding="utf-8"))
        regressions = check_regression(report, baseline)
        if regressions:
            print("\n⚠️ 评测退化：")
            for msg in regressions:
                print(f"  - {msg}")
            sys.exit(1)
        print("\n✅ 无退化，通过基线检查")

    if args.save_baseline:
        DEFAULT_BASELINE.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"基线已保存：{DEFAULT_BASELINE}")


if __name__ == "__main__":
    main()
