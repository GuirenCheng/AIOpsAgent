"""Token 用量查看脚本：把 data/token_usage.sqlite 里的记账聚合成易读报表。

每次 DashScope 调用（chat / rerank / embedding）都会由 app.services.token_usage 记一笔
input/output/total tokens。本脚本读同一份库，输出：
- 总量（调用次数 / 输入 / 输出 / 总 tokens）
- 按模型、按来源（chat=大模型 / rerank=重排 / embedding=嵌入）聚合
- 近 7 天每日用量
- 每日预算状态（若启用了 token_budget）

用法：
    python scripts/token_usage.py            # 纯文本报表
    python scripts/token_usage.py --json     # 机器可读 JSON

注意：只读，不写库；用量库不存在或为空时输出零值。
"""
import argparse
import json
import sys
from pathlib import Path

# 让 `python scripts/token_usage.py` 也能从项目根导入 app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import config  # noqa: E402
from app.services.token_usage import token_usage_recorder  # noqa: E402


def _collect() -> dict:
    stats = token_usage_recorder.stats()
    budget = token_usage_recorder.check_daily_budget()
    return {"stats": stats, "budget": budget, "db_path": config.token_usage_db_path}


def _render_text(data: dict) -> str:
    stats = data["stats"]
    budget = data["budget"]
    total = stats["total"]

    lines = [f"Token 用量库：{data['db_path']}", ""]
    lines.append(
        f"总量：{total['records']} 次调用 | "
        f"输入 {total['input_tokens']:,} | 输出 {total['output_tokens']:,} | "
        f"合计 {total['total_tokens']:,} tokens"
    )

    if budget["enabled"]:
        mark = "⚠️ 已超预算" if budget["exceeded"] else "✅ 未超预算"
        lines.append(
            f"今日预算：{budget['today_tokens']:,} / {budget['limit']:,} tokens（{mark}）"
        )

    if stats["by_model"]:
        lines.append("")
        lines.append("按模型：")
        for r in stats["by_model"]:
            lines.append(
                f"  {r['model']:<24} {r['calls']:>5} 次 | "
                f"输入 {r['input_tokens']:>10,} | 输出 {r['output_tokens']:>10,} | "
                f"合计 {r['total_tokens']:>12,}"
            )

    if stats["by_source"]:
        lines.append("")
        lines.append("按来源：")
        for r in stats["by_source"]:
            lines.append(f"  {r['source']:<12} {r['calls']:>5} 次 | 合计 {r['total_tokens']:>12,} tokens")

    if stats["by_day"]:
        lines.append("")
        lines.append("近 7 天：")
        for r in stats["by_day"]:
            lines.append(f"  {r['day']}  {r['calls']:>5} 次 | {r['total_tokens']:>12,} tokens")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="查看 Token 用量报表")
    parser.add_argument("--json", action="store_true", help="输出 JSON 而非文本")
    args = parser.parse_args()

    data = _collect()
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(_render_text(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
