"""
Plan-Execute-Replan 诊断准确率评估脚本（真实 LLM）。

与 tests/unit 的确定性单测不同，本脚本：
- 真实调用 ChatQwen（读取 .env 的 DASHSCOPE_API_KEY），测量"真实诊断质量"；
- 但把 MCP 工具替换为脚本化的 canned 数据，保证场景可复现；
- 按根因关键词打分，统计准确率。

非 pytest，不会被 testpaths 收集，默认不进 CI。有 API 成本且结果非确定。

用法：
    python scripts/eval_diagnosis.py --scenarios scripts/eval/scenarios
    python scripts/eval_diagnosis.py --scenarios scripts/eval/scenarios --limit 3
"""
import argparse
import asyncio
import json
import os
import sys
import uuid
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

# 让 `python scripts/eval_diagnosis.py` 也能从项目根导入 app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 必须在导入 app 之前设置：禁用 SQLite checkpointer（用内存 MemorySaver），避免写文件
os.environ["SESSION_STORE_ENABLED"] = "false"

from langchain_core.tools import tool  # noqa: E402


# ---------------------------------------------------------------------------
# 场景状态（每个场景通过 _CURRENT 提供 canned 数据）
# ---------------------------------------------------------------------------

_CURRENT = None          # 当前场景 dict
_CALL_COUNTS = {}        # 每工具调用次数（用于 returns 的循环返回）


def _lookup(tool_name):
    """按工具名取当前场景的 canned 返回值。"""
    spec = (_CURRENT or {}).get("tools", {}).get(tool_name, {})
    if "returns" in spec:
        i = _CALL_COUNTS.get(tool_name, 0)
        _CALL_COUNTS[tool_name] = i + 1
        seq = spec["returns"]
        return seq[i % len(seq)]
    return spec.get("return")


# ---------------------------------------------------------------------------
# 本地工具覆盖（与 app 中的 get_current_time / retrieve_knowledge 同名同签名）
# ---------------------------------------------------------------------------

@tool
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取当前时间。"""
    return (_CURRENT or {}).get("time_now", "2026-10-12 10:00:00")


@tool
def retrieve_knowledge(query: str) -> str:
    """从知识库中检索相关信息来回答问题。"""
    return (_CURRENT or {}).get("knowledge", "") or "未找到可靠的相关资料。"


# ---------------------------------------------------------------------------
# MCP 工具 fixture（与真实 MCP 工具同名，返回脚本化数据）
# ---------------------------------------------------------------------------

@tool
def get_current_timestamp() -> int:
    """获取当前 Unix 时间戳（毫秒）。"""
    return _lookup("get_current_timestamp")


@tool
def get_region_code_by_name(region_name: str) -> dict:
    """根据地域名获取 region code。"""
    return _lookup("get_region_code_by_name")


@tool
def get_topic_info_by_name(topic_name: str, region_code: str = None) -> dict:
    """根据日志主题名获取主题信息。"""
    return _lookup("get_topic_info_by_name")


@tool
def search_topic_by_service_name(service_name: str, region_code: str = None, fuzzy: bool = True) -> dict:
    """按服务名搜索 CLS 日志主题，返回 topic_id 等。"""
    return _lookup("search_topic_by_service_name")


@tool
def search_log(topic_id, start_time, end_time, query: str = None, limit: int = 100) -> dict:
    """在指定日志主题内按时间范围检索日志。"""
    return _lookup("search_log")


@tool
def query_cpu_metrics(service_name: str, start_time: str = None, end_time: str = None, interval: str = "1m") -> dict:
    """查询服务的 CPU 使用率指标。"""
    return _lookup("query_cpu_metrics")


@tool
def query_memory_metrics(service_name: str, start_time: str = None, end_time: str = None, interval: str = "1m") -> dict:
    """查询服务的内存使用率指标。"""
    return _lookup("query_memory_metrics")


@tool
def get_host_metrics() -> dict:
    """获取主机整体 CPU / 内存 / 磁盘使用率。"""
    return _lookup("get_host_metrics")


# 暴露给 get_tools() 的 MCP 工具集合（本地工具走 DIRECT_TOOLS 覆盖）
FIXTURE_TOOLS = [
    get_current_timestamp,
    get_region_code_by_name,
    get_topic_info_by_name,
    search_topic_by_service_name,
    search_log,
    query_cpu_metrics,
    query_memory_metrics,
    get_host_metrics,
]


class _StubClient:
    def __init__(self, tools):
        self._tools = tools

    async def get_tools(self):
        return self._tools


async def _fake_get_mcp_client(servers=None, tool_interceptors=None, force_new=False):
    return _StubClient(FIXTURE_TOOLS)


def _patchers():
    """返回针对当前场景的全部 monkeypatch 上下文管理器。"""
    return [
        patch("app.agent.aiops.planner.get_mcp_client_with_retry", new=_fake_get_mcp_client),
        patch("app.agent.aiops.executor.get_mcp_client_with_retry", new=_fake_get_mcp_client),
        patch("app.agent.aiops.replanner.get_mcp_client_with_retry", new=_fake_get_mcp_client),
        patch("app.agent.aiops.planner.retrieve_knowledge", new=retrieve_knowledge),
        patch("app.agent.aiops.planner.get_current_time", new=get_current_time),
        patch("app.agent.aiops.replanner.retrieve_knowledge", new=retrieve_knowledge),
        patch("app.agent.aiops.replanner.get_current_time", new=get_current_time),
        patch("app.agent.aiops.executor.DIRECT_TOOLS", new={
            "get_current_time": get_current_time,
            "retrieve_knowledge": retrieve_knowledge,
        }),
    ]


# ---------------------------------------------------------------------------
# 场景加载 / 打分 / 运行
# ---------------------------------------------------------------------------

def _load_scenarios(path):
    p = Path(path)
    if p.is_file():
        return [json.loads(p.read_text(encoding="utf-8"))]
    files = sorted(p.glob("*.json"))
    scenarios = []
    for f in files:
        scenarios.append(json.loads(f.read_text(encoding="utf-8")))
    return scenarios


def _score(response, expected):
    keywords = expected.get("root_cause_keywords", [])
    min_hits = expected.get("min_keyword_hits", 1)
    low = (response or "").lower()
    matched = [k for k in keywords if k.lower() in low]
    return len(matched) >= min_hits, matched


async def _run_scenario(scenario):
    """运行单个场景，返回最终诊断报告文本。"""
    global _CURRENT, _CALL_COUNTS
    _CURRENT = scenario
    _CALL_COUNTS = {}

    # 延迟导入：确保上面的 patch 目标模块已就绪，且单例已用 MemorySaver 构建
    from app.services.aiops_service import aiops_service

    stack = ExitStack()
    for p in _patchers():
        stack.enter_context(p)

    try:
        response = ""
        async for event in aiops_service.execute(
            scenario.get("alert_input", ""), session_id=uuid.uuid4().hex
        ):
            if event.get("type") == "complete":
                response = event.get("response", "")
        return response
    finally:
        stack.close()


async def main():
    parser = argparse.ArgumentParser(description="Plan-Execute-Replan 诊断准确率评估")
    parser.add_argument("--scenarios", default="scripts/eval/scenarios",
                        help="场景目录或单个 JSON 文件")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个场景（0=全部）")
    parser.add_argument("--min-accuracy", type=float, default=0.0,
                        help="准确率低于该阈值时以非零码退出")
    args = parser.parse_args()

    scenarios = _load_scenarios(args.scenarios)
    if not scenarios:
        print(f"未找到场景文件：{args.scenarios}")
        sys.exit(2)
    if args.limit:
        scenarios = scenarios[: args.limit]

    print(f"共 {len(scenarios)} 个场景，开始评估（真实 LLM）...\n")
    passed = 0
    for sc in scenarios:
        name = sc.get("name", "unnamed")
        response = await _run_scenario(sc)
        ok, matched = _score(response, sc.get("expected", {}))
        passed += int(ok)
        preview = (response or "").replace("\n", " ")[:120]
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"      命中关键词: {matched or '无'}  (预期至少 {sc.get('expected', {}).get('min_keyword_hits', 1)} 个)")
        print(f"      响应预览: {preview}\n")

    total = len(scenarios)
    acc = passed / total if total else 0.0
    print(f"准确率: {passed}/{total} = {acc:.1%}")

    if acc < args.min_accuracy:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
