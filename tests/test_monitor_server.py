"""监控 MCP server 回归测试。

回归点：query_memory_metrics 此前把百分比先换算成 GB、又把 GB 值当百分比二次换算，
导致 used_gb 计算错误。本测试验证 value 始终是 0-100 的百分比，且 used_gb 与 value 单位一致。
"""
import importlib.util
import pathlib
import sys


def _load_monitor_server():
    path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "mcp_servers"
        / "monitor_server.py"
    )
    spec = importlib.util.spec_from_file_location("monitor_server", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["monitor_server"] = module
    spec.loader.exec_module(module)
    return module


def test_query_memory_metrics_value_and_used_gb_consistency():
    monitor = _load_monitor_server()
    # FastMCP 的 @mcp.tool() 会把函数包装为 FunctionTool，底层可调用函数在其 .fn 上
    result = monitor.query_memory_metrics.fn(
        service_name="data-sync-service",
        start_time="2026-02-14 10:00:00",
        end_time="2026-02-14 10:05:00",
        interval="1m",
    )

    data_points = result["data_points"]
    assert data_points, "应生成内存数据点"

    total_gb = data_points[0]["total_gb"]
    for point in data_points:
        value = point["value"]
        assert 0.0 <= value <= 100.0, f"内存使用率越界：{value}"
        # used_gb 必须与百分比 value 同单位换算，而非二次换算
        assert point["used_gb"] == round((value / 100.0) * total_gb, 2)

    # 统计信息完整
    assert "statistics" in result
    assert {"avg", "max", "min"}.issubset(result["statistics"])
