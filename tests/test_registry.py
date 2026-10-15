"""ToolRegistry：已知工具查得到、未知工具回退保守默认、种子完整。"""
from app.harness.registry import tool_registry


def test_known_read_only_tools_are_idempotent():
    for name in ("search_log", "query_cpu_metrics", "retrieve_knowledge", "get_current_time"):
        assert tool_registry.is_idempotent(name) is True
        assert tool_registry.is_read_only(name) is True


def test_restart_service_is_write_non_idempotent():
    spec = tool_registry.get("restart_service")
    assert spec.read_only is False
    assert spec.idempotent is False
    assert spec.target_resource == "service_name"
    assert spec.has_rollback is True


def test_unknown_tool_falls_back_to_conservative_default():
    spec = tool_registry.get("some_unknown_tool")
    assert spec.idempotent is False  # 未知工具不重试
    assert spec.read_only is False   # 未知工具视为写操作


def test_seed_contains_all_declared_tools():
    expected = {
        "retrieve_knowledge", "get_current_time",
        "get_current_timestamp", "get_region_code_by_name", "get_topic_info_by_name",
        "search_topic_by_service_name", "search_log",
        "query_cpu_metrics", "query_memory_metrics", "get_host_metrics",
        "restart_service",
    }
    assert expected.issubset(set(tool_registry))
