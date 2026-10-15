"""工具注册表：维护「工具名 → ToolSpec」的单一真相源。

后续里程碑（资源锁、调度、审批）都从这里读取工具元数据，避免在各处散落硬编码。
"""
from app.config import config
from app.harness.tool_spec import ToolSpec, RiskLevel


class ToolRegistry:
    """工具元数据注册表（进程内，全局单例 tool_registry）。"""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """登记（或覆盖）一个工具的契约元数据。"""
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        """按名查询；未登记的工具回退到保守默认 spec。

        保守默认：idempotent=False（不自动重试）、read_only=False（视为写操作），
        保证未知工具不会被盲目重试或被误判为安全只读。
        """
        return self._specs.get(
            name,
            ToolSpec(
                name=name,
                read_only=False,
                idempotent=False,
                risk=RiskLevel.MEDIUM,
                timeout_seconds=config.harness_tool_timeout_seconds,
            ),
        )

    def is_idempotent(self, name: str) -> bool:
        return self.get(name).idempotent

    def is_read_only(self, name: str) -> bool:
        return self.get(name).read_only

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __iter__(self):
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)


def _seed_registry(registry: ToolRegistry) -> ToolRegistry:
    """注册所有已知工具的契约元数据（import 时执行一次）。"""
    # 本地工具（只读、幂等、低风险）
    registry.register(ToolSpec(name="retrieve_knowledge", read_only=True, idempotent=True, risk=RiskLevel.LOW))
    registry.register(ToolSpec(name="get_current_time", read_only=True, idempotent=True, risk=RiskLevel.LOW))

    # CLS 日志检索（只读、幂等）
    for name in (
        "get_current_timestamp",
        "get_region_code_by_name",
        "get_topic_info_by_name",
        "search_topic_by_service_name",
        "search_log",
    ):
        registry.register(ToolSpec(name=name, read_only=True, idempotent=True, risk=RiskLevel.LOW))

    # Monitor 监控查询（只读、幂等）
    for name in ("query_cpu_metrics", "query_memory_metrics", "get_host_metrics"):
        registry.register(ToolSpec(name=name, read_only=True, idempotent=True, risk=RiskLevel.LOW))

    # 写操作样例（mock）：重启服务——非只读、非幂等、高危、有回滚、高优先级（故障恢复优先入槽）
    registry.register(ToolSpec(
        name="restart_service",
        read_only=False,
        idempotent=False,
        target_resource="service_name",
        risk=RiskLevel.HIGH,
        priority=10,
        has_rollback=True,
        rollback_tool="rollback_service",
        timeout_seconds=60.0,
    ))
    # 回滚动作（mock）：risk=MEDIUM（低于默认 HIGH 审批阈值）→ 自动回滚无需人工在环；
    # 幂等可重试、锁同一资源；rollback_tool=None 避免回滚失败再次触发回滚（无限回滚）。
    registry.register(ToolSpec(
        name="rollback_service",
        read_only=False,
        idempotent=True,
        target_resource="service_name",
        risk=RiskLevel.MEDIUM,
        has_rollback=False,
        rollback_tool=None,
        timeout_seconds=60.0,
    ))
    return registry


# 全局单例（与 milvus_manager / vector_store_manager 同风格）
tool_registry = _seed_registry(ToolRegistry())
