"""ToolSpec 元数据模型的单元测试。"""
from app.harness.tool_spec import ToolSpec, RiskLevel


def test_defaults_are_read_only_idempotent_low_risk():
    spec = ToolSpec(name="search_log")
    assert spec.read_only is True
    assert spec.idempotent is True
    assert spec.risk == RiskLevel.LOW
    assert spec.has_rollback is False
    assert spec.timeout_seconds == 30.0
    assert spec.target_resource is None


def test_write_tool_spec_marks_non_idempotent_high_risk():
    spec = ToolSpec(
        name="restart_service",
        read_only=False,
        idempotent=False,
        target_resource="service_name",
        risk=RiskLevel.HIGH,
        has_rollback=True,
        timeout_seconds=60.0,
    )
    assert spec.read_only is False
    assert spec.idempotent is False
    assert spec.risk == RiskLevel.HIGH
    assert spec.has_rollback is True
    assert spec.timeout_seconds == 60.0
