"""可插拔写操作后端（A2）：把「重启 / 回滚服务」从 mock 抽成可替换的后端。

生产最终要接到真实 K8s / systemd / 腾讯云 API；本模块提供统一接口 + 按环境变量
切换，默认 mock 只模拟耗时（与改造前行为一致）。真实后端（k8s/systemd/tencent）
目前是占位：选中即显式报错，绝不静默回退 mock，避免「以为重启了其实没动」。

配置（放 .env，由 MCP 独立进程直接读环境变量，不经过 app 配置）：
    WRITE_OP_BACKEND=mock | k8s | systemd | tencent   （默认 mock）

注意：本模块是 mcp_servers/monitor_server.py 的同目录依赖，保持零第三方依赖、
同步接口（与 FastMCP 的同步 tool 一致），方便在独立进程里直接用。
"""
import os
import time
from datetime import datetime


class WriteOpBackendNotConfigured(RuntimeError):
    """所选写操作后端尚未实现 / 未配置，拒绝执行（fail-closed）。"""


class WriteOpBackend:
    """写操作后端基类：restart / rollback 返回结果 dict，失败抛异常。"""

    name = "base"

    def restart(self, service_name: str, reason: str = "") -> dict:
        raise NotImplementedError

    def rollback(self, service_name: str, reason: str = "") -> dict:
        raise NotImplementedError


class MockWriteOpBackend(WriteOpBackend):
    """默认后端：只模拟耗时并返回成功，不真正动生产（演示用）。"""

    name = "mock"

    def restart(self, service_name: str, reason: str = "") -> dict:
        time.sleep(1)  # 模拟重启耗时
        return {
            "service_name": service_name,
            "status": "restarted",
            "reason": reason or "manual",
            "restarted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mock": True,
        }

    def rollback(self, service_name: str, reason: str = "") -> dict:
        time.sleep(1)  # 模拟回滚耗时
        return {
            "service_name": service_name,
            "status": "rolled_back",
            "reason": reason or "auto",
            "rolled_back_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mock": True,
        }


class _UnconfiguredBackend(WriteOpBackend):
    """占位后端：k8s / systemd / tencent 尚未实现，选中即显式报错，绝不静默回退 mock。"""

    def __init__(self, name: str) -> None:
        self.name = name

    def _fail(self, action: str) -> None:
        raise WriteOpBackendNotConfigured(
            f"写操作后端 {self.name!r} 尚未实现（{action} 未执行）。"
            f"请先实现真实后端，并在 WRITE_OP_BACKEND 中选中它。"
        )

    def restart(self, service_name: str, reason: str = "") -> dict:
        self._fail(f"重启 {service_name}")

    def rollback(self, service_name: str, reason: str = "") -> dict:
        self._fail(f"回滚 {service_name}")


def _create_backend() -> WriteOpBackend:
    """按 WRITE_OP_BACKEND 环境变量选择后端。未知值显式报错，不回退 mock。"""
    name = os.getenv("WRITE_OP_BACKEND", "mock").strip().lower()
    if name == "mock":
        return MockWriteOpBackend()
    if name in ("k8s", "systemd", "tencent"):
        return _UnconfiguredBackend(name)
    raise WriteOpBackendNotConfigured(
        f"未知的 WRITE_OP_BACKEND={name!r}，可选 mock / k8s / systemd / tencent"
    )


# 全局单例（与 app 侧 audit_logger / tool_registry 同风格）
write_op_backend = _create_backend()
