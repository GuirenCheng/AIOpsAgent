"""可插拔写操作后端（A2）：把「重启 / 回滚服务」从 mock 抽成可替换的后端。

生产最终要接到真实 K8s / systemd / 腾讯云 API；本模块提供统一接口 + 按环境变量
切换，默认 mock 只模拟耗时（与改造前行为一致）。

已实现：mock（默认）、systemd（真实 `systemctl restart` + 「版本目录 + current 软链」
回滚）。k8s / tencent 仍是占位：选中即显式报错，绝不静默回退 mock，避免「以为
重启了其实没动」。

配置（放 .env，由 MCP 独立进程直接读环境变量，不经过 app 配置）：
    WRITE_OP_BACKEND=mock | systemd | k8s | tencent   （默认 mock）
    SYSTEMD_DEPLOY_ROOT=部署根目录（systemd 后端用，默认 /srv）

注意：本模块是 mcp_servers/monitor_server.py 的同目录依赖，保持零第三方依赖、
同步接口（与 FastMCP 的同步 tool 一致），方便在独立进程里直接用。
"""
import os
import re
import subprocess
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


class SystemdWriteOpBackend(WriteOpBackend):
    """真实 systemd 后端：重启走 `systemctl restart`，回滚走「版本目录 + current 软链切换」。

    适用于「几台服务器跑进程、用 systemd 管理、用 releases/<version> + current 软链」
    的部署布局（Capistrano 风格）：

        {deploy_root}/{service}/
            releases/20261018/   ← 各版本目录
            releases/20261017/
            current -> releases/20261018   ← 服务实际从这个软链运行

    - restart：`systemctl restart <unit>`（service_name 不带 .service 时自动补）。
    - rollback：把 current 软链原子切回上一个版本目录，再 restart 使其生效。
      systemd 本身无回滚语义，因此回滚强依赖上面的「软链 + 版本目录」布局；
      缺软链 / 版本不足时 fail-closed（抛 WriteOpBackendNotConfigured），绝不假装回滚成功。

    配置（由 MCP 独立进程直接读环境变量）：
        SYSTEMD_DEPLOY_ROOT=部署根目录（默认 /srv）
    """

    name = "systemd"
    _SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

    def __init__(self, deploy_root: str | None = None) -> None:
        self.deploy_root = deploy_root or os.getenv("SYSTEMD_DEPLOY_ROOT", "/srv")

    @staticmethod
    def _unit_name(service_name: str) -> str:
        return service_name if service_name.endswith(".service") else f"{service_name}.service"

    @classmethod
    def _validate_service_name(cls, service_name: str) -> None:
        """只允许 [A-Za-z0-9][A-Za-z0-9._-]*：挡住路径穿越与任意 systemd unit 注入。

        护栏白名单（app 侧 execute_guarded）是权威闸门；这里是 MCP 侧第二道独立防线。
        """
        if not service_name or not cls._SAFE_NAME.fullmatch(service_name):
            raise ValueError(f"非法服务名：{service_name!r}")

    def _run_systemctl(self, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
        """执行 systemctl（列表参数、不经过 shell，避免注入）；非零退出即 fail-closed 抛异常。"""
        proc = subprocess.run(
            ["systemctl", *args], capture_output=True, text=True, timeout=timeout
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(
                f"systemctl {' '.join(args)} 失败：{detail or ('exit ' + str(proc.returncode))}"
            )
        return proc

    def restart(self, service_name: str, reason: str = "") -> dict:
        self._validate_service_name(service_name)
        unit = self._unit_name(service_name)
        self._run_systemctl("restart", unit)
        return {
            "service_name": service_name,
            "unit": unit,
            "status": "restarted",
            "reason": reason or "manual",
            "restarted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "backend": "systemd",
            "mock": False,
        }

    def rollback(self, service_name: str, reason: str = "") -> dict:
        self._validate_service_name(service_name)
        service_dir = os.path.join(self.deploy_root, service_name)
        current_link = os.path.join(service_dir, "current")
        releases_dir = os.path.join(service_dir, "releases")

        if not os.path.islink(current_link):
            raise WriteOpBackendNotConfigured(
                f"systemd 回滚依赖「current 软链」部署布局，未找到 {current_link}，"
                f"无法回滚（未执行任何操作）"
            )
        if not os.path.isdir(releases_dir):
            raise WriteOpBackendNotConfigured(
                f"systemd 回滚依赖版本目录 {releases_dir}，目录不存在，无法回滚（未执行任何操作）"
            )

        releases = sorted(
            d for d in os.listdir(releases_dir)
            if os.path.isdir(os.path.join(releases_dir, d))
        )
        if len(releases) < 2:
            raise WriteOpBackendNotConfigured(
                f"发布目录 {releases_dir} 中不足两个版本，无上一版本可回滚"
            )

        current_version = os.path.basename(os.path.normpath(os.readlink(current_link)))
        # 回滚 = 严格回退到「当前版本的前一个」；版本名按日期排序即按时间排序。
        # 注意不能取「最新但非 current」的版本：若已回滚过，最新版本仍在 releases 里，
        # 那样会错误地「向前回滚」。找不到 current 或已是最早版本都 fail-closed。
        try:
            idx = releases.index(current_version)
        except ValueError:
            raise WriteOpBackendNotConfigured(
                f"current 指向的版本 {current_version} 不在 {releases_dir} 中，无法定位上一版本"
            )
        if idx == 0:
            raise WriteOpBackendNotConfigured(
                f"当前已是最早版本 {current_version}，无上一版本可回滚"
            )
        previous = releases[idx - 1]

        previous_target = os.path.join(releases_dir, previous)
        self._switch_current(current_link, previous_target)

        unit = self._unit_name(service_name)
        self._run_systemctl("restart", unit)
        return {
            "service_name": service_name,
            "unit": unit,
            "status": "rolled_back",
            "reason": reason or "auto",
            "from_version": current_version,
            "to_version": previous,
            "rolled_back_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "backend": "systemd",
            "mock": False,
        }

    @staticmethod
    def _switch_current(current_link: str, new_target: str) -> None:
        """把 current 软链原子切到新目标：先建临时软链再 os.replace，避免 current 短暂不存在。"""
        tmp_link = current_link + ".new"
        if os.path.islink(tmp_link) or os.path.exists(tmp_link):
            os.remove(tmp_link)
        os.symlink(new_target, tmp_link)
        os.replace(tmp_link, current_link)


class _UnconfiguredBackend(WriteOpBackend):
    """占位后端：k8s / tencent 尚未实现，选中即显式报错，绝不静默回退 mock。"""

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
    if name == "systemd":
        return SystemdWriteOpBackend()
    if name in ("k8s", "tencent"):
        return _UnconfiguredBackend(name)
    raise WriteOpBackendNotConfigured(
        f"未知的 WRITE_OP_BACKEND={name!r}，可选 mock / systemd / k8s / tencent"
    )


# 全局单例（与 app 侧 audit_logger / tool_registry 同风格）
write_op_backend = _create_backend()
