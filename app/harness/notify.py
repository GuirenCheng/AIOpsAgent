"""审批 / 诊断结论通知通道（B2）。

把「需要人工审批」和「告警诊断完成」两件事推送到外部 IM / 邮件，让运维不必
主动轮询审批列表。默认走 log（零依赖、本地可用），配置凭据后切到
钉钉群机器人 / 企业微信群机器人 / SMTP 邮件。

通知失败绝不打断主流程：send 内部 try/except，失败仅告警、返回 False。
"""
import asyncio
import base64
import hashlib
import hmac
import json
import smtplib
import time
import urllib.parse
from email.header import Header
from email.mime.text import MIMEText

from loguru import logger

from app.config import config


def _truncate_bytes(s: str, limit: int) -> str:
    """按 UTF-8 字节数截断（IM 通道多有字节上限，如企微 markdown ≤4096 字节）。"""
    if len(s.encode("utf-8")) <= limit:
        return s
    out = ""
    for ch in s:
        if len((out + ch).encode("utf-8")) > limit:
            break
        out += ch
    return out + "…"


class Notifier:
    """通知器抽象基类：send 返回是否成功，失败只记日志、不抛异常。"""

    name = "base"

    async def send(self, title: str, content: str) -> bool:
        raise NotImplementedError


class LogNotifier(Notifier):
    """默认通道：把通知写到日志（本地开发零依赖）。"""

    name = "log"

    async def send(self, title: str, content: str) -> bool:
        logger.info(f"[通知:{self.name}] {title}\n{content}")
        return True


class DingTalkNotifier(Notifier):
    """钉钉群机器人（支持加签）。msgtype=markdown，换行需 \\n\\n。"""

    name = "dingtalk"

    def __init__(self, webhook_url: str, secret: str = "") -> None:
        self.webhook_url = webhook_url
        self.secret = secret

    def _signed_url(self) -> str:
        """按钉钉官方加签算法在 URL 上追加 timestamp + sign。"""
        if not self.secret:
            return self.webhook_url
        timestamp = str(round(time.time() * 1000))
        string_to_sign = f"{timestamp}\n{self.secret}"
        digest = hmac.new(
            self.secret.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(digest))
        sep = "&" if "?" in self.webhook_url else "?"
        return f"{self.webhook_url}{sep}timestamp={timestamp}&sign={sign}"

    async def send(self, title: str, content: str) -> bool:
        try:
            import httpx
        except ImportError:
            logger.warning("未安装 httpx，无法发送钉钉通知")
            return False
        payload = {
            "msgtype": "markdown",
            "markdown": {"title": title, "text": content.replace("\n", "\n\n")},
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(self._signed_url(), json=payload)
                data = resp.json()
                ok = data.get("errcode") == 0
                if not ok:
                    logger.warning(f"钉钉通知失败：{data}")
                return ok
        except Exception as e:  # noqa: BLE001
            logger.warning(f"钉钉通知发送异常（忽略）：{e}")
            return False


class WeComNotifier(Notifier):
    """企业微信群机器人。msgtype=markdown（≤4096 字节，截断兜底）。"""

    name = "wecom"

    def __init__(self, webhook_url: str) -> None:
        self.webhook_url = webhook_url

    async def send(self, title: str, content: str) -> bool:
        try:
            import httpx
        except ImportError:
            logger.warning("未安装 httpx，无法发送企业微信通知")
            return False
        text = _truncate_bytes(f"**{title}**\n{content}", 3800)
        payload = {"msgtype": "markdown", "markdown": {"content": text}}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(self.webhook_url, json=payload)
                data = resp.json()
                ok = data.get("errcode") == 0
                if not ok:
                    logger.warning(f"企业微信通知失败：{data}")
                return ok
        except Exception as e:  # noqa: BLE001
            logger.warning(f"企业微信通知发送异常（忽略）：{e}")
            return False


class EmailNotifier(Notifier):
    """SMTP 邮件通知（同步 smtplib 用 to_thread，避免阻塞事件循环）。"""

    name = "email"

    def __init__(self, host, port, use_ssl, username, password, from_addr, to_addr):
        self.host = host
        self.port = port
        self.use_ssl = use_ssl
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addr = to_addr

    def _send_sync(self, title: str, content: str) -> None:
        msg = MIMEText(content, "plain", "utf-8")
        msg["Subject"] = Header(title, "utf-8")
        msg["From"] = self.from_addr
        msg["To"] = self.to_addr
        if self.use_ssl:
            server = smtplib.SMTP_SSL(self.host, self.port, timeout=10)
        else:
            server = smtplib.SMTP(self.host, self.port, timeout=10)
            server.starttls()
        try:
            if self.username:
                server.login(self.username, self.password)
            server.sendmail(self.from_addr, [self.to_addr], msg.as_string())
        finally:
            server.quit()

    async def send(self, title: str, content: str) -> bool:
        try:
            await asyncio.to_thread(self._send_sync, title, content)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"邮件通知发送异常（忽略）：{e}")
            return False


def _create_notifier() -> Notifier:
    channel = config.notify_channel.strip().lower()
    if channel == "dingtalk":
        if config.dingtalk_webhook_url:
            return DingTalkNotifier(config.dingtalk_webhook_url, config.dingtalk_secret)
        logger.warning("notify_channel=dingtalk 但未配置 dingtalk_webhook_url，回退 log")
    elif channel == "wecom":
        if config.wecom_webhook_url:
            return WeComNotifier(config.wecom_webhook_url)
        logger.warning("notify_channel=wecom 但未配置 wecom_webhook_url，回退 log")
    elif channel == "email":
        if config.smtp_host:
            return EmailNotifier(
                config.smtp_host,
                config.smtp_port,
                config.smtp_use_ssl,
                config.smtp_username,
                config.smtp_password,
                config.smtp_from,
                config.smtp_to,
            )
        logger.warning("notify_channel=email 但未配置 smtp_host，回退 log")
    elif channel != "log":
        logger.warning(f"未知 notify_channel={config.notify_channel}，回退 log")
    return LogNotifier()


# 全局单例（与 audit_logger / approval_manager 同风格）
notifier = _create_notifier()


async def notify(title: str, content: str) -> bool:
    """发送一条通知到已配置通道；失败返回 False（绝不抛出）。"""
    try:
        return await notifier.send(title, content)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"通知发送异常（忽略）：{e}")
        return False


async def notify_approval(approval_id: str, tool: str, args, server: str) -> None:
    """高危操作创建审批单后，推送「待审批」通知。"""
    args_str = json.dumps(args, ensure_ascii=False, default=str) if args else "-"
    lines = [
        f"审批单号：{approval_id}",
        f"工具：{tool}",
        f"服务器：{server or '-'}",
        f"参数：{args_str}",
    ]
    base = config.notify_base_url.rstrip("/")
    if base:
        lines.append(f"审批列表：{base}/api/harness/approvals")
    lines.append("请尽快审批：批准才放行，拒绝/超时则不执行（fail-closed）。")
    await notify("⚠️ 高危操作待审批", "\n".join(lines))


async def notify_alert_diagnosis(alertname: str, severity: str, report: str) -> None:
    """告警 webhook 诊断完成后，推送诊断结论（报告过长按字节截断）。"""
    body = _truncate_bytes(report or "（无诊断报告）", 3800)
    await notify(
        f"✅ 告警诊断完成：{alertname}",
        f"告警：{alertname}\n级别：{severity}\n\n{body}",
    )


async def notify_approval_timeout(approval_id: str, tool: str, args, server: str) -> None:
    """审批单超时（fail-closed 拒绝执行）后推送告警。"""
    args_str = json.dumps(args, ensure_ascii=False, default=str) if args else "-"
    await notify(
        "⏰ 高危操作审批超时",
        f"审批单号：{approval_id}\n工具：{tool}\n服务器：{server or '-'}\n参数：{args_str}\n"
        "超时未审批，已按 fail-closed 拒绝执行。",
    )


async def notify_rollback_failure(failed_tool: str, rollback_tool: str, error: str, server: str) -> None:
    """自动回滚失败后推送告警（回滚本身出错，需人工介入确认状态）。"""
    await notify(
        "🚨 自动回滚失败",
        f"失败工具：{failed_tool}\n回滚工具：{rollback_tool}\n服务器：{server or '-'}\n"
        f"原因：{error}\n请人工介入确认状态。",
    )
