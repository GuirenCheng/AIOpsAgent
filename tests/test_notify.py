"""通知通道（B2）单元测试"""
from app.config import config
from app.harness import notify as notify_mod
from app.harness.notify import (
    DingTalkNotifier,
    LogNotifier,
    _create_notifier,
    _truncate_bytes,
    notify_alert_diagnosis,
    notify_approval,
    notify_approval_timeout,
    notify_rollback_failure,
)


def test_truncate_bytes_short_unchanged():
    assert _truncate_bytes("短文本", 100) == "短文本"


def test_truncate_bytes_truncates_by_utf8_bytes():
    # 三个中文字符 = 9 字节，limit 6 字节只能装下 2 个
    assert _truncate_bytes("一二三", 6) == "一二…"


async def test_log_notifier_returns_true():
    assert await LogNotifier().send("t", "c") is True


def test_dingtalk_signed_url_without_secret():
    url = "https://oapi.dingtalk.com/robot/send?access_token=x"
    assert DingTalkNotifier(url)._signed_url() == url


def test_dingtalk_signed_url_with_secret():
    url = "https://oapi.dingtalk.com/robot/send?access_token=x"
    signed = DingTalkNotifier(url, "SECRET")._signed_url()
    assert "timestamp=" in signed
    assert "sign=" in signed
    assert "access_token=x" in signed


def test_create_notifier_default_is_log():
    assert isinstance(_create_notifier(), LogNotifier)


def test_create_notifier_dingtalk_with_url(monkeypatch):
    monkeypatch.setattr(config, "notify_channel", "dingtalk")
    monkeypatch.setattr(config, "dingtalk_webhook_url", "https://x")
    assert isinstance(_create_notifier(), DingTalkNotifier)


def test_create_notifier_dingtalk_without_url_falls_back(monkeypatch):
    monkeypatch.setattr(config, "notify_channel", "dingtalk")
    monkeypatch.setattr(config, "dingtalk_webhook_url", "")
    assert isinstance(_create_notifier(), LogNotifier)


async def test_notify_approval_formats_and_sends(monkeypatch):
    captured = {}

    async def fake_notify(title, content):
        captured["title"] = title
        captured["content"] = content
        return True

    monkeypatch.setattr(notify_mod, "notify", fake_notify)
    monkeypatch.setattr(config, "notify_base_url", "")
    await notify_approval("appr-1", "restart_service", {"service_name": "web"}, "prod-01")

    assert "待审批" in captured["title"]
    assert "appr-1" in captured["content"]
    assert "restart_service" in captured["content"]
    assert "prod-01" in captured["content"]


async def test_notify_alert_diagnosis_sends(monkeypatch):
    captured = {}

    async def fake_notify(title, content):
        captured["title"] = title
        captured["content"] = content
        return True

    monkeypatch.setattr(notify_mod, "notify", fake_notify)
    await notify_alert_diagnosis("HighCPU", "critical", "# 报告")

    assert "HighCPU" in captured["title"]
    assert "HighCPU" in captured["content"]
    assert "critical" in captured["content"]
    assert "# 报告" in captured["content"]


async def test_notify_approval_timeout_formats(monkeypatch):
    captured = {}

    async def fake_notify(title, content):
        captured["title"] = title
        captured["content"] = content
        return True

    monkeypatch.setattr(notify_mod, "notify", fake_notify)
    await notify_approval_timeout("appr-9", "restart_service", {"service_name": "web"}, "prod-01")

    assert "超时" in captured["title"]
    assert "appr-9" in captured["content"]
    assert "restart_service" in captured["content"]
    assert "prod-01" in captured["content"]


async def test_notify_rollback_failure_formats(monkeypatch):
    captured = {}

    async def fake_notify(title, content):
        captured["title"] = title
        captured["content"] = content
        return True

    monkeypatch.setattr(notify_mod, "notify", fake_notify)
    await notify_rollback_failure("restart_service", "rollback_service", "回滚失败原因", "prod-01")

    assert "回滚失败" in captured["title"]
    assert "restart_service" in captured["content"]
    assert "rollback_service" in captured["content"]
    assert "回滚失败原因" in captured["content"]
