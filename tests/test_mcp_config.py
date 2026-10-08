"""MCP 配置接线回归测试。

回归点：config.mcp_servers 此前返回空字典 {}，导致 MultiServerMCPClient 连接零个服务器、
Agent 的 MCP 工具数恒为 0。本测试确保返回的配置包含两个服务器且格式正确。
"""
from app.config import config


def test_mcp_servers_config_is_populated():
    servers = config.mcp_servers
    assert set(servers.keys()) == {"cls", "monitor"}, "应包含 cls 与 monitor 两个 MCP server"

    for name in ("cls", "monitor"):
        assert servers[name]["transport"] in ("streamable-http", "streamable_http")
        assert servers[name]["url"].startswith("http://")
        assert servers[name]["url"].endswith("/mcp")

    # 两个服务器应指向不同的端口
    assert servers["cls"]["url"] != servers["monitor"]["url"]
