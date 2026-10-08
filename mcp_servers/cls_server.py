"""腾讯云 CLS (Cloud Log Service) MCP Server

本地实现的 CLS 日志服务 MCP Server，提供日志查询、检索和分析功能。
"""
import functools
import json
import logging
import os
from datetime import datetime,timedelta
from typing import Optional, Dict, Any

from fastmcp import FastMCP

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("CLS_MCP_Server")

mcp = FastMCP("CLS")

# 腾讯云 CLS 凭证（从进程环境读取；MCP 为独立进程，与主应用 .env 解耦）
_CLS_SECRET_ID = os.getenv("TENCENTCLOUD_SECRET_ID", "")
_CLS_SECRET_KEY = os.getenv("TENCENTCLOUD_SECRET_KEY", "")
_CLS_REGION = os.getenv("TENCENTCLOUD_REGION", "ap-guangzhou")
_HAS_CLS_CREDENTIALS = bool(_CLS_SECRET_ID and _CLS_SECRET_KEY)

if not _HAS_CLS_CREDENTIALS:
    logger.warning(
        "未检测到腾讯云 CLS 凭证（TENCENTCLOUD_SECRET_ID / TENCENTCLOUD_SECRET_KEY），"
        "日志查询工具将以 Mock 模式运行（返回模拟数据，不访问真实 CLS）。"
    )

_CLS_CLIENT = None


def _get_cls_client():
    """懒加载腾讯云 CLS 客户端；无凭证或未安装 SDK 时返回 None。"""
    global _CLS_CLIENT
    if not _HAS_CLS_CREDENTIALS:
        return None
    if _CLS_CLIENT is not None:
        return _CLS_CLIENT
    try:
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile
        from tencentcloud.cls.v20201016 import cls_client

        cred = credential.Credential(_CLS_SECRET_ID, _CLS_SECRET_KEY)
        http_profile = HttpProfile()
        http_profile.endpoint = "cls.tencentcloudapi.com"
        client_profile = ClientProfile()
        client_profile.httpProfile = http_profile
        _CLS_CLIENT = cls_client.ClsClient(cred, _CLS_REGION, client_profile)
        logger.info(f"已初始化腾讯云 CLS 客户端（region={_CLS_REGION}）")
    except ImportError:
        logger.warning("未安装 tencentcloud-sdk-python-cls，CLS 工具回退到本地/Mock 模式")
        return None
    except Exception as e:
        logger.error(f"初始化腾讯云 CLS 客户端失败：{e}")
        return None
    return _CLS_CLIENT


def _describe_topics(topic_name=None, precise=True, limit=20):
    """调用真实 CLS DescribeTopics，返回主题元数据列表；无凭证时抛 RuntimeError。"""
    client = _get_cls_client()
    if client is None:
        raise RuntimeError("未配置腾讯云 CLS 凭证")
    from tencentcloud.cls.v20201016 import models

    req = models.DescribeTopicsRequest()
    req.Limit = limit
    req.PreciseSearch = precise
    if topic_name:
        f = models.Filter()
        f.Key = "topicName"
        f.Values = [topic_name]
        req.Filters = [f]
    resp = client.DescribeTopics(req)
    topics = []
    for t in (resp.Topics or []):
        topics.append({
            "topic_id": t.TopicId,
            "topic_name": t.TopicName,
            # CLS 无 service_name 概念，用主题名近似，便于 Agent 按服务名检索
            "service_name": t.TopicName,
            "region_code": _CLS_REGION,
            "create_time": t.CreateTime or "",
            "log_count": None,
            "description": t.Describes or "",
        })
    return topics

# 本地真实数据（短期路线）：从 JSONL 加载结构化日志与主题元数据。
# 文件不存在时 _LOCAL_LOGS/_LOCAL_TOPICS 为 None，工具自动回退到 Mock 模式。
_CLS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cls")
_CLS_LOGS_FILE = os.path.join(_CLS_DATA_DIR, "logs.jsonl")
_CLS_TOPICS_FILE = os.path.join(_CLS_DATA_DIR, "topics.json")


def _load_local_logs():
    """加载本地结构化日志（logs.jsonl），无文件返回 None。"""
    if not os.path.exists(_CLS_LOGS_FILE):
        return None
    logs = []
    with open(_CLS_LOGS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return logs


def _load_local_topics():
    """加载本地主题元数据（topics.json），无文件返回 None。"""
    if not os.path.exists(_CLS_TOPICS_FILE):
        return None
    try:
        with open(_CLS_TOPICS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


_LOCAL_LOGS = _load_local_logs()
_LOCAL_TOPICS = _load_local_topics()
_HAS_LOCAL_DATA = bool(_LOCAL_LOGS)

if _HAS_LOCAL_DATA:
    logger.info(
        f"CLS 本地数据就绪：{len(_LOCAL_LOGS)} 条日志，"
        f"主题数 {len(_LOCAL_TOPICS or [])}"
    )


def log_tool_call(func):
    """装饰器：记录工具调用的日志，包括方法名、参数和返回状态"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        method_name = func.__name__

        # 记录调用信息
        logger.info(f"=" * 80)
        logger.info(f"调用方法:{method_name}")

        # 记录参数（排除self等）
        if kwargs:
            # 使用 json.dumps 格式化参数，处理可能的序列化错误
            try:
                params_str = json.dumps(kwargs,ensure_ascii=False,indent=2)
            except (TypeError, ValueError):
                params_str = str(kwargs)
        else:
            logger.info("参数信息:无")

        # 执行方法
        try:
            result = func(*args, **kwargs)

            # 记录返回状态
            logger.info(f"返回状态: SUCCESS")

            # 记录返回结果摘要（避免日志过长）
            if isinstance(result,dict):
                summary = {k: v if not isinstance(v,(list,dict)) else f"<{type(v).__name__} with {len(v)} items>"
                           for k, v in list(result.items())[:5]}
                logger.info(f"返回寄过摘要：{json.dumps(summary,ensure_ascii=False)}")
            else:
                logger.info(f"返回结果: {result}")

            logger.info(f"=" * 80)
            return result


        except Exception as e:
            # 记录错误状态
            logger.error(f"返回状态：ERROR")
            logger.error(f"错误信息：{str(e)}")
            logger.error(f"=" * 80)
            raise

    return wrapper


def parse_time_or_default(time_str:Optional[str],default_offset_hours:int = 0) -> datetime:
    """解析时间字符串或返回默认时间。

    Args:
        time_str: 时间字符串（格式：YYYY-MM-DD HH:MM:SS）
        default_offset_hours: 默认时间偏移（小时）

    Returns:
        datetime: 解析后的时间对象
    """
    if time_str:
        try:
            return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return datetime.now() + timedelta(hours=default_offset_hours)


def generate_time_series(base_time:datetime,minutes_offset: int) -> str:
    """生成基于基准时间的时间字符串。

    Args:
        base_time: 基准时间
        minutes_offset: 分钟偏移量

    Returns:
        str: 格式化的时间字符串
    """
    result_time = base_time + timedelta(minutes=minutes_offset)
    return result_time.strftime("%Y-%m-%d %H:%M:%S")


@mcp.tool()
@log_tool_call
def get_current_timestamp() -> int:
    """获取当前时间戳（以毫秒为单位）。

    此工具用于获取标准的毫秒时间戳，可用于：
    1. 作为 search_log 的 end_time 参数（查询到现在）
    2. 计算历史时间点作为 start_time 参数

    Returns:
        int: 当前时间戳（毫秒），例如: 1708012345000

    使用示例:
        # 获取当前时间
        current = get_current_timestamp()

        # 计算15分钟前的时间
        fifteen_min_ago = current - (15 * 60 * 1000)

        # 计算1小时前的时间
        one_hour_ago = current - (60 * 60 * 1000)

        # 用于搜索最近15分钟的日志
        search_log(
            topic_id="topic-001",
            start_time=fifteen_min_ago,
            end_time=current
        )
    """
    return int(datetime.now().timestamp() * 1000)

@mcp.tool()
@log_tool_call
def get_region_code_by_name(region_name: str) -> Dict[str,Any]:
    """根据地区名称搜索对应的地区参数。

    Args:
        region_name: 地区名称（如：北京、上海、广州等）

    Returns:
        Dict: 包含地区代码和相关信息的字典
            - region_code: 地区代码
            - region_name: 地区名称
            - available: 是否可用
    """
    # 模拟地区映射表（实际应该从配置或数据库读取）
    region_mapping = {
        "北京": {"region_code": "ap-beijing", "region_name": "北京", "available": True},
        "上海": {"region_code": "ap-shanghai", "region_name": "上海", "available": True},
        "广州": {"region_code": "ap-guangzhou", "region_name": "广州", "available": True},
    }

    result = region_mapping.get(region_name)
    if result:
        return result
    else:
        return {
            "region_code": None,
            "region_name": region_name,
            "available": False,
            "error": f"未找到地区: {region_name}"
        }


@mcp.tool()
@log_tool_call
def get_topic_info_by_name(topic_name:str,region_code:Optional[str]=None) -> Dict[str,Any]:
    """根据主题名称搜索相关的主题信息。

    Args:
        topic_name: 主题名称
        region_code: 地区代码（可选）

    Returns:
        Dict: 包含主题信息的字典
            - topic_id: 主题ID
            - topic_name: 主题名称
            - region_code: 所属地区
            - create_time: 创建时间
            - log_count: 日志数量
    """
    client = _get_cls_client()
    if client is not None:
        try:
            topics = _describe_topics(topic_name=topic_name, precise=True)
            for topic in topics:
                if region_code is None or topic.get("region_code") == region_code:
                    return topic
            return {
                "topic_id": None,
                "topic_name": topic_name,
                "region_code": region_code,
                "error": f"未找到主题: {topic_name}"
            }
        except Exception as e:
            logger.error(f"腾讯云 CLS 查询主题失败，回退本地/Mock：{e}")

    if _LOCAL_TOPICS is not None:
        for topic in _LOCAL_TOPICS:
            if topic.get("topic_name") == topic_name:
                if region_code is None or topic.get("region_code") == region_code:
                    return topic
        return {
            "topic_id": None,
            "topic_name": topic_name,
            "region_code": region_code,
            "error": f"未找到主题: {topic_name}"
        }

    mock_topics = [
        {
            "topic_id": "topic-001",
            "topic_name": "数据同步服务日志",
            "service_name": "data-sync-service",
            "region_code": "ap-beijing",
            "create_time": "2024-01-01 10:00:00",
            "log_count": 0,
            "description": "服务应用日志"
        }
    ]

    # 根据名称和地区筛选
    for topic in mock_topics:
        if topic['topic_name'] == topic_name:
            if region_code is None or topic["region_code"] == region_code:
                return topic

    return {
        "topic_id": None,
        "topic_name": topic_name,
        "region_code": region_code,
        "error": f"未找到主题: {topic_name}"
    }

@mcp.tool()
@log_tool_call
def search_topic_by_service_name(
        service_name: str,
        region_code:Optional[str]=None,
        fuzzy:bool = True
) -> Dict[str,Any]:
    """根据服务名称搜索相关的日志主题信息，支持模糊搜索。

    此工具用于根据服务名称查找对应的日志主题（topic），便于后续进行日志查询。

    Args:
        service_name: 服务名称（必填）
            示例: "data-sync-service", "sync", "data-sync"
            说明: 当 fuzzy=True 时，支持部分匹配

        region_code: 地区代码（可选）
            示例: "ap-beijing", "ap-shanghai"
            说明: 如果指定，只返回该地区的主题

        fuzzy: 是否启用模糊搜索（可选，默认 True）
            True: 部分匹配，例如 "sync" 可以匹配 "data-sync-service"
            False: 精确匹配，必须完全一致

    Returns:
        Dict: 搜索结果
            - total: 匹配到的主题数量
            - topics: 主题列表，每个主题包含:
                * topic_id: 主题ID（用于后续日志查询）
                * topic_name: 主题名称
                * service_name: 服务名称
                * region_code: 所属地区
                * create_time: 创建时间
                * log_count: 日志数量
                * description: 主题描述
            - query: 查询条件

    使用示例:
        # 示例1: 模糊搜索（推荐）
        search_topic_by_service_name(service_name="data-sync")
        # 可以匹配: "data-sync-service", "data-sync-worker" 等

        # 示例2: 精确搜索
        search_topic_by_service_name(
            service_name="data-sync-service",
            fuzzy=False
        )

        # 示例3: 指定地区搜索
        search_topic_by_service_name(
            service_name="sync",
            region_code="ap-beijing"
        )

        # 示例4: 查找后进行日志搜索的完整流程
        # 步骤1: 根据服务名查找 topic
        result = search_topic_by_service_name(service_name="data-sync-service")

        # 步骤2: 获取 topic_id
        topic_id = result["topics"][0]["topic_id"]  # "topic-001"

        # 步骤3: 使用 topic_id 查询日志
        current_ts = get_current_timestamp()
        start_ts = current_ts - (15 * 60 * 1000)
        search_log(
            topic_id=topic_id,
            start_time=start_ts,
            end_time=current_ts
        )
    """
    client = _get_cls_client()
    if client is not None:
        try:
            # CLS 无 service_name，按主题名模糊匹配
            topics = _describe_topics(topic_name=service_name, precise=False)
            matched = [t for t in topics
                       if not region_code or t.get("region_code") == region_code]
            return {
                "total": len(matched),
                "topics": matched,
                "data_source": "cls",
                "query": {"service_name": service_name, "region_code": region_code, "fuzzy": fuzzy},
                "message": (f"找到 {len(matched)} 个匹配的日志主题"
                            if matched else f"未找到服务 '{service_name}' 的日志主题")
            }
        except Exception as e:
            logger.error(f"腾讯云 CLS 查询主题失败，回退本地/Mock：{e}")

    if _LOCAL_TOPICS is not None:
        matched_topics = []
        for topic in _LOCAL_TOPICS:
            if region_code and topic.get("region_code") != region_code:
                continue
            topic_service_name = topic.get("service_name", "")
            if fuzzy:
                if (service_name.lower() in topic_service_name.lower() or
                        topic_service_name.lower() in service_name.lower()):
                    matched_topics.append(topic)
            else:
                if topic_service_name == service_name:
                    matched_topics.append(topic)
        return {
            "total": len(matched_topics),
            "topics": matched_topics,
            "data_source": "file",
            "query": {"service_name": service_name, "region_code": region_code, "fuzzy": fuzzy},
            "message": (f"找到 {len(matched_topics)} 个匹配的日志主题"
                        if matched_topics else f"未找到服务 '{service_name}' 的日志主题")
        }

    # Mock 主题数据（实际应该从配置或数据库读取）
    mock_topics = [
        {
            "topic_id": "topic-001",
            "topic_name": "数据同步服务日志",
            "service_name": "data-sync-service",
            "region_code": "ap-beijing",
            "create_time": "2024-01-01 10:00:00",
            "log_count": 0,
            "description": "数据同步服务的应用日志，包含同步任务执行情况"
        },
        {
            "topic_id": "topic-002",
            "topic_name": "数据同步服务错误日志",
            "service_name": "data-sync-service",
            "region_code": "ap-beijing",
            "create_time": "2024-01-01 10:00:00",
            "log_count": 0,
            "description": "数据同步服务的错误日志"
        },
        {
            "topic_id": "topic-003",
            "topic_name": "API网关服务日志",
            "service_name": "api-gateway-service",
            "region_code": "ap-shanghai",
            "create_time": "2024-01-01 10:00:00",
            "log_count": 0,
            "description": "API网关服务日志"
        }
    ]
    matched_topics = []

    # 搜索逻辑
    for topic in mock_topics:
        # 地区筛选
        if region_code and topic['region_code'] != region_code:
            continue

        # 服务名称匹配
        topic_service_name = topic.get("service_name","")

        if fuzzy:
            # 模糊匹配：服务名包含查询字符串，或查询字符串包含服务名
            if (service_name.lower() in topic_service_name.lower() or
            topic_service_name.lower() in service_name.lower()):
                matched_topics.append(topic)
        else:
            # 精确匹配
            if topic_service_name == service_name:
                matched_topics.append(topic)

    return {
        "total": len(matched_topics),
        "topics": matched_topics,
        "query": {
            "service_name": service_name,
            "region_code": region_code,
            "fuzzy": fuzzy
        },
        "message": f"找到 {len(matched_topics)} 个匹配的日志主题" if matched_topics else f"未找到服务 '{service_name}' 的日志主题"
    }

@mcp.tool()
@log_tool_call
def search_log(
        topic_id: int,
        start_time:int,
        end_time:int,
        query:Optional[str] = None,
        limit:int = 100
) -> Dict[str,Any]:
    """基于提供的查询参数搜索日志。

    Args:
        topic_id: 主题ID（必填）
            示例: "topic-001"

        start_time: 开始时间戳，单位为毫秒（必填，int类型）
            重要: 必须传递整数类型的毫秒时间戳
            获取方式:
            1. 使用 get_current_timestamp() 工具获取当前时间戳
            2. 计算历史时间: current_timestamp - (分钟数 * 60 * 1000)
            示例:
            - 当前时间: 1708012345000
            - 15分钟前: 1708012345000 - (15 * 60 * 1000) = 1708011445000
            - 1小时前: 1708012345000 - (60 * 60 * 1000) = 1708008745000

        end_time: 结束时间戳，单位为毫秒（必填，int类型）
            重要: 必须传递整数类型的毫秒时间戳
            通常使用 get_current_timestamp() 工具获取当前时间作为结束时间
            示例: 1708012345000

        query: 查询语句（可选，CLS 查询语法）
            示例: "level:ERROR" 或 "message:异常"

        limit: 返回结果数量限制（默认100，可选）

    Returns:
        Dict: 搜索结果
            - topic_id: 主题ID
            - start_time: 开始时间戳
            - end_time: 结束时间戳
            - query: 查询语句
            - limit: 结果限制
            - total: 实际返回的日志条数
            - logs: 日志列表，每条日志包含:
                * timestamp: 日志时间（格式: YYYY-MM-DD HH:MM:SS）
                * level: 日志级别
                * message: 日志内容
            - took_ms: 查询耗时（毫秒）
            - message: 查询状态消息

    使用示例:
        # 步骤1: 获取当前时间戳
        current_ts = get_current_timestamp()  # 返回: 1708012345000

        # 步骤2: 计算开始时间（15分钟前）
        start_ts = current_ts - (15 * 60 * 1000)  # 1708011445000

        # 步骤3: 搜索日志
        search_log(
            topic_id="topic-001",
            start_time=start_ts,     # int类型: 1708011445000
            end_time=current_ts,     # int类型: 1708012345000
            limit=100
        )
    """
    # 真实 CLS 优先：已配置凭证时调用腾讯云 CLS（From/To 单位为毫秒，与入参一致）
    client = _get_cls_client()
    if client is not None:
        try:
            from tencentcloud.cls.v20201016 import models

            req = models.SearchLogRequest()
            req.TopicId = str(topic_id)
            req.From = int(start_time)
            req.To = int(end_time)
            if query:
                req.QueryString = query
            req.Limit = int(limit)
            resp = client.SearchLog(req)

            logs = []
            for item in (resp.Results or []):
                try:
                    log_json = json.loads(item.LogJson or "{}")
                except (json.JSONDecodeError, TypeError):
                    log_json = {}
                message = (log_json.get("__CONTENT__") or log_json.get("message")
                           or log_json.get("content") or item.RawLog
                           or json.dumps(log_json, ensure_ascii=False))
                level = log_json.get("level") or log_json.get("L") or "INFO"
                ts = item.Time or 0
                time_str = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M:%S")
                logs.append({"timestamp": time_str, "level": level, "message": message})

            return {
                "topic_id": topic_id,
                "data_source": "cls",
                "start_time": start_time,
                "end_time": end_time,
                "query": query,
                "limit": limit,
                "total": len(logs),
                "logs": logs,
                "took_ms": 0,
                "list_over": getattr(resp, "ListOver", None),
                "message": f"成功从腾讯云 CLS 查询 {len(logs)} 条日志"
            }
        except Exception as e:
            logger.error(f"腾讯云 CLS 查询失败，回退本地/Mock：{e}")

    # 本地真实数据优先（短期路线）：按 topic + 时间范围 + 关键词过滤本地日志
    if _HAS_LOCAL_DATA:
        results = []
        for entry in _LOCAL_LOGS:
            if str(entry.get("topic_id")) != str(topic_id):
                continue
            ts = entry.get("timestamp")
            if ts:
                try:
                    t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                    t_ms = int(t.timestamp() * 1000)
                    if t_ms < start_time or t_ms > end_time:
                        continue
                except ValueError:
                    pass
            if query:
                if query.lower() not in str(entry.get("message", "")).lower():
                    continue
            results.append(entry)
        results = results[:limit]
        return {
            "topic_id": topic_id,
            "data_source": "file",
            "start_time": start_time,
            "end_time": end_time,
            "query": query,
            "limit": limit,
            "total": len(results),
            "logs": results,
            "took_ms": 0,
            "message": f"成功查询 {len(results)} 条日志（本地文件）"
        }

    # 根据 topic_id 返回不同的结果
    if topic_id == "topic-001":
        # topic-001:应用日志：动态生成INFO日志
        logs = []
        current_time_ms = start_time
        count = 0

        # 计算最大可生成的日志条数（基于时间范围）
        max_logs_by_time = int((end_time - start_time) / (60*1000)) + 1

        # 实际生成的日志数量取limit和时间范围内最大日志数的较小值
        actual_limit = min(limit,max_logs_by_time)

        while current_time_ms <= end_time and count<actual_limit:
            # 将毫秒时间戳转换为可读格式
            log_time = datetime.fromtimestamp(current_time_ms / 1000)
            time_str = log_time.strftime("%Y-%m-%d %H:%M:%S")

            log_entry = {
                "timestamp":time_str,
                "level":"INFO",
                "message":"正在同步元数据......"
            }

            logs.append(log_entry)
            count += 1

            # 下一条日志时间增加1分钟（60秒 * 1000毫秒）
            current_time_ms += 60*1000

        return {
            "topic_id": topic_id,
            "data_source": "mock",
            "start_time": start_time,
            "end_time": end_time,
            "query": query,
            "limit": limit,
            "total": len(logs),
            "logs": logs,
            "took_ms": 50,
            "message": f"成功查询 {len(logs)} 条应用日志"
        }
    else:
        # 其他topic_id:返回错误，表示topic不存在
        return {
            "topic_id": topic_id,
            "data_source": "mock",
            "start_time": start_time,
            "end_time": end_time,
            "query": query,
            "limit": limit,
            "total": 0,
            "logs": [],
            "took_ms": 0,
            "error": f"主题不存在: {topic_id}",
            "message": f"错误: 未找到主题 {topic_id}，请检查 topic_id 是否正确"
        }


if __name__ == '__main__':
    import os

    # 容器部署时需绑定 0.0.0.0 才能被其他容器访问；本地开发默认 127.0.0.1 更安全
    host = os.getenv("MCP_HOST", "127.0.0.1")
    mcp.run(transport='streamable-http', host=host, port=8003, path='/mcp')