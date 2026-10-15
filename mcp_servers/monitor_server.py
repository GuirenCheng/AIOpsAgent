"""智能运维监控 MCP Server

本地实现的监控服务 MCP Server，提供：
- 监控数据查询（CPU、内存、磁盘、网络等）
- 进程信息查询
- 历史工单查询
- 服务信息查询

用于支持运维 Agent 的故障排查场景。
"""
import csv
import functools
import json
import logging
import os
import random
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import FastAPI
from fastmcp import FastMCP

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("Monitor_MCP_Server")

mcp = FastMCP("Monitor")

# 真实主机指标采集：psutil 为可选依赖，未安装时 get_host_metrics 不可用
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    psutil = None
    _HAS_PSUTIL = False
    logger.warning(
        "未安装 psutil：get_host_metrics 工具不可用；"
        "query_cpu_metrics / query_memory_metrics 等时序工具为模拟数据。"
    )

# 本地真实数据（短期路线）：从 CSV（timestamp,value）加载指标时序。
# 文件不存在时返回 None，工具自动回退到模拟数据。
_MONITOR_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "monitor")


def _load_metric_csv(filename: str):
    """加载本地指标 CSV（timestamp,value），无文件返回 None。"""
    path = os.path.join(_MONITOR_DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    points = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # 跳过表头
        for row in reader:
            if len(row) < 2:
                continue
            try:
                ts = datetime.strptime(row[0].strip(), "%Y-%m-%d %H:%M:%S")
                val = float(row[1].strip())
                points.append((ts, val))
            except (ValueError, TypeError):
                continue
    return sorted(points, key=lambda p: p[0])


def _format_metric_result(service_name, metric_name, interval, start_dt, end_dt, points):
    """基于真实数据点，过滤时间范围并计算统计信息，返回统一结构。"""
    window = [(t, v) for t, v in points if start_dt <= t <= end_dt]
    if not window:
        return {
            "service_name": service_name,
            "metric_name": metric_name,
            "interval": interval,
            "data_source": "file",
            "data_points": [],
            "statistics": {},
            "message": "本地数据在该时间范围内无数据点",
        }

    data_points = [
        {"timestamp": t.strftime("%H:%M"), "value": round(v, 2)}
        for t, v in window
    ]
    values = [v for _, v in window]
    avg = round(sum(values) / len(values), 2)
    mx = max(values)
    mn = min(values)
    p95 = round(sorted(values)[int(len(values) * 0.95)] if len(values) > 1 else mx, 2)

    threshold = 80.0 if "cpu" in metric_name else 70.0
    spike = mx > threshold

    return {
        "service_name": service_name,
        "metric_name": metric_name,
        "interval": interval,
        "data_source": "file",
        "data_points": data_points,
        "statistics": {
            "avg": avg,
            "max": mx,
            "min": mn,
            "p95": p95,
            "spike_detected": spike,
        },
        "alert_info": {
            "triggered": spike,
            "threshold": threshold,
            "message": f"{metric_name} 超过 {threshold}% 阈值" if spike else "指标正常",
        },
    }


def log_tool_call(func):
    """装饰器：记录工具调用的日志，包括方法名、参数和返回状态"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        method_name = func.__name__

        # 记录调用信息
        logger.info(f"=" * 80)
        logger.info(f"调用方法:{method_name}")

        # 记录参数(排除self等）
        if kwargs:
            # 使用json.dumps格式化参数，处理可能的序列错误
            try:
                params_str = json.dumps(kwargs,ensure_ascii=False,indent = 2)
            except (TypeError,ValueError):
                params_str = str(kwargs)
            logger.info(f"参数信息:\n{params_str}")
        else:
            logger.info("参数信息:无")

        # 执行方法
        try:
            result = func(*args,**kwargs)

            # 记录返回状态
            logger.info(f"返回状态: SUCCESS")

            # 记录返回结果摘要（避免日志过长）
            if isinstance(result,dict):
                summary = {k: v if not isinstance(v,(list,dict)) else f"<{type(v).__name__} with {len(v)} items>"
                           for k,v in list(result.items())[:5]}
                logger.info(f"返回结果摘要: {json.dumps(summary,ensure_ascii=False)}")
            else:
                logger.info(f"返回结果: {result}")

            logger.info(f"=" * 80)
            return result

        except Exception as e:
            # 记录错误状态
            logger.error(f"返回状态：ERROR")
            logger.error(f"错误信息:{str(e)}")
            logger.error(f"=" * 80)
            raise

    return wrapper

# ============================================================
# 辅助函数
# ============================================================

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
            return datetime.strptime(time_str,"%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    #返回默认时间（当前时间+偏移）
    return datetime.now() + timedelta(hours=default_offset_hours)


def generate_time_series(base_time:datetime,minutes_offset: int,format_str:str = "%Y-%m-%d %H:%M:%S") -> str:
    """生成时间序列字符串。

    Args:
        base_time: 基准时间
        minutes_offset: 分钟偏移量
        format_str: 时间格式字符串

    Returns:
        str: 格式化的时间字符串
    """
    result_time = base_time+timedelta(minutes=minutes_offset)
    return result_time.strftime(format_str)




# ============================================================
# 监控数据查询工具
# ============================================================

@mcp.tool()
@log_tool_call
def query_cpu_metrics(
        service_name: str,
        start_time:Optional[str] = None,
        end_time:Optional[str] = None,
        interval: str = '1m'
) -> Dict[str,Any]:
    """查询服务的 CPU 使用率监控数据。

    Args:
        service_name: 服务名称（必填）
            示例: "data-sync-service"

        start_time: 开始时间（可选，字符串类型）
            格式: "YYYY-MM-DD HH:MM:SS"
            示例: "2026-02-14 10:00:00"
            默认值: 如果不传，默认为当前时间的1小时前
            注意: 必须使用字符串格式，而非时间戳

        end_time: 结束时间（可选，字符串类型）
            格式: "YYYY-MM-DD HH:MM:SS"
            示例: "2026-02-14 11:00:00"
            默认值: 如果不传，默认为当前时间
            注意: 必须使用字符串格式，而非时间戳

        interval: 数据聚合间隔（可选）
            可选值: "1m" (1分钟), "5m" (5分钟), "1h" (1小时)
            默认值: "1m"
            说明: 控制数据点的时间间隔

    Returns:
        Dict: CPU 监控数据
            - service_name: 服务名称
            - metric_name: 指标名称 (cpu_usage_percent)
            - interval: 数据聚合间隔
            - data_points: 数据点列表，每个点包含:
                * timestamp: 时间点（格式: HH:MM）
                * value: CPU 使用率百分比
            - statistics: 统计信息
                * average: 平均值
                * max: 最大值
                * min: 最小值
            - alert: 告警信息（如有）
                * triggered: 是否触发告警
                * threshold: 告警阈值
                * message: 告警消息

    使用示例:
        # 示例1: 使用默认时间（最近1小时）
        query_cpu_metrics(service_name="data-sync-service")

        # 示例2: 指定时间范围
        query_cpu_metrics(
            service_name="data-sync-service",
            start_time="2026-02-14 10:00:00",
            end_time="2026-02-14 11:00:00",
            interval="5m"
        )

        # 示例3: 只指定开始时间（结束时间自动为当前时间）
        query_cpu_metrics(
            service_name="data-sync-service",
            start_time="2026-02-14 10:00:00"
        )
    """
    # 解析时间参数
    start_dt = parse_time_or_default(start_time,default_offset_hours=-1)
    end_dt = parse_time_or_default(end_time,default_offset_hours=0)

    # 本地真实数据优先（短期路线）：存在 CSV 时返回真实指标，否则走下方模拟逻辑
    local_points = _load_metric_csv("cpu_metrics.csv")
    if local_points is not None:
        return _format_metric_result(service_name, "cpu_usage_percent", interval,
                                     start_dt, end_dt, local_points)

    # 解析间隔时间（interval: 1m,5m,1h 等）
    interval_minutes = 1 # 默认1分钟
    if interval.endswith('m'):
        interval_minutes = int(interval[:-1])
    elif interval.endswith('h'):
        interval_minutes = int(interval[:-1]) * 60

    # 动态生成CPU使用率数据：从低到高逐渐增长
    data_points = []
    current_time = start_dt
    time_index= 0

    # 初始化 CPU 使用率（10%）
    base_cpu = 10.0

    while current_time <= end_dt:
        # CPU 使用率逐渐升高的算法：
        # - 前几个数据点保持在 10% 左右
        # - 然后开始快速上升
        # - 最终达到 95% 左右

        if time_index < 3:
            # 初始阶段：10% 左右波动
            cpu_value = base_cpu+(time_index*0.5)
        else:
            # 上升阶段：使用指数增长模型
            growth_factor = (time_index - 2) * 8.5
            cpu_value = min(base_cpu + growth_factor,96.0)

        # 添加一些随机波动（±2%）
        cpu_value = round(cpu_value + random.uniform(-2,2),1)
        cpu_value = max(0,min(100,cpu_value))# 确保在0-100范围内

        data_point = {
            "timestamp":current_time.strftime("%H:%M"),
            "value":cpu_value,
            "process_id":"pid-12345"
        }

        data_points.append(data_point)

        # 下一个时间点
        current_time += timedelta(minutes=interval_minutes)
        time_index += 1

    # 计算统计信息
    if data_points:
        values = [d["value"]for d in data_points]
        avg_value = round(sum(values) /len(values),2)
        max_value =max(values)
        min_value =min(values)

        # 检测是否有CPU突增（超过80%）
        spike_detected = max_value > 80.0

        return {
            "service_name":service_name,
            "metric_name":"cpu_usage_percent",
            "interval":interval,
            'data_source':'simulated',
            'data_points':data_points,
            'statistics':{
                'avg':avg_value,
                'max':max_value,
                'min':min_value,
                'p95':round(sorted(values)[int(len(values)*0.95)] if len(values)>1 else max_value,2),
                'spike_detected':spike_detected
            },
            'alert_info':{
                'triggered':spike_detected,
                'threshold':80.0,
                'message':'CPU 使用率持续超过 80% 阈值' if spike_detected else "CPU使用率正常"
            }
        }
    else:
        return {
            'service_name':service_name,
            'metric_name':'cpu_usage_percent',
            'interval':interval,
            'data_points':[],
            'statistics':{},
        }

@mcp.tool()
@log_tool_call
def query_memory_metrics(
    service_name: str,
    start_time:Optional[str] = None,
    end_time:Optional[str] = None,
        interval: str = '1m'
)->Dict[str,Any]:
    """查询服务的内存使用监控数据。

    Args:
        service_name: 服务名称（必填）
            示例: "data-sync-service"

        start_time: 开始时间（可选，字符串类型）
            格式: "YYYY-MM-DD HH:MM:SS"
            示例: "2026-02-14 10:00:00"
            默认值: 如果不传，默认为当前时间的1小时前
            注意: 必须使用字符串格式，而非时间戳

        end_time: 结束时间（可选，字符串类型）
            格式: "YYYY-MM-DD HH:MM:SS"
            示例: "2026-02-14 11:00:00"
            默认值: 如果不传，默认为当前时间
            注意: 必须使用字符串格式，而非时间戳

        interval: 数据聚合间隔（可选）
            可选值: "1m" (1分钟), "5m" (5分钟), "1h" (1小时)
            默认值: "1m"

    Returns:
        Dict: 内存监控数据
            - service_name: 服务名称
            - metric_name: 指标名称 (memory_usage_percent)
            - interval: 数据聚合间隔
            - data_points: 数据点列表，每个点包含:
                * timestamp: 时间点（格式: HH:MM）
                * value: 内存使用率百分比
                * used_gb: 已使用内存（GB）
                * total_gb: 总内存（GB）
            - statistics: 统计信息
                * average: 平均值
                * max: 最大值
                * min: 最小值
            - alert: 告警信息（如有）
                * triggered: 是否触发告警
                * threshold: 告警阈值
                * message: 告警消息

    使用示例:
        # 示例1: 使用默认时间（最近1小时）
        query_memory_metrics(service_name="data-sync-service")

        # 示例2: 指定时间范围
        query_memory_metrics(
            service_name="data-sync-service",
            start_time="2026-02-14 10:00:00",
            end_time="2026-02-14 11:00:00",
            interval="5m"
        )
    """
    # 解析时间参数
    start_dt = parse_time_or_default(start_time,default_offset_hours=-1)
    end_dt = parse_time_or_default(end_time,default_offset_hours=0)

    # 本地真实数据优先（短期路线）：存在 CSV 时返回真实指标，否则走下方模拟逻辑
    local_points = _load_metric_csv("memory_metrics.csv")
    if local_points is not None:
        return _format_metric_result(service_name, "memory_usage_percent", interval,
                                     start_dt, end_dt, local_points)

    # 解析间隔时间（interval:1m,5m,1h等）
    interval_minutes = 1# 默认1分钟
    if interval.endswith('m'):
        interval_minutes = int(interval[:-1])
    elif interval.endswith('h'):
        interval_minutes = int(interval[:-1]) * 60

    # 动态生成内存使用率数据：从低到高逐渐增长
    data_points = []
    current_time = start_dt
    time_index = 0

    # 初始内存使用率（30%）
    base_memory = 30.0
    total_gb = 8.0 # 总内存8 GB

    while current_time <= end_dt:
        # 内存使用率逐渐升高的算法：
        # - 前几个数据点保持在 30% 左右
        # - 然后开始逐步上升
        # - 最终达到 85% 左右

        if time_index<3:
            # 初始阶段：30% 左右波动
            memory_value = base_memory + (time_index * 1.0)
        else:
            # 上升阶段，使用线性增长模型（内存增长比CPU慢）
            growth_factor = (time_index -2) * 5.5
            memory_value = min(base_memory + growth_factor,85.0)

        # 添加一些随机波动（±1%）
        memory_value = round(memory_value + random.uniform(-1, 1), 2)
        memory_value = max(0.0, min(100.0, memory_value)) # 确保在0-100范围内

        # 计算已使用内存（GB）
        used_gb = round((memory_value / 100.0) * total_gb, 2)

        data_point = {
            "timestamp":current_time.strftime("%H:%M"),
            "value":memory_value,
            "used_gb":used_gb,
            "total_gb":total_gb,
        }

        data_points.append(data_point)

        # 下一个时间点
        current_time += timedelta(minutes=interval_minutes)
        time_index += 1

    # 计算统计信息
    if data_points:
        values = [d['value']for d in data_points]
        avg_value = round(sum(values) /len(values),2)
        max_value =max(values)
        min_value =min(values)

        # 检测是否有内存压力（超过70%）
        memory_pressure = max_value > 70.0

        return {
            "service_name":service_name,
            "metric_name":"memory_usage_percent",
            "interval":interval,
            'data_source':'simulated',
            'data_points':data_points,
            'statistics':{
                'avg':avg_value,
                'max':max_value,
                'min':min_value,
                'p95':round(sorted(values)[int(len(values)*0.95)] if len(values)>1 else max_value,2),
                "memory_pressure":memory_pressure,
            },
            "alert_info":{
                'triggered':memory_pressure,
                'threshold':70.0,
                "message":"内存使用率超过 70% 阈值，存在内存压力" if memory_pressure else "内存使用率正常"
            }
        }
    else:
        return {
            'service_name':service_name,
            'metric_name':'memory_usage_percent',
            'interval':interval,
            'data_points':[],
            'statistics':{},
            'error':"时间范围无效或没有生成数据点"
        }


@mcp.tool()
@log_tool_call
def get_host_metrics() -> Dict[str, Any]:
    """采集当前主机的实时资源指标（CPU / 内存 / 磁盘 / 网络）。

    这是监控服务中唯一基于真实系统数据（psutil）的工具；query_cpu_metrics /
    query_memory_metrics 等时序工具在未接入监控后端（如 Prometheus）前返回模拟数据。

    Returns:
        Dict: 实时主机指标
            - timestamp: 采集时间
            - data_source: "realtime"
            - cpu: {percent, cores, physical_cores, load_avg}
            - memory: {percent, used_gb, total_gb, available_gb}
            - disk: {percent, used_gb, total_gb}
            - network: {bytes_sent, bytes_recv}
    """
    if not _HAS_PSUTIL:
        return {"error": "未安装 psutil，无法采集真实主机指标", "available": False}

    import os as _os
    import datetime as _dt

    try:
        # Windows 下取系统盘根目录，Unix 下取根目录
        disk_path = _os.path.abspath(_os.sep)

        mem = psutil.virtual_memory()
        disk = psutil.disk_usage(disk_path)
        net = psutil.net_io_counters()

        load_avg = list(psutil.getloadavg()) if hasattr(psutil, "getloadavg") else None

        return {
            "timestamp": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_source": "realtime",
            "cpu": {
                "percent": psutil.cpu_percent(interval=0.1),
                "cores": psutil.cpu_count(logical=True),
                "physical_cores": psutil.cpu_count(logical=False),
                "load_avg": load_avg,
            },
            "memory": {
                "percent": mem.percent,
                "used_gb": round(mem.used / (1024 ** 3), 2),
                "total_gb": round(mem.total / (1024 ** 3), 2),
                "available_gb": round(mem.available / (1024 ** 3), 2),
            },
            "disk": {
                "percent": disk.percent,
                "used_gb": round(disk.used / (1024 ** 3), 2),
                "total_gb": round(disk.total / (1024 ** 3), 2),
            },
            "network": {
                "bytes_sent": net.bytes_sent,
                "bytes_recv": net.bytes_recv,
            },
        }
    except Exception as e:
        return {"error": f"采集主机指标失败：{e}", "available": False}


@mcp.tool()
@log_tool_call
def restart_service(service_name: str, reason: str = "") -> Dict[str, Any]:
    """重启指定服务（Mock 演示：只模拟耗时并返回，不真正重启进程）。

    这是 harness 工具契约层的写操作样例：非只读、非幂等、高危、有回滚。
    用于验证「非幂等工具不自动重试」等契约行为；接入真实部署前请替换为实际重启逻辑。

    Args:
        service_name: 服务名称（必填），示例 "data-sync-service"
        reason: 重启原因（可选），便于审计留痕

    Returns:
        Dict: {"service_name", "status": "restarted", "reason", "restarted_at", "mock"}
    """
    time.sleep(1)  # 模拟重启耗时
    return {
        "service_name": service_name,
        "status": "restarted",
        "reason": reason or "manual",
        "restarted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mock": True,
    }


if __name__ == '__main__':
    import os

    # 容器部署时需绑定 0.0.0.0 才能被其他容器访问；本地开发默认 127.0.0.1 更安全
    host = os.getenv("MCP_HOST", "127.0.0.1")
    # 使用streamable-http 模式，运行在8004端口
    mcp.run(transport='streamable-http', host=host, port=8004, path='/mcp')


            
