"""把 Loghub 原始日志（.log）转成 cls_server 可查询的本地数据格式。

用法示例（先到 https://github.com/logpai/loghub 下载一个小数据集，如 Zookeeper）：
    python mcp_servers/import_loghub.py Zookeeper_2k.log \\
        --topic-id topic-001 --topic-name "ZooKeeper 服务日志" --service-name zookeeper

输出（默认写入 mcp_servers/data/cls/）：
    logs.jsonl   一行一条结构化日志：{"timestamp","level","message","topic_id","service_name"}
    topics.json  主题元数据列表

说明：
  - 日志级别用正则抽取（TRACE/DEBUG/INFO/WARN/ERROR/FATAL/CRITICAL），适用于
    Zookeeper / Hadoop / HDFS 等带级别标记的日志；Apache 访问日志等无级别字段的
    会默认标记为 INFO。
  - 时间戳支持 "YYYY-MM-DD HH:MM:SS"、"YYYY-MM-DDTHH:MM:SS"、"DD/Mon/YYYY:HH:MM:SS"。
    解析不出来的行会用导入时间兜底，保证 time_range 过滤可用（但精度受限）。
"""
import argparse
import json
import os
import re
from datetime import datetime

# 日志级别（按词边界匹配，避免误伤普通单词）
_LEVEL_RE = re.compile(
    r"\b(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b", re.IGNORECASE
)

# 常见时间戳模式（依次尝试），统一转成 "YYYY-MM-DD HH:MM:SS"
_TIMESTAMP_PATTERNS = [
    (re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}"), "%Y-%m-%d %H:%M:%S"),
    (re.compile(r"\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}"), "%d/%b/%Y:%H:%M:%S"),
]


def extract_level(line: str) -> str:
    m = _LEVEL_RE.search(line)
    return m.group(1).upper() if m else "INFO"


def extract_timestamp(line: str):
    """抽取并规范化时间戳，解析失败返回 None。"""
    for pattern, fmt in _TIMESTAMP_PATTERNS:
        m = pattern.search(line)
        if m:
            raw = m.group(0)
            try:
                if "T" in raw:
                    return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%S").strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                return datetime.strptime(raw, fmt).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logfile", help="Loghub 原始 .log 文件路径")
    ap.add_argument("--topic-id", default="topic-001")
    ap.add_argument("--topic-name", default="导入的日志主题")
    ap.add_argument("--service-name", default="imported-service")
    ap.add_argument("--region-code", default="ap-guangzhou")
    ap.add_argument("--out-dir", default=None, help="输出目录，默认 mcp_servers/data/cls")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "cls"
    )
    os.makedirs(out_dir, exist_ok=True)

    logs = []
    fallback_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(args.logfile, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            ts = extract_timestamp(line) or fallback_ts
            logs.append(
                {
                    "timestamp": ts,
                    "level": extract_level(line),
                    "message": line,
                    "topic_id": args.topic_id,
                    "service_name": args.service_name,
                }
            )

    topics = [
        {
            "topic_id": args.topic_id,
            "topic_name": args.topic_name,
            "service_name": args.service_name,
            "region_code": args.region_code,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "log_count": len(logs),
            "description": f"从 Loghub 导入的 {args.service_name} 日志",
        }
    ]

    logs_path = os.path.join(out_dir, "logs.jsonl")
    topics_path = os.path.join(out_dir, "topics.json")
    with open(logs_path, "w", encoding="utf-8") as f:
        for entry in logs:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open(topics_path, "w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)

    print(f"完成：{len(logs)} 条日志 -> {logs_path}")
    print(f"主题元数据 -> {topics_path}")


if __name__ == "__main__":
    main()
