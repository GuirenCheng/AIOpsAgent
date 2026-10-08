"""把带 (timestamp, value) 的指标 CSV 转成 monitor_server 的本地数据格式。

支持 NAB（https://github.com/numenta/NAB，CSV 列：timestamp,value）或任意两列 CSV。

用法示例：
    python mcp_servers/import_metrics.py ec2_cpu_utilization_5ea553.csv --metric cpu
    python mcp_servers/import_metrics.py memory.csv --metric memory

输出（默认写入 mcp_servers/data/monitor/）：
    cpu_metrics.csv / memory_metrics.csv（列：timestamp,value，时间统一为 "YYYY-MM-DD HH:MM:SS"）
"""
import argparse
import csv
import os
from datetime import datetime

_TIMESTAMP_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")


def parse_ts(s: str):
    """解析时间戳，支持 ISO 与 epoch 秒，失败返回 None。"""
    s = s.strip()
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    try:
        return datetime.fromtimestamp(float(s)).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OSError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csvfile", help="包含 timestamp,value 两列的 CSV 文件")
    ap.add_argument("--metric", choices=["cpu", "memory"], required=True)
    ap.add_argument("--out-dir", default=None, help="输出目录，默认 mcp_servers/data/monitor")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "monitor"
    )
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    with open(args.csvfile, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # 有表头则跳过；否则把第一行也当数据
        if header and header[0].strip().lower() in ("timestamp", "time", "date", "ts"):
            pass
        elif header and len(header) >= 2:
            rows.append(header)
        for row in reader:
            rows.append(row)

    out_name = "cpu_metrics.csv" if args.metric == "cpu" else "memory_metrics.csv"
    out_path = os.path.join(out_dir, out_name)
    written = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "value"])
        for row in rows:
            if len(row) < 2:
                continue
            ts = parse_ts(row[0])
            if ts is None:
                continue
            try:
                val = float(row[1])
            except ValueError:
                continue
            w.writerow([ts, val])
            written += 1

    print(f"完成：{written} 个数据点 -> {out_path}")


if __name__ == "__main__":
    main()
