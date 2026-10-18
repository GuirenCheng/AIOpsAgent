# 评测体系（D1）

把「检索质量」和「诊断准确率」的量化评估收口成一个统一入口，配合基线做回归检查。

## 三个命令

| 命令 | 作用 |
|---|---|
| `python scripts/run_eval.py` | 全量评测（检索 + 诊断），打印汇总报告 |
| `python scripts/run_eval.py --save-baseline` | 跑完把结果存成基线 `scripts/eval/baseline.json` |
| `python scripts/run_eval.py --check` | 跑完和基线对比，任一指标退化则以非零码退出 |

也可以用 Makefile：`make eval` / `make eval-baseline` / `make eval-check`。

## 前置条件

- **检索**：本地 Milvus（`localhost:19530`）已起、`aiops-docs/` 已索引；`.env` 有 `DASHSCOPE_API_KEY`（embedding 真实调用）。
- **诊断**：`.env` 有 `DASHSCOPE_API_KEY`（真实 LLM，**有 API 成本**、结果非确定）。

## 报告结构与回归规则

统一报告（JSON）：

```json
{
  "timestamp": "…",
  "retrieval": {"n": 10, "top_k": 3, "threshold": 0.5, "hit_at_3": 1.0, "precision_at_3": 0.6},
  "diagnosis": {"n": 5, "passed": 4, "accuracy": 0.8}
}
```

回归对比（`--check`）只对比「两边都存在的指标」，容差见 `app/eval_metrics.py` 的
`DEFAULT_TOLERANCES`：

| 指标 | 容差 | 含义 |
|---|---|---|
| `retrieval.hit@3` | 0 | 正确 runbook 有没有被喂给 LLM，硬门槛 |
| `retrieval.precision@3` | 0.10 | 噪声占比，允许小幅波动 |
| `diagnosis.accuracy` | 0.10 | LLM 非确定，给 10% 缓冲 |

## 如何扩充评测集

- **检索 golden**：往 `scripts/eval/retrieval_golden.json` 加条目，字段 `query`（自然语言问法）+ `relevant_file`（预期命中的 runbook 文件名）。
- **诊断场景**：往 `scripts/eval/scenarios/` 加 JSON，结构见 `cpu_high_usage_loop.json`：`alert_input` + `knowledge` + `time_now` + `tools`（canned 返回）+ `expected.root_cause_keywords` / `min_keyword_hits`。

> 真实规模的「成百上千条」需要靠长期积累线上真实案例喂进去；框架本身已就绪，扩量只需加 JSON，无需改代码。
