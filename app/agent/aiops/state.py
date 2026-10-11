"""
通用 Plan-Execute-Replan 状态定义
基于 LangGraph 官方教程实现
"""
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Literal
import operator

from pydantic import BaseModel, Field, field_validator


# 证据类型：步骤要收集的证据类别（Replanner 据此判断证据覆盖度）
EvidenceType = Literal["logs", "metrics", "knowledge", "time", "synthesis"]

# 模型可能输出的近义词 -> 规范证据类型，避免 structured output 校验失败导致整段 fallback
_EVIDENCE_ALIASES = {
    "log": "logs",
    "metric": "metrics",
    "cpu": "metrics",
    "memory": "metrics",
    "knowledge_base": "knowledge",
    "knowledge_retrieval": "knowledge",
    "docs": "knowledge",
    "clock": "time",
    "timestamp": "time",
    "report": "synthesis",
    "analyze": "synthesis",
    "analysis": "synthesis",
}


class PlanStep(BaseModel):
    '''执行计划中的单个结构化步骤'''

    description: str = Field(..., description="该步骤的具体、可操作描述")
    evidence: EvidenceType = Field(
        ..., description="该步骤要收集的证据类型：logs/metrics/knowledge/time/synthesis"
    )
    tool_name: Optional[str] = Field(None, description="如果该步骤明确需要某个工具，填写工具名")
    tool_args: Optional[Dict[str, Any]] = Field(None, description="tool_name 对应的调用参数")

    @field_validator("evidence", mode="before")
    @classmethod
    def _normalize_evidence(cls, v):
        if isinstance(v, str):
            v = v.strip().lower()
            return _EVIDENCE_ALIASES.get(v, v)
        return v


class Observation(BaseModel):
    '''单步执行后产生的观察记录'''

    step_description: str
    evidence: EvidenceType
    tool_name: Optional[str] = None
    tool_names: List[str] = Field(default_factory=list)
    success: bool
    result_summary: str

    @field_validator("evidence", mode="before")
    @classmethod
    def _normalize_evidence(cls, v):
        if isinstance(v, str):
            v = v.strip().lower()
            return _EVIDENCE_ALIASES.get(v, v)
        return v


def coerce_step(step) -> PlanStep:
    '''把 plan 中的步骤归一化为 PlanStep（兼容旧 checkpoint 的 str / serde 降级的 dict）'''
    if isinstance(step, PlanStep):
        return step
    if isinstance(step, dict):
        data = dict(step)
        data.setdefault("evidence", "synthesis")
        return PlanStep(**data)
    if isinstance(step, str):
        return PlanStep(description=step, evidence="synthesis")
    raise TypeError(f"unsupported plan step type: {type(step)}")


def coerce_observation(obs) -> Observation:
    '''把 past_steps 中的观察归一化为 Observation（兼容旧 tuple / serde 降级的 dict）'''
    if isinstance(obs, Observation):
        return obs
    if isinstance(obs, dict):
        data = dict(obs)
        data.setdefault("evidence", "synthesis")
        data.setdefault("success", True)
        data.setdefault("step_description", "")
        data.setdefault("result_summary", "")
        return Observation(**data)
    if isinstance(obs, (tuple, list)) and len(obs) == 2:
        task, result = obs
        result_str = str(result)
        return Observation(
            step_description=str(task),
            evidence="synthesis",
            success=not result_str.startswith("执行失败"),
            result_summary=result_str,
        )
    raise TypeError(f"unsupported observation type: {type(obs)}")


class PlanExecuteState(TypedDict):
    '''Plan-Execute-Replan 状态'''

    # 用户输入
    input: str

    # 执行计划（结构化步骤列表）
    plan: List[PlanStep]

    # 已执行的观察历史
    # 使用operator.add实现追加式更新(而非覆盖)
    past_steps: Annotated[List[Observation], operator.add]

    # 最终响应/报告
    response: str
