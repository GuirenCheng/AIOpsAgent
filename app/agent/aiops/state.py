"""
通用 Plan-Execute-Replan 状态定义
基于 LangGraph 官方教程实现
"""

from typing import List,TypedDict,Annotated
import operator

class PlanExecuteState(TypedDict):
    '''Plan-Execute-Replan 状态'''

    # 用户输入
    input:str

    # 执行计划（步骤列表）
    plan:List[str]

    # 已执行的步骤历史
    # 使用operator.add实现追加式更新(而非覆盖)
    past_steps:Annotated[List[tuple],operator.add]

    # 最终响应/报告
    response:str