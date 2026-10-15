"""本地工具护栏包装：把 retrieve_knowledge / get_current_time 包进 harness 收口点。

本地工具是 sync @tool 生成的 StructuredTool，原本直接 .ainvoke 不走任何护栏。
这里用 execute_guarded 包一层，让它们与 MCP 工具一样经过审批门/并发槽位/资源锁/幂等重试/超时/审计。

retrieve_knowledge 原 response_format='content_and_artifact'，但 .ainvoke() 只返回 content 字符串、
artifact(List[Document]) 无任何调用方消费，故统一 response_format='content'，对 agent 等价无损。
"""
from langchain_core.tools import StructuredTool

from app.harness.guard import execute_guarded
from app.tools.knowledge_tool import retrieve_knowledge
from app.tools.time_tool import get_current_time


def guard_local_tool(tool) -> StructuredTool:
    """包装一个本地 StructuredTool，使其 .ainvoke 走 execute_guarded。"""
    async def _guarded(**kwargs):
        result = await execute_guarded(tool.name, kwargs, "local", lambda: tool.ainvoke(kwargs))
        # 成功透传原值；失败返回错误文案（与工具内部 catch 返回错误字符串的风格一致，agent 可见不编造）
        return result.value if result.ok else result.error

    return StructuredTool.from_function(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=_guarded,
        response_format="content",
    )


# 一次性包装单例：RAG / AIOps planner / executor 统一改引这两个
guarded_retrieve_knowledge = guard_local_tool(retrieve_knowledge)
guarded_get_current_time = guard_local_tool(get_current_time)
