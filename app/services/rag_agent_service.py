"""RAG Agent 服务 - 基于 LangGraph 的智能代理

使用 langchain_qwq 的 ChatQwen 原生集成，
支持真正的流式输出和更好的模型适配。
"""
from collections.abc import AsyncGenerator
from typing import Annotated, Sequence, Any, Dict
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    before_model,
)
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_qwq import ChatQwen
from langgraph.graph import add_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from loguru import logger
from typing_extensions import TypedDict

from app.agent.guardrails import citation_check, no_context_guard, tool_failure_guard
from app.agent.mcp_client import get_mcp_client_with_retry
from app.config import config
from app.core.checkpointer import create_checkpointer
from app.tools.tool_wrapper import guarded_get_current_time, guarded_retrieve_knowledge


# 阿里千问大模型和langchain集成参考： https://docs.langchain.com/oss/python/integrations/chat/qwen
# 注意：需要配置环境变量 DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1 否则默认访问的是新加坡站点
# 同时也需要配置环境变量 DASHSCOPE_API_KEY=your_api_key

class AgentState(TypedDict):
    '''Agent 状态'''
    messages:Annotated[Sequence[BaseMessage],add_messages]

@before_model
def trim_messages_middleware(state:AgentState, runtime) -> dict[str,Any] | None:
    """
    在每次调用模型前裁剪对话历史，避免长对话撑爆上下文窗口后被硬截断，
    导致模型丢失前面的指令与事实（防幻觉/防质量骤降）。

    策略：
    - 保留第一条系统消息（System Message）
    - 保留最近的 N-1 条消息
    - 消息数不超过 rag_history_max_messages 时不做裁剪

    Args:
        state: Agent 状态（含 messages）
        runtime: LangChain 运行时上下文（此处未使用）

    Returns:
        包含裁剪后消息的状态更新字典，无需裁剪则返回 None
    """
    messages = state['messages']

    # 消息数较少时无需裁剪
    max_messages = config.rag_history_max_messages
    if len(messages) <= max_messages:
        return None

    # 提取第一条系统消息
    first_msg = messages[0]

    # 保留最近的 max_messages-1 条消息
    recent_messages = messages[-(max_messages - 1):]

    # 构建新的消息队列（先移除全部，再放回裁剪后的消息）
    new_messages = [first_msg] + list(recent_messages)

    logger.debug(f"修剪消息历史：{len(messages)} -> {len(new_messages)} 条")

    return {
        "messages":[
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


class RagAgentService:
    '''RAG Agent 服务 - 使用LangGraph + ChatQwen 原生集成'''

    def __init__(self,streaming:bool = True):
        """初始化 RAG Agent 服务

        Args:
            streaming: 是否启用流式输出，默认为 True
        """
        self.model_name = config.rag_model
        self.streaming = streaming
        self.system_prompt = self._build_system_prompt()


        self.model = ChatQwen(
            model=self.model_name,
            api_key = config.dashscope_api_key,
            base_url = config.dashscope_api_base,
            temperature=config.rag_temperature,
            streaming = streaming
        )

        # 降级模型（可选）：主模型反复失败时切换，避免拿到残缺上下文乱答。
        # 仅在配置了 rag_fallback_model 且与主模型不同时启用。
        self.fallback_model = None
        if config.rag_fallback_model and config.rag_fallback_model != self.model_name:
            self.fallback_model = ChatQwen(
                model=config.rag_fallback_model,
                api_key = config.dashscope_api_key,
                base_url = config.dashscope_api_base,
                temperature=config.rag_temperature,
                streaming = streaming
            )

        # 定义基础工具
        self.tools = [guarded_retrieve_knowledge,guarded_get_current_time]

        # MCP 客户端(延迟初始化，使用全局管理)
        self.mcp_tools:list = []

        # 创建检查点（用于会话管理；持久化与否取决于是否安装 SQLite 依赖与配置）
        self.checkpointer = create_checkpointer()

        # Agent初始化（会在异步方法中完成）
        self.agent = None
        self._agent_initialized = False

        logger.info(f"RAG Agent 服务初始化完成 (ChatQwen),model={self.model_name},streaming={self.streaming},streaming={streaming}")


    async def _initialize_agent(self):
        '''异步初始化 Agent (包括 MCP 工具)'''
        if self._agent_initialized:
            return

        # MCP 工具加载失败不应阻断服务：MCP 服务（CLS/Monitor）未启动时降级为
        # 仅本地工具（retrieve_knowledge / get_current_time），与 Milvus 不可用时的
        # 降级策略一致。langchain-mcp-adapters 内部用 asyncio.TaskGroup 并发连接，
        # 任一服务器连不上会把真实异常包成 "unhandled errors in a TaskGroup"。
        try:
            # 使用全局 MCP 客户端管理器(带重试拦截器）
            mcp_client = await get_mcp_client_with_retry()

            # 获取 MCP 工具
            self.mcp_tools = await mcp_client.get_tools()
            logger.info(f"成功加载 {len(self.mcp_tools)} 个MCP 工具")
        except Exception as e:
            self.mcp_tools = []
            logger.warning(f"MCP 工具加载失败，降级为仅本地工具（服务仍可用）：{e}")

        # 合并所有工具
        all_tools = self.tools + self.mcp_tools

        # 模型韧性中间件：重试主模型；配置了降级模型时，重试耗尽后切换降级模型
        if self.fallback_model is not None:
            model_resilience = [
                # 外层 Fallback：主模型（含内层重试）最终抛错时切换到降级模型
                ModelFallbackMiddleware(self.fallback_model),
                # 内层 Retry：先对主模型重试，耗尽后 re-raise 交给外层 Fallback
                ModelRetryMiddleware(
                    max_retries=config.rag_model_retry,
                    on_failure="error",
                ),
            ]
        else:
            # 无降级模型时，重试耗尽后返回错误提示而非崩溃
            model_resilience = [
                ModelRetryMiddleware(
                    max_retries=config.rag_model_retry,
                    on_failure="continue",
                ),
            ]

        middleware = [
            no_context_guard,       # 检索未命中/出错 → 硬短路兜底话术
            tool_failure_guard,     # 工具失败 → 改写为"禁止编造"指令
            citation_check,         # 脱稿检测（日志告警）
            *model_resilience,      # 模型重试 + 降级
            ToolCallLimitMiddleware(
                run_limit=config.rag_max_tool_calls,
                exit_behavior="continue",  # 达到上限后阻止继续调用，让模型收尾
            ),
            trim_messages_middleware,  # 历史裁剪（放最后，先兜底再裁剪）
        ]

        self.agent = create_agent(
            self.model,
            tools = all_tools,
            checkpointer = self.checkpointer,
            middleware = middleware,
        )
        self._agent_initialized = True


        if all_tools:
            tools_names = [tool.name if hasattr(tool,'name') else str(tool) for tool in all_tools]
            logger.info(f"可用工具列表: {', '.join(tools_names)}")

    def _build_system_prompt(self)->str:
        """
        构建系统提示词（含防幻觉约束）

        注意：LangChain 框架会自动将工具信息传递给 LLM，
        因此系统提示词中无需列举具体的工具列表。

        Returns:
            str: 系统提示词
        """
        from textwrap import dedent

        return dedent("""
            你是一个专业的AI助手，能够使用工具（知识库检索、时间查询等）来帮助用户解决问题。

            回答必须遵守以下铁律：
            1. 涉及事实、专业知识、文档内容的问题，必须先调用「知识检索」工具获取参考资料，
               只依据参考资料作答。
            2. 参考资料中没有的信息，严禁编造、补全或臆测；不确定就直接说「不知道」或
               「未检索到可靠信息」，不要凭自身记忆或猜测作答。
            3. 如果检索结果与问题不相关，或检索失败/未命中，如实告知用户「未找到相关资料」，
               并建议换一种问法，而不是强行给出一个看似合理的答案。
            4. 回答保持简洁、重点突出；每个事实性结论必须标注依据，形如【参考资料 1】；
               若某个说法无法对应到任何参考资料，必须明确写出「此点无据可查」，
               严禁把推断包装成事实。
            5. 如果某个工具调用失败（返回错误），严禁编造或猜测该工具的返回结果；
               如实告知用户失败原因，并建议稍后重试。

            请根据用户的问题，灵活使用可用工具，提供准确、有据可查的帮助。
        """).strip()

    async def query(
            self,
            question:str,
            session_id:str,
    )->str:
        """
        非流式处理用户问题（一次性返回完整答案）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Returns:
            str: 完整答案
        """
        try:
            await self._initialize_agent()

            logger.info(f"[会话{session_id}] RAG Agent 收到查询(非流式）:{question}")


            # 构建消息列表（系统提示 + 用户问题）
            messages = [
                SystemMessage(content = self.system_prompt),
                HumanMessage(content = question)
            ]

            # 构建 Agent 输入
            agent_input = {"messages":messages}

            # 配置thread_id（用于会话持久化）
            config_dict = {
                "configurable":{
                    "thread_id":session_id
                }
            }

            result = await self.agent.ainvoke(
                input = agent_input,
                config = config_dict,
            )

            # 提取最终答案
            messages_result = result.get('messages',[])
            if messages_result:
                last_message = messages_result[-1]
                answer = last_message.content if hasattr(last_message,'content') else str(last_message)

                # 记录工具调用
                if hasattr(last_message,"tool_calls") and last_message.tool_calls:
                    tool_names = [tc.get("name","unknown") for tc in last_message.tool_calls]
                    logger.info(f"[会话 {session_id} Agent 调用了工具: {tool_names}]")

                logger.info(f"[会话 {session_id}] RAG Agent 查询完成（非流式）")
                return answer

            logger.warning(f"[会话 {session_id}] Agent 返回结果为空")
            return ""

        except Exception as e:
            logger.error(f"[会话 {session_id} RAG Agent查询失败(非流式)：{e}")
            raise

    async def query_stream(
            self,
            question:str,
            session_id:str,
    ) -> AsyncGenerator[Dict[str,Any],None]:
        """
        流式处理用户问题（逐步返回答案片段）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Yields:
            Dict[str, Any]: 包含流式数据的字典
                - type: "content" | "tool_call" | "complete" | "error"
                - data: 具体内容
        """
        try:
            await self._initialize_agent()

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询(流式)：{question}")

            # 构建消息列表（系统提示 + 用户问题）
            messages = [
                SystemMessage(content = self.system_prompt),
                HumanMessage(content = question)
            ]

            # 构建 Agent 输入
            agent_input = {"messages":messages}

            # 配置 thread_id （用于会话持久化）
            config_dict = {
                "configurable":{
                    "thread_id":session_id
                }
            }

            async for token,metadata in self.agent.astream(
                input = agent_input,
                config = config_dict,
                stream_mode = 'messages',
            ):
                node_name = metadata.get('langgraph_node','unknown') if isinstance(metadata,dict) else 'unknown'
                message_type = type(token).__name__

                if message_type in ('AIMessage','AIMessageChunk'):
                    content_blocks = getattr(token,"content_blocks",None)

                    if content_blocks and isinstance(content_blocks,list):
                        for block in content_blocks:
                            if isinstance(block,dict) and block.get('type'):
                                text_content = block.get('text',"")
                                if text_content:
                                    yield {
                                        'type':"content",
                                        'data':text_content,
                                        'node':node_name,
                                    }
            logger.info(f"[会话 {session_id}] RAG Agent 查询完成(流式)")
            yield {'type':"complete"}

        except Exception as e:
            logger.error(f"会话 {session_id}] RAG Agent 查询失败（流式）：{e}")
            yield {
                'type':"error",
                'data':str(e),
            }
            raise

    async def get_session_history(self,session_id:str)->list:
        """
        获取会话历史（从 checkpointer 中读取）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            list: 消息历史列表 [{"role": "user|assistant", "content": "...", "timestamp": "..."}]
        """
        try:
            config = {"configurable":{"thread_id":session_id}}

            # AsyncSqliteSaver 的同步 get 在事件循环线程会抛 InvalidStateError，须用异步 aget
            checkpoint = await self.checkpointer.aget(config)

            if not checkpoint:
                logger.info(f"获取会话历史:{session_id},消息数量:0")
                return []

            # checkpoint 为 dict，消息存放在 channel_values.messages
            messages = checkpoint.get('channel_values',{}).get('messages',[])

            # 转换为前端需要的格式
            history = []
            for msg in messages:
                # 跳过系统消息
                if isinstance(msg,SystemMessage):
                    continue

                role = "user" if isinstance(msg,HumanMessage) else "assistant"
                content = msg.content if hasattr(msg,"content") else str(msg)

                # 提取时间戳（如果有的话）
                timestamp = getattr(msg,"timestamp",None)
                if timestamp:
                    history.append({
                        "role":role,
                        "content":content,
                        "timestamp":timestamp,
                    })
                else:
                    from datetime import datetime
                    history.append({
                        "role":role,
                        "content":content,
                        "timestamp":datetime.now().isoformat()
                    })

            logger.info(f"获取会话历史：{session_id},消息数量:{len(history)}")
            return history

        except Exception as e:
            logger.error(f"获取会话历史失败:{session_id},错误：{e}")
            return []

    async def clear_session(self,session_id:str)->bool:
        """
         清空会话历史（从 checkpointer 中删除）

         Args:
             session_id: 会话ID（即 thread_id）

         Returns:
             bool: 是否成功
         """
        try:
            # AsyncSqliteSaver 的同步 delete_thread 在事件循环线程会抛 InvalidStateError，须用异步
            await self.checkpointer.adelete_thread(session_id)

            logger.info(f"已清除会话历史：{session_id}")
            return True

        except Exception as e:
            logger.error(f"清空会话历史失败:{session_id},错误:{e}")
            return False

    async def clean(self):
        '''清理资源'''
        try:
            logger.info("清理 RAG Agent 服务资源...")
            # MCP 客户端由全局管理器统一管理，无需手动清理
            logger.info("RAG Agent 服务资源已清理")
        except Exception as e:
            logger.error(f"清理资源失败:{e}")



rag_agent_service = RagAgentService(streaming = True)