"""请求数据模型

定义 API 请求的 Pydantic 模型
"""

from pydantic import BaseModel, Field

from app.config import config


class ChatRequest(BaseModel):
    '''对话请求'''

    id:str = Field(...,description='会话ID',alias = 'Id',min_length=1,max_length=200)
    question: str = Field(
        ...,
        description='用户问题',
        alias = 'Question',
        min_length=1,
        max_length=config.max_question_length,
    )

    class Config:
        populate_by_name = True
        json_schema_extra = {
            'example': {
                'Id':'session-123',
                'Question':'什么是向量数据库？'
            }
        }

class AgentRequest(BaseModel):
    '''统一入口请求（意图识别 + 自动分流到诊断/问答）'''

    question: str = Field(
        ...,
        description='用户问题（系统自动判断诊断/问答意图）',
        min_length=1,
        max_length=config.max_question_length,
    )
    session_id: str = Field('default', description='会话ID，用于追踪对话/诊断历史')

    class Config:
        json_schema_extra = {
            'example': {
                'question': '数据同步服务 CPU 飙高，帮我查一下原因',
                'session_id': 'session-123',
            }
        }


class ClearRequest(BaseModel):
    '''清空会话请求'''

    session_id:str = Field(...,description='会话ID',alias='sessionId')

    class Config:
        populate_by_name = True
