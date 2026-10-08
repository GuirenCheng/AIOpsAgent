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

class ClearRequest(BaseModel):
    '''清空会话请求'''

    session_id:str = Field(...,description='会话ID',alias='sessionId')

    class Config:
        populate_by_name = True
