"""响应数据模型

定义 API 响应的 Pydantic 模型
"""

from typing import Optional, Any, List, Dict
from pydantic import BaseModel,Field


class SessionInfoResponse(BaseModel):
    '''会话信息响应'''

    session_id: str = Field(...,description='会话 ID')
    message_count: int = Field(...,description="消息数量")
    history:List[Dict[str,str]] = Field(...,description="历史消息列表")

class ApiResponse(BaseModel):
    '''通用API响应'''

    status :str = Field(...,description='状态')
    message:str = Field(...,description='消息')
    data:Optional[Any] = Field(None,description = '数据')
