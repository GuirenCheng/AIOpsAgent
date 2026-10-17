"""
AIOps 请求和响应模型
"""
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field


class AIOPsRequest(BaseModel):
    '''AIOps 诊断请求'''

    session_id:Optional[str] = Field(
        default = 'default',
        description='会话ID，用于追踪诊断历史'
    )

    class Config:
        json_shema_extra = {
            'example':{
                'session_id':"session_123"
            }
        }

class AlterInfo(BaseModel):
    '''告警信息'''
    altername:str
    severity:str
    instance:str
    duration:str
    description:Optional[str] = None

class DiagnosisResponse(BaseModel):
    '''诊断响应（非流式）'''

    code: int = 200
    message: str = 'success'
    data: Dict[str,Any]

    class Config:
        json_shema_extra = {
            'example':{
                'code':200,
                'message':'success',
                'data':{
                    'status':'completed',
                    'target_alert':{
                        'alertname':'HighCPUUsage',
                        "severity":"Critical"
                    },
                    "diagnosis":{
                        'root_cause':"数据库连接池耗尽",
                        'recommendations':['扩容数据库连接池',"优化SQL查询"]
                    }
                }
            }
        }


class AlertWebhook(BaseModel):
    '''归一化后的告警信息（监控平台 webhook 触发诊断用）'''

    alertname: str = Field(..., description='告警名称')
    severity: str = Field(default='warning', description='告警级别')
    instance: str = Field(default='', description='受影响实例/主机')
    description: Optional[str] = Field(default=None, description='告警描述')
    starts_at: Optional[str] = Field(default=None, description='首次触发时间')


class WebhookResponse(BaseModel):
    '''告警 webhook 的即时响应'''

    code: int = 202
    message: str = '告警已接收，诊断已在后台启动'
    alertname: Optional[str] = None
