"""
OpenHands 事件系统核心模块

技术栈:
- Python 3.12+ (核心语言)
- dataclasses - 数据类装饰器，简化类定义
- datetime - 时间处理
- Enum - 枚举类型，提供类型安全
- Pydantic - 数据验证和序列化 (通过Metrics)

架构说明:
事件系统是OpenHands的核心通信机制，所有的交互都通过事件进行。
事件驱动架构确保了系统的松耦合和可扩展性。

事件流程:
User Input → MessageAction → Agent → Action → Runtime → Observation → Agent → ...

设计模式:
- 事件驱动架构: 所有交互通过事件传递
- 观察者模式: 事件流订阅和通知
- 命令模式: Action作为可执行命令
- 状态模式: 通过事件改变系统状态
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from openhands.events.tool import ToolCallMetadata  # 工具调用元数据
from openhands.llm.metrics import Metrics  # LLM调用指标


class EventSource(str, Enum):
    """
    事件源枚举

    标识事件的来源，用于:
    1. 事件路由和处理
    2. 权限控制和验证
    3. 日志记录和审计
    4. 用户界面显示
    """

    AGENT = 'agent'  # 来自Agent的事件 (Action)
    USER = 'user'  # 来自用户的事件 (用户输入)
    ENVIRONMENT = 'environment'  # 来自环境的事件 (Observation)


class FileEditSource(str, Enum):
    """
    文件编辑源枚举

    标识文件编辑操作的来源，用于:
    1. 编辑历史追踪
    2. 权限控制
    3. 审计日志
    4. 冲突解决
    """

    LLM_BASED_EDIT = 'llm_based_edit'  # LLM驱动的编辑
    OH_ACI = 'oh_aci'  # OpenHands ACI (Agent Code Interface)


class FileReadSource(str, Enum):
    """
    文件读取源枚举

    标识文件读取操作的来源，用于:
    1. 访问控制
    2. 缓存策略
    3. 性能优化
    4. 安全审计
    """

    OH_ACI = 'oh_aci'  # OpenHands ACI
    DEFAULT = 'default'  # 默认读取方式


class RecallType(str, Enum):
    """
    回忆类型枚举

    定义可以从微代理检索的信息类型，用于:
    1. 知识管理和检索
    2. 上下文增强
    3. 任务专门化
    4. 记忆系统
    """

    WORKSPACE_CONTEXT = 'workspace_context'
    """
    工作空间上下文

    包含仓库指令、运行时环境、项目结构等信息
    """

    KNOWLEDGE = 'knowledge'
    """
    知识微代理

    包含专门领域的知识和技能
    """


@dataclass
class Event:
    """
    事件基类

    OpenHands中所有事件的基类，定义了事件的通用属性和行为。
    事件是系统中信息传递的基本单位，包含了执行操作所需的所有信息。

    核心职责:
    1. 提供事件的基本属性 (ID、时间戳、来源等)
    2. 支持事件的序列化和反序列化
    3. 管理事件的生命周期
    4. 提供事件的元数据支持

    事件类型:
    - Action: Agent发出的操作指令
    - Observation: 环境对Action的响应
    - Message: 用户和Agent之间的消息

    属性说明:
    - _id: 事件唯一标识符
    - _timestamp: 事件创建时间
    - _source: 事件来源 (Agent/User/Environment)
    - _cause: 触发此事件的原因事件ID
    - _timeout: 事件超时时间
    - _llm_metrics: LLM调用指标
    - _tool_call_metadata: 工具调用元数据
    - _response_id: LLM响应ID
    """

    INVALID_ID = -1

    @property
    def message(self) -> str | None:
        if hasattr(self, '_message'):
            msg = getattr(self, '_message')
            return str(msg) if msg is not None else None
        return ''

    @property
    def id(self) -> int:
        if hasattr(self, '_id'):
            id_val = getattr(self, '_id')
            return int(id_val) if id_val is not None else Event.INVALID_ID
        return Event.INVALID_ID

    @property
    def timestamp(self) -> str | None:
        if hasattr(self, '_timestamp') and isinstance(self._timestamp, str):
            ts = getattr(self, '_timestamp')
            return str(ts) if ts is not None else None
        return None

    @timestamp.setter
    def timestamp(self, value: datetime) -> None:
        if isinstance(value, datetime):
            self._timestamp = value.isoformat()

    @property
    def source(self) -> EventSource | None:
        if hasattr(self, '_source'):
            src = getattr(self, '_source')
            return EventSource(src) if src is not None else None
        return None

    @property
    def cause(self) -> int | None:
        if hasattr(self, '_cause'):
            cause_val = getattr(self, '_cause')
            return int(cause_val) if cause_val is not None else None
        return None

    @property
    def timeout(self) -> float | None:
        if hasattr(self, '_timeout'):
            timeout_val = getattr(self, '_timeout')
            return float(timeout_val) if timeout_val is not None else None
        return None

    def set_hard_timeout(self, value: float | None, blocking: bool = True) -> None:
        """Set the timeout for the event.

        NOTE, this is a hard timeout, meaning that the event will be blocked
        until the timeout is reached.
        """
        self._timeout = value
        if value is not None and value > 600:
            from openhands.core.logger import openhands_logger as logger

            logger.warning(
                'Timeout greater than 600 seconds may not be supported by '
                'the runtime. Consider setting a lower timeout.'
            )

        # Check if .blocking is an attribute of the event
        if hasattr(self, 'blocking'):
            # .blocking needs to be set to True if .timeout is set
            self.blocking = blocking

    # optional metadata, LLM call cost of the edit
    @property
    def llm_metrics(self) -> Metrics | None:
        if hasattr(self, '_llm_metrics'):
            metrics = getattr(self, '_llm_metrics')
            return metrics if isinstance(metrics, Metrics) else None
        return None

    @llm_metrics.setter
    def llm_metrics(self, value: Metrics) -> None:
        self._llm_metrics = value

    # optional field, metadata about the tool call, if the event has a tool call
    @property
    def tool_call_metadata(self) -> ToolCallMetadata | None:
        if hasattr(self, '_tool_call_metadata'):
            metadata = getattr(self, '_tool_call_metadata')
            return metadata if isinstance(metadata, ToolCallMetadata) else None
        return None

    @tool_call_metadata.setter
    def tool_call_metadata(self, value: ToolCallMetadata) -> None:
        self._tool_call_metadata = value

    # optional field, the id of the response from the LLM
    @property
    def response_id(self) -> str | None:
        if hasattr(self, '_response_id'):
            return self._response_id  # type: ignore[attr-defined]
        return None

    @response_id.setter
    def response_id(self, value: str) -> None:
        self._response_id = value
