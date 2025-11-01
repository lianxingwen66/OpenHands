"""
OpenHands Agent 状态管理模块

这个模块定义了Agent运行状态的数据结构和管理机制，是OpenHands系统的核心状态管理组件。

技术栈：
- Python dataclasses - 数据类定义和自动生成方法
- pickle + base64 - 状态序列化和持久化
- Enum - 枚举类型定义状态常量
- 类型提示 - 提供静态类型检查
- 文件存储 - 支持状态的保存和恢复

核心概念：
1. State: Agent的完整运行状态，包含历史、配置、指标等
2. TrafficControlState: 流量控制状态，管理Agent的执行频率
3. 状态持久化: 支持会话状态的保存和恢复
4. 多Agent支持: 支持委托Agent和多层级协作
5. 指标收集: 内置LLM使用指标和成本统计

设计特点：
- 数据完整性：包含Agent运行的所有必要信息
- 序列化支持：可以保存到文件并恢复
- 多层级支持：支持Agent委托和嵌套
- 性能监控：内置指标收集和分析
- 状态一致性：确保状态转换的正确性
"""

from __future__ import annotations

import base64
import os
import pickle
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import openhands
from openhands.core.logger import openhands_logger as logger
from openhands.core.schema import AgentState
from openhands.events.action import (
    MessageAction,
)
from openhands.events.action.agent import AgentFinishAction
from openhands.events.event import Event, EventSource
from openhands.llm.metrics import Metrics
from openhands.memory.view import View
from openhands.storage.files import FileStore
from openhands.storage.locations import get_conversation_agent_state_filename


class TrafficControlState(str, Enum):
    """
    流量控制状态枚举

    定义Agent执行的流量控制状态，用于管理Agent的执行频率和资源使用。
    这有助于防止过度使用LLM API和控制成本。

    技术特点：
    - 继承自str和Enum：既是字符串又是枚举
    - 状态转换：支持在不同控制状态间切换
    - 资源管理：控制API调用频率和成本
    """

    # 默认状态，无速率限制
    # Agent可以正常执行，没有任何限制
    NORMAL = 'normal'

    # 由于流量控制而暂停任务
    # 当达到迭代次数或成本限制时进入此状态
    THROTTLING = 'throttling'

    # 流量控制临时暂停
    # 用户可以选择临时恢复执行
    PAUSED = 'paused'


# 可恢复的Agent状态列表
# 这些状态下的Agent可以从保存的会话中恢复执行
RESUMABLE_STATES = [
    AgentState.RUNNING,  # 正在运行
    AgentState.PAUSED,  # 已暂停
    AgentState.AWAITING_USER_INPUT,  # 等待用户输入
    AgentState.FINISHED,  # 已完成
]


@dataclass
class State:
    """
    OpenHands系统中Agent的运行状态表示，保存其操作和内存数据

    这个类是OpenHands系统的核心数据结构，包含了Agent运行所需的所有状态信息。
    它支持多Agent协作、状态持久化、指标收集和错误处理。

    技术架构：
    - 数据类设计：使用@dataclass自动生成构造函数和方法
    - 序列化支持：支持pickle序列化和base64编码
    - 类型安全：使用类型提示确保数据一致性
    - 默认值：使用field()提供合理的默认值

    主要功能分类：

    1. 多Agent/委托状态：
       - 存储任务（Agent与用户之间的对话）
       - 子任务（Agent与用户或其他Agent之间的对话）
       - 全局和本地迭代计数
       - 多Agent交互的委托级别
       - 几乎卡住的状态检测

    2. Agent运行状态：
       - 当前Agent状态（如LOADING、RUNNING、PAUSED）
       - 用于速率限制的流量控制状态
       - 确认模式设置
       - 遇到的最后一个错误

    3. 保存和恢复Agent的数据：
       - 保存到会话并从会话恢复
       - 使用pickle和base64进行序列化

    4. 消息历史的保存/恢复数据：
       - Agent历史中事件的开始和结束ID
       - 摘要和委托摘要

    5. 指标：
       - 当前任务的全局指标
       - 当前子任务的本地指标

    6. 额外数据：
       - 特定任务的附加数据
    """

    # 会话标识符，用于标识和恢复特定的Agent会话
    session_id: str = ''

    # 当前任务的全局迭代次数
    # 跨所有Agent（包括委托Agent）的总迭代计数
    iteration: int = 0

    # 当前子任务的本地迭代次数
    # 仅计算当前Agent的迭代次数
    local_iteration: int = 0

    # 当前任务的最大迭代次数限制
    # 用于防止无限循环和控制资源使用
    max_iterations: int = 100

    # 确认模式标志
    # 当启用时，某些动作需要用户确认才能执行
    confirmation_mode: bool = False

    # 事件历史列表
    # 包含Agent执行过程中的所有动作和观察
    history: list[Event] = field(default_factory=list)

    # 输入参数字典
    # 存储任务的初始输入和配置参数
    inputs: dict = field(default_factory=dict)

    # 输出结果字典
    # 存储任务完成后的结果和输出数据
    outputs: dict = field(default_factory=dict)

    # 当前Agent状态
    # 表示Agent的执行状态（加载中、运行中、暂停等）
    agent_state: AgentState = AgentState.LOADING

    # 恢复状态
    # 用于从保存的会话中恢复时的状态信息
    resume_state: AgentState | None = None

    # 流量控制状态
    # 用于管理Agent的执行频率和资源使用
    traffic_control_state: TrafficControlState = TrafficControlState.NORMAL

    # 当前任务的全局指标
    # 包含整个任务的LLM使用统计和成本信息
    metrics: Metrics = field(default_factory=Metrics)

    # 当前子任务的本地指标
    # 包含当前Agent的LLM使用统计和成本信息
    local_metrics: Metrics = field(default_factory=Metrics)

    # 委托级别
    # 根Agent为0级，每个委托Agent增加1级
    delegate_level: int = 0

    # 历史事件范围的开始ID
    # 用于跟踪Agent历史中事件的范围
    start_id: int = -1

    # 历史事件范围的结束ID
    # 用于跟踪Agent历史中事件的范围
    end_id: int = -1

    # 委托Agent字典
    # 键为(start_id, end_id)元组，值为(agent_name, task_description)元组
    delegates: dict[tuple[int, int], tuple[str, str]] = field(default_factory=dict)

    # 额外数据字典
    # 注意：控制器永远不会使用这个字段，但它可以被不同的评估任务用来
    # 存储跟踪任务进度/状态所需的额外数据
    extra_data: dict[str, Any] = field(default_factory=dict)

    # 最后一个错误信息
    # 存储Agent遇到的最后一个错误的详细描述
    last_error: str = ''

    def save_to_session(
        self, sid: str, file_store: FileStore, user_id: str | None
    ) -> None:
        """
        将状态保存到会话文件

        使用pickle序列化和base64编码将当前状态保存到文件存储中。
        支持用户特定的存储路径，并处理旧版本的兼容性。

        Args:
            sid (str): 会话标识符
            file_store (FileStore): 文件存储实例
            user_id (str | None): 用户ID，用于多用户环境

        Raises:
            Exception: 当保存失败时抛出异常

        技术流程：
        1. 使用pickle序列化状态对象
        2. 使用base64编码序列化数据
        3. 写入到指定的文件路径
        4. 清理旧版本的状态文件（如果存在）

        设计考虑：
        - 向后兼容：处理旧版本的文件路径
        - 错误处理：提供详细的错误信息
        - 用户隔离：支持多用户环境的数据隔离
        """
        # 使用pickle序列化状态对象
        pickled = pickle.dumps(self)
        logger.debug(f'Saving state to session {sid}:{self.agent_state}')

        # 使用base64编码序列化数据
        encoded = base64.b64encode(pickled).decode('utf-8')

        try:
            # 写入到指定的文件路径
            file_store.write(
                get_conversation_agent_state_filename(sid, user_id), encoded
            )

            # 在SaaS/远程使用情况下，检查旧目录中是否有状态文件并删除它
            if user_id:
                filename = get_conversation_agent_state_filename(sid)
                try:
                    file_store.delete(filename)
                except Exception:
                    # 忽略删除旧文件时的异常
                    pass
        except Exception as e:
            logger.error(f'Failed to save state to session: {e}')
            raise e

    @staticmethod
    def restore_from_session(
        sid: str, file_store: FileStore, user_id: str | None = None
    ) -> 'State':
        """
        从之前保存的会话中恢复状态

        从文件存储中读取序列化的状态数据，反序列化并恢复State对象。
        支持向后兼容和多用户环境。

        Args:
            sid (str): 会话标识符
            file_store (FileStore): 文件存储实例
            user_id (str | None): 用户ID，用于多用户环境

        Returns:
            State: 恢复的状态对象

        Raises:
            FileNotFoundError: 当会话文件不存在时抛出
            Exception: 当反序列化失败时抛出

        技术流程：
        1. 从文件存储读取编码的状态数据
        2. 使用base64解码数据
        3. 使用pickle反序列化状态对象
        4. 更新状态以准备恢复执行
        5. 设置适当的恢复状态

        设计考虑：
        - 向后兼容：尝试从旧路径读取文件
        - 状态重置：将Agent状态重置为LOADING
        - 恢复标记：设置resume_state用于恢复逻辑
        """
        state: State
        try:
            # 从文件存储读取编码的状态数据
            encoded = file_store.read(
                get_conversation_agent_state_filename(sid, user_id)
            )
            # 使用base64解码数据
            pickled = base64.b64decode(encoded)
            # 使用pickle反序列化状态对象
            state = pickle.loads(pickled)
        except FileNotFoundError:
            # 如果提供了user_id，我们处于SaaS/远程使用情况
            # 需要检查状态是否在旧目录中
            if user_id:
                filename = get_conversation_agent_state_filename(sid)
                encoded = file_store.read(filename)
                pickled = base64.b64decode(encoded)
                state = pickle.loads(pickled)
            else:
                raise FileNotFoundError(
                    f'Could not restore state from session file for sid: {sid}'
                )
        except Exception as e:
            logger.debug(f'Could not restore state from session: {e}')
            raise e

        # 更新状态以准备恢复执行
        if state.agent_state in RESUMABLE_STATES:
            # 保存原始状态用于恢复逻辑
            state.resume_state = state.agent_state
        else:
            # 不可恢复的状态不设置恢复状态
            state.resume_state = None

        # 恢复后的第一个状态总是LOADING
        state.agent_state = AgentState.LOADING
        return state

    def __getstate__(self) -> dict:
        """
        自定义pickle序列化行为

        在序列化时排除某些不需要持久化的属性，如历史记录和缓存。
        历史记录将从事件流中恢复，缓存将在需要时重建。

        Returns:
            dict: 要序列化的状态字典

        技术说明：
        - 排除历史：历史记录将从事件流恢复，不需要序列化
        - 清理缓存：移除视图缓存属性，它们将在重新加载后重建
        - 内存优化：减少序列化数据的大小
        """
        # 不要pickle历史记录，它将从事件流中恢复
        state = self.__dict__.copy()
        state['history'] = []

        # 移除任何视图缓存属性。它们将在历史重新加载后重建
        state.pop('_history_checksum', None)
        state.pop('_view', None)

        return state

    def __setstate__(self, state: dict) -> None:
        """
        自定义pickle反序列化行为

        在反序列化时确保所有必要的属性都存在，特别是历史记录属性。

        Args:
            state (dict): 反序列化的状态字典

        技术说明：
        - 属性恢复：更新对象的所有属性
        - 默认值：确保关键属性有默认值
        - 兼容性：处理不同版本间的属性差异
        """
        # 更新对象的所有属性
        self.__dict__.update(state)

        # 确保我们总是有history属性
        if not hasattr(self, 'history'):
            self.history = []

    def get_current_user_intent(self) -> tuple[str | None, list[str] | None]:
        """
        获取当前用户意图

        返回在FinishAction之后出现的最新用户消息和图像（如果提供），
        或者如果还没有完成任何任务则返回第一个（任务）。

        Returns:
            tuple[str | None, list[str] | None]: 用户消息内容和图像URL列表

        技术流程：
        1. 从最新事件开始反向遍历
        2. 查找用户消息和图像
        3. 检查是否有FinishAction分隔
        4. 返回适当的用户意图

        使用场景：
        - 任务理解：理解用户的当前需求
        - 上下文分析：分析用户的意图和目标
        - 多模态支持：处理文本和图像输入
        """
        last_user_message = None
        last_user_message_image_urls: list[str] | None = []

        # 从最新事件开始反向遍历视图
        for event in reversed(self.view):
            if isinstance(event, MessageAction) and event.source == 'user':
                # 找到用户消息，记录内容和图像
                last_user_message = event.content
                last_user_message_image_urls = event.image_urls
            elif isinstance(event, AgentFinishAction):
                # 如果遇到FinishAction且已找到用户消息，返回该消息
                if last_user_message is not None:
                    return last_user_message, None

        # 返回最后找到的用户消息和图像
        return last_user_message, last_user_message_image_urls

    def get_last_agent_message(self) -> MessageAction | None:
        """
        获取最后一条Agent消息

        从事件历史中查找最近的Agent发送的消息。

        Returns:
            MessageAction | None: 最后一条Agent消息，如果没有则返回None

        使用场景：
        - 对话分析：分析Agent的最后回应
        - 状态检查：检查Agent的最后输出
        - 调试工具：查看Agent的最近活动
        """
        # 从最新事件开始反向遍历
        for event in reversed(self.view):
            if isinstance(event, MessageAction) and event.source == EventSource.AGENT:
                return event
        return None

    def get_last_user_message(self) -> MessageAction | None:
        """
        获取最后一条用户消息

        从事件历史中查找最近的用户发送的消息。

        Returns:
            MessageAction | None: 最后一条用户消息，如果没有则返回None

        使用场景：
        - 用户意图：理解用户的最新需求
        - 对话分析：分析用户的输入模式
        - 上下文维护：保持对话上下文
        """
        # 从最新事件开始反向遍历
        for event in reversed(self.view):
            if isinstance(event, MessageAction) and event.source == EventSource.USER:
                return event
        return None

    def to_llm_metadata(self, agent_name: str) -> dict:
        """
        生成LLM元数据字典

        创建包含会话信息、版本信息和标签的元数据，用于LLM调用的跟踪和分析。

        Args:
            agent_name (str): Agent名称

        Returns:
            dict: LLM元数据字典

        技术特点：
        - 版本跟踪：包含OpenHands版本信息
        - 环境信息：包含Web主机信息
        - 标签系统：支持多维度的数据标记

        使用场景：
        - 调用跟踪：跟踪LLM API调用
        - 性能分析：分析不同环境的性能
        - 调试支持：提供调试所需的上下文信息
        """
        return {
            'session_id': self.session_id,
            'trace_version': openhands.__version__,
            'tags': [
                f'agent:{agent_name}',
                f'web_host:{os.environ.get("WEB_HOST", "unspecified")}',
                f'openhands_version:{openhands.__version__}',
            ],
        }

    @property
    def view(self) -> View:
        """
        获取事件历史的视图对象

        提供对事件历史的高级访问接口，支持缓存以提高性能。
        当历史记录发生变化时，自动重新创建视图。

        Returns:
            View: 事件历史的视图对象

        技术特点：
        - 缓存机制：使用校验和检测历史变化
        - 延迟计算：只在需要时重新创建视图
        - 性能优化：避免重复的视图创建操作

        设计说明：
        - 校验和：使用历史长度作为简单的校验和
        - 缓存失效：历史变化时自动失效缓存
        - 内存管理：合理管理视图对象的生命周期
        """
        # 计算历史的简单校验和，看是否可以重用缓存的视图
        history_checksum = len(self.history)
        old_history_checksum = getattr(self, '_history_checksum', -1)

        # 如果历史发生了变化，需要重新创建视图并更新缓存
        if history_checksum != old_history_checksum:
            self._history_checksum = history_checksum
            self._view = View.from_events(self.history)

        return self._view
