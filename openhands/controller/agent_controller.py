"""
OpenHands Agent Controller 核心控制器模块

这是OpenHands系统的核心控制器，负责管理AI Agent的完整生命周期，包括初始化、执行、状态管理、
错误处理、多Agent协作和资源控制。它是整个系统的大脑，协调所有组件的工作。

技术栈详解：
=============

核心技术栈：
- Python 3.8+ (现代Python特性，类型提示，异步编程)
- asyncio (高性能异步事件循环，并发处理)
- LiteLLM (统一的大语言模型接口，支持多种LLM提供商)
- 事件驱动架构 (Event-driven Architecture)
- 观察者模式 (Observer Pattern)
- 状态机模式 (State Machine Pattern)

设计模式应用：
- 观察者模式: EventStream订阅机制，实现松耦合的事件通信
- 状态机模式: Agent状态管理，清晰的状态转换逻辑
- 工厂模式: 动态创建委托Agent实例
- 策略模式: 不同的错误处理和恢复策略
- 责任链模式: 事件处理链，按优先级处理不同类型的事件
- 模板方法模式: 定义Agent执行的标准流程

异步编程架构：
- 基于asyncio的高性能异步事件处理
- 非阻塞的Agent步骤执行
- 并发的多Agent管理
- 异步的状态持久化和恢复

核心功能模块：
1. Agent生命周期管理 - 创建、初始化、执行、销毁
2. 事件流处理 - 订阅、分发、过滤事件
3. 状态管理 - 持久化、恢复、同步状态
4. 多Agent协作 - 委托机制、层级管理
5. 错误处理 - 异常捕获、恢复、降级
6. 性能监控 - 指标收集、成本控制、流量管理
7. 循环检测 - 防止Agent陷入无限循环
8. 轨迹重放 - 支持调试和测试

技术特点：
- 高并发: 支持多个Agent同时运行
- 高可用: 完善的错误处理和恢复机制
- 可扩展: 插件化架构，易于添加新功能
- 可监控: 内置指标收集和性能分析
- 可调试: 支持轨迹重放和详细日志
"""

from __future__ import annotations

import asyncio
import copy
import os
import time
import traceback
from typing import Callable

# LiteLLM - 统一的大语言模型接口库
# 支持OpenAI、Anthropic、Google等多种LLM提供商
import litellm  # noqa

# LiteLLM异常类型 - 用于精确的错误处理和分类
from litellm.exceptions import (  # noqa
    APIConnectionError,  # API连接错误
    APIError,  # 通用API错误
    AuthenticationError,  # 认证错误
    BadRequestError,  # 请求格式错误
    ContentPolicyViolationError,  # 内容政策违规
    ContextWindowExceededError,  # 上下文窗口超限
    InternalServerError,  # 服务器内部错误
    NotFoundError,  # 资源未找到
    OpenAIError,  # OpenAI特定错误
    RateLimitError,  # 速率限制错误
    ServiceUnavailableError,  # 服务不可用
    Timeout,  # 超时错误
)

# OpenHands核心组件导入
# ========================
# Controller模块 - 核心控制组件
from openhands.controller.agent import Agent  # Agent抽象基类
from openhands.controller.replay import ReplayManager  # 轨迹重放管理器
from openhands.controller.state.state import State, TrafficControlState  # 状态管理
from openhands.controller.stuck import StuckDetector  # 循环检测器

# 配置管理 - Agent和LLM配置
from openhands.core.config import AgentConfig, LLMConfig

# 异常处理 - 专门的异常类型，用于精确的错误分类和处理
from openhands.core.exceptions import (
    AgentStuckInLoopError,  # Agent陷入循环错误
    FunctionCallNotExistsError,  # 函数调用不存在错误
    FunctionCallValidationError,  # 函数调用验证错误
    LLMContextWindowExceedError,  # LLM上下文窗口超限错误
    LLMMalformedActionError,  # LLM格式错误的动作
    LLMNoActionError,  # LLM未返回动作错误
    LLMResponseError,  # LLM响应错误
)

# 日志系统 - 结构化日志和调试支持
from openhands.core.logger import LOG_ALL_EVENTS  # 全事件日志标志
from openhands.core.logger import openhands_logger as logger  # 主日志器

# 核心模式定义
from openhands.core.schema import AgentState  # Agent状态枚举

# 事件系统 - 事件驱动架构的核心组件
from openhands.events import (
    EventSource,  # 事件源枚举（USER, AGENT, ENVIRONMENT）
    EventStream,  # 事件流管理器
    EventStreamSubscriber,  # 事件流订阅者枚举
    RecallType,  # 回忆类型枚举
)

# 动作事件 - Agent可以执行的各种动作类型
from openhands.events.action import (
    Action,  # 动作基类
    ActionConfirmationStatus,  # 动作确认状态
    AgentDelegateAction,  # Agent委托动作
    AgentFinishAction,  # Agent完成动作
    AgentRejectAction,  # Agent拒绝动作
    ChangeAgentStateAction,  # 改变Agent状态动作
    CmdRunAction,  # 命令执行动作
    IPythonRunCellAction,  # IPython代码执行动作
    MessageAction,  # 消息动作
    NullAction,  # 空动作
    SystemMessageAction,  # 系统消息动作
)

# Agent特定动作 - Agent内部使用的特殊动作
from openhands.events.action.agent import CondensationAction, RecallAction

# 事件基础设施
from openhands.events.event import Event  # 事件基类
from openhands.events.event_filter import EventFilter  # 事件过滤器

# 观察事件 - 环境对Agent动作的响应
from openhands.events.observation import (
    AgentDelegateObservation,  # Agent委托观察
    AgentStateChangedObservation,  # Agent状态变化观察
    ErrorObservation,  # 错误观察
    NullObservation,  # 空观察
    Observation,  # 观察基类
)

# 事件序列化 - 用于轨迹记录和重放
from openhands.events.serialization.event import event_to_trajectory, truncate_content

# LLM系统 - 大语言模型管理和指标收集
from openhands.llm.llm import LLM  # LLM接口封装
from openhands.llm.metrics import Metrics, TokenUsage  # 性能指标和令牌使用统计

# 内存管理 - 事件历史的高级视图
from openhands.memory.view import View

# 系统常量定义
# ===============

# 流量控制提醒消息（仅在Web GUI中可用）
TRAFFIC_CONTROL_REMINDER = (
    "Please click on resume button if you'd like to continue, or start a new task."
)

# 动作未执行错误的标识符和消息
ERROR_ACTION_NOT_EXECUTED_ID = 'AGENT_ERROR$ERROR_ACTION_NOT_EXECUTED'
ERROR_ACTION_NOT_EXECUTED = (
    'The action has not been executed. This may have occurred because the user '
    'pressed the stop button, or because the runtime system crashed and restarted '
    'due to resource constraints. Any previously established system state, '
    'dependencies, or environment variables may have been lost.'
)


class AgentController:
    """
    OpenHands Agent控制器 - 系统核心控制组件

    这是OpenHands系统的核心控制器类，负责管理AI Agent的完整生命周期。
    它实现了事件驱动架构，支持多Agent协作，并提供完善的错误处理和性能监控。

    技术架构特点：
    ================

    1. 事件驱动架构 (Event-Driven Architecture):
       - 基于EventStream的异步消息传递
       - 观察者模式实现松耦合通信
       - 支持事件过滤和路由

    2. 异步编程模型 (Async Programming):
       - 基于asyncio的高性能异步处理
       - 非阻塞的Agent步骤执行
       - 并发的多Agent管理

    3. 状态机模式 (State Machine):
       - 清晰的Agent状态转换逻辑
       - 状态持久化和恢复
       - 状态一致性保证

    4. 多Agent协作 (Multi-Agent Collaboration):
       - 委托机制支持任务分解
       - 层级化的Agent管理
       - 父子Agent状态同步

    5. 错误处理和恢复 (Error Handling & Recovery):
       - 分层的异常处理机制
       - 自动错误恢复策略
       - 优雅降级处理

    6. 性能监控 (Performance Monitoring):
       - 实时指标收集
       - 成本控制和预算管理
       - 流量控制和限流

    7. 循环检测 (Loop Detection):
       - 智能循环模式识别
       - 防止Agent陷入无限循环
       - 多种检测算法

    核心功能模块：
    =============

    - Agent生命周期管理: 创建、初始化、执行、销毁
    - 事件流处理: 订阅、分发、过滤事件
    - 状态管理: 持久化、恢复、同步状态
    - 多Agent协作: 委托机制、层级管理
    - 错误处理: 异常捕获、恢复、降级
    - 性能监控: 指标收集、成本控制、流量管理
    - 循环检测: 防止Agent陷入无限循环
    - 轨迹重放: 支持调试和测试

    使用场景：
    =========

    1. 单Agent任务执行
    2. 多Agent协作和委托
    3. 长时间运行的复杂任务
    4. 需要状态持久化的会话
    5. 需要成本控制的生产环境
    6. 调试和测试场景
    """

    # 核心标识和组件
    # ===============

    id: str  # 控制器唯一标识符，用于日志和会话管理
    agent: Agent  # 被控制的Agent实例
    max_iterations: int  # 最大迭代次数限制，防止无限循环
    event_stream: EventStream  # 事件流管理器，实现事件驱动通信
    state: State  # Agent状态对象，包含历史和配置
    confirmation_mode: bool  # 确认模式，某些动作需要用户确认

    # 多Agent配置管理
    # ===============

    agent_to_llm_config: dict[str, LLMConfig]  # Agent名称到LLM配置的映射
    agent_configs: dict[str, AgentConfig]  # Agent名称到Agent配置的映射

    # 多Agent协作支持
    # ===============

    parent: 'AgentController | None' = None  # 父控制器引用（委托场景）
    delegate: 'AgentController | None' = None  # 委托控制器引用（子Agent）

    # 内部状态管理
    # =============

    _pending_action_info: tuple[Action, float] | None = None  # (动作, 时间戳)
    _closed: bool = False  # 控制器是否已关闭
    _cached_first_user_message: MessageAction | None = None  # 缓存的首条用户消息

    def __init__(
        self,
        agent: Agent,
        event_stream: EventStream,
        max_iterations: int,
        max_budget_per_task: float | None = None,
        agent_to_llm_config: dict[str, LLMConfig] | None = None,
        agent_configs: dict[str, AgentConfig] | None = None,
        sid: str | None = None,
        confirmation_mode: bool = False,
        initial_state: State | None = None,
        is_delegate: bool = False,
        headless_mode: bool = True,
        status_callback: Callable | None = None,
        replay_events: list[Event] | None = None,
    ):
        """
        初始化AgentController实例 - 系统核心初始化流程

        这是AgentController的构造函数，负责设置所有必要的组件和配置。
        它实现了复杂的初始化逻辑，包括事件流订阅、状态管理、多Agent支持等。

        技术架构初始化：
        ================

        1. 事件驱动架构设置:
           - 订阅EventStream实现观察者模式
           - 配置事件过滤器和路由规则
           - 设置异步事件处理机制

        2. 状态管理初始化:
           - 初始化或恢复Agent状态
           - 设置状态持久化机制
           - 配置状态同步策略

        3. 多Agent协作准备:
           - 配置委托Agent的LLM和配置映射
           - 设置父子Agent关系
           - 初始化Agent层级管理

        4. 性能监控设置:
           - 初始化指标收集器
           - 设置成本控制和预算管理
           - 配置流量控制机制

        5. 错误处理准备:
           - 设置异常处理策略
           - 初始化循环检测器
           - 配置错误恢复机制

        Args:
            agent (Agent): 要控制的Agent实例
                技术说明: Agent必须实现step()方法和相关接口

            event_stream (EventStream): 事件流管理器
                技术说明: 实现观察者模式的核心组件，支持异步事件传递

            max_iterations (int): Agent可运行的最大迭代次数
                技术说明: 防止无限循环的重要安全机制

            max_budget_per_task (float | None): 每个任务的最大预算（美元）
                技术说明: 成本控制机制，超出预算时Agent将停止执行

            agent_to_llm_config (dict[str, LLMConfig] | None): Agent名称到LLM配置的映射
                技术说明: 支持多Agent场景下的不同LLM配置

            agent_configs (dict[str, AgentConfig] | None): Agent名称到Agent配置的映射
                技术说明: 支持委托不同类型的Agent时使用不同配置

            sid (str | None): 会话ID
                技术说明: 用于会话管理和状态持久化的唯一标识符

            confirmation_mode (bool): 是否启用Agent动作确认模式
                技术说明: 安全机制，某些敏感动作需要用户确认

            initial_state (State | None): 控制器的初始状态
                技术说明: 支持从保存的会话中恢复或从父Agent继承状态

            is_delegate (bool): 此控制器是否为委托控制器
                技术说明: 委托控制器不直接订阅事件流，由父控制器管理

            headless_mode (bool): Agent是否在无头模式下运行
                技术说明: 影响用户交互和循环检测策略

            status_callback (Callable | None): 状态更新回调函数
                技术说明: 用于向外部系统报告状态变化

            replay_events (list[Event] | None): 要重放的事件列表
                技术说明: 支持轨迹重放功能，用于调试和测试

        初始化流程：
        ===========

        1. 基础属性设置
        2. 事件流订阅（仅非委托控制器）
        3. 事件过滤器配置
        4. 状态初始化或恢复
        5. 多Agent配置设置
        6. 性能监控组件初始化
        7. 循环检测器创建
        8. 轨迹重放管理器设置
        9. 系统消息添加

        技术考虑：
        ==========

        - 线程安全: 初始化过程是线程安全的
        - 异常处理: 初始化失败时提供清晰的错误信息
        - 资源管理: 合理管理内存和系统资源
        - 可扩展性: 支持未来添加新的初始化步骤
        """

        # 1. 基础属性设置
        # ================

        # 设置控制器唯一标识符，优先使用提供的sid，否则使用事件流的sid
        self.id = sid or event_stream.sid

        # 绑定要控制的Agent实例
        self.agent = agent

        # 设置运行模式标志
        self.headless_mode = headless_mode  # 无头模式影响用户交互策略
        self.is_delegate = is_delegate  # 委托模式影响事件订阅策略

        # 2. 事件流设置和订阅
        # ===================

        # 事件流必须在订阅之前设置
        self.event_stream = event_stream

        # 只有非委托控制器才直接订阅事件流
        # 委托控制器的事件由父控制器转发
        if not self.is_delegate:
            self.event_stream.subscribe(
                EventStreamSubscriber.AGENT_CONTROLLER,  # 订阅者类型
                self.on_event,  # 事件处理回调
                self.id,  # 订阅者标识
            )

        # 3. 事件过滤器配置
        # ===================

        # 配置Agent历史事件过滤器，过滤掉与Agent不相关的事件
        # 这些事件不会包含在Agent的历史记录中，提高处理效率
        self.agent_history_filter = EventFilter(
            exclude_types=(
                NullAction,  # 空动作 - 无实际意义
                NullObservation,  # 空观察 - 无实际内容
                ChangeAgentStateAction,  # 状态变更动作 - 内部管理用
                AgentStateChangedObservation,  # 状态变更观察 - 内部管理用
            ),
            exclude_hidden=True,  # 排除隐藏事件
        )

        # 4. 状态初始化或恢复
        # ===================

        # 设置初始状态：可能来自之前的会话、父Agent或全新状态
        # 这是状态机模式的核心初始化步骤
        self.set_initial_state(
            state=initial_state,  # 初始状态对象
            max_iterations=max_iterations,  # 最大迭代次数
            confirmation_mode=confirmation_mode,  # 确认模式设置
        )

        # 5. 多Agent配置设置
        # ===================

        # 设置成本控制预算
        self.max_budget_per_task = max_budget_per_task

        # 配置多Agent场景下的LLM配置映射
        # 支持不同Agent使用不同的LLM配置
        self.agent_to_llm_config = agent_to_llm_config if agent_to_llm_config else {}

        # 配置多Agent场景下的Agent配置映射
        # 支持委托到不同类型的Agent
        self.agent_configs = agent_configs if agent_configs else {}

        # 保存初始配置，用于重置和恢复
        self._initial_max_iterations = max_iterations
        self._initial_max_budget_per_task = max_budget_per_task

        # 6. 循环检测器初始化
        # ====================

        # 创建循环检测器，防止Agent陷入无限循环
        # 这是系统稳定性的重要保障
        self._stuck_detector = StuckDetector(self.state)

        # 设置状态更新回调函数
        self.status_callback = status_callback

        # 7. 轨迹重放管理器设置
        # ======================

        # 初始化重放管理器，支持调试和测试场景
        # 可以重放之前记录的事件序列
        self._replay_manager = ReplayManager(replay_events)

        # 8. 系统消息添加
        # ================

        # 向事件流添加系统消息，为Agent提供初始上下文
        # 这是Agent执行的重要准备步骤
        self._add_system_message()

    def _add_system_message(self):
        """
        添加系统消息到事件流

        这个方法负责向事件流添加Agent的系统消息，为Agent提供初始上下文和指令。
        系统消息是Agent执行的重要基础，包含了Agent的角色定义、能力说明和行为指导。

        技术实现：
        ==========

        1. 重复检查机制:
           - 检查事件流中是否已存在系统消息
           - 避免重复添加相同的系统消息
           - 处理向后兼容性问题

        2. 消息生成和添加:
           - 从Agent获取系统消息内容
           - 验证消息内容的有效性
           - 将消息添加到事件流中

        3. 日志记录:
           - 记录系统消息的预览内容
           - 便于调试和监控

        设计考虑：
        ==========

        - 向后兼容性: 处理旧版本事件流的兼容性
        - 幂等性: 多次调用不会产生副作用
        - 性能优化: 避免不必要的消息重复
        - 调试支持: 提供清晰的日志信息
        """

        # 遍历事件流中的现有事件，检查是否需要添加系统消息
        for event in self.event_stream.get_events(start_id=self.state.start_id):
            if isinstance(event, MessageAction) and event.source == EventSource.USER:
                # 向后兼容性处理 - 计划在2025年6月1日后移除
                # 如果首先遇到用户消息，说明事件流在引入SystemMessageAction之前就存在
                # 我们期望Agent能够优雅地处理这种情况
                return

            if isinstance(event, SystemMessageAction):
                # 如果系统消息已经存在，不要重复添加
                # 这确保了幂等性 - 多次调用不会产生副作用
                return

        # 向事件流添加系统消息
        # 这应该对所有Agent执行，包括委托Agent
        system_message = self.agent.get_system_message()

        # 验证系统消息的有效性
        if system_message and system_message.content:
            # 创建消息预览用于日志记录
            # 限制预览长度以避免日志过长
            preview = (
                system_message.content[:50] + '...'
                if len(system_message.content) > 50
                else system_message.content
            )

            # 记录调试日志，便于监控和调试
            logger.debug(f'System message: {preview}')

            # 将系统消息添加到事件流中，标记为Agent源
            self.event_stream.add_event(system_message, EventSource.AGENT)

    async def close(self, set_stop_state: bool = True) -> None:
        """
        关闭Agent控制器 - 优雅的资源清理和状态保存

        这是控制器生命周期的最后阶段，负责取消正在进行的任务、取消事件流订阅、
        并确保状态的完整性。正确的关闭过程对于数据完整性至关重要。

        技术实现：
        ==========

        1. 状态设置:
           - 将Agent状态设置为STOPPED
           - 确保状态转换的一致性

        2. 历史记录整理:
           - 从事件流中提取完整的历史记录
           - 包含委托Agent的事件
           - 应用事件过滤器排除无关事件

        3. 资源清理:
           - 取消事件流订阅
           - 标记控制器为已关闭状态
           - 释放相关资源

        Args:
            set_stop_state (bool): 是否设置停止状态
                技术说明: 某些情况下可能不需要设置停止状态

        设计考虑：
        ==========

        - 数据完整性: 确保历史记录的完整性
        - 资源管理: 正确释放所有资源
        - 异常安全: 即使出现异常也要尽量清理资源
        - 性能优化: 高效的历史记录提取

        注意事项：
        =========

        正确关闭非常重要，否则状态将不完整，可能影响：
        - 评估脚本的运行
        - 测试结果的准确性
        - 会话恢复的可靠性
        """

        # 1. 设置Agent状态为停止
        # ========================

        if set_stop_state:
            # 异步设置Agent状态，确保状态转换的一致性
            await self.set_agent_state_to(AgentState.STOPPED)

        # 2. 历史记录整理和保存
        # ======================

        # 我们创造了历史，现在是重写它的时候了！
        # 最终的state.history将被外部脚本（如评估、测试等）使用
        # 历史记录需要包含委托Agent的事件
        # 像常规Agent历史一样，它不包括：
        # - '隐藏'事件，hidden=True的事件
        # - 后端事件（默认'过滤掉'的类型，self.filter_out中的类型）

        # 确定历史记录的起始ID
        start_id = self.state.start_id if self.state.start_id >= 0 else 0

        # 确定历史记录的结束ID
        end_id = (
            self.state.end_id
            if self.state.end_id >= 0
            else self.event_stream.get_latest_event_id()
        )

        # 从事件流中搜索并提取完整的历史记录
        self.state.history = list(
            self.event_stream.search_events(
                start_id=start_id,  # 起始事件ID
                end_id=end_id,  # 结束事件ID
                reverse=False,  # 按时间顺序排列
                filter=self.agent_history_filter,  # 应用事件过滤器
            )
        )

        # 3. 资源清理和取消订阅
        # ======================

        # 取消事件流订阅
        # 只有根父控制器才订阅事件流，委托控制器不需要取消订阅
        if not self.is_delegate:
            self.event_stream.unsubscribe(
                EventStreamSubscriber.AGENT_CONTROLLER,  # 订阅者类型
                self.id,  # 订阅者标识
            )

        # 标记控制器为已关闭状态
        self._closed = True

    def log(self, level: str, message: str, extra: dict | None = None) -> None:
        """
        记录日志消息 - 结构化日志记录

        这个方法提供了统一的日志记录接口，自动添加会话信息和控制器标识。
        支持结构化日志记录，便于监控和调试。

        技术特点：
        ==========

        1. 统一格式:
           - 自动添加控制器标识前缀
           - 统一的消息格式

        2. 结构化数据:
           - 自动添加session_id
           - 支持额外的结构化字段

        3. 灵活性:
           - 支持所有日志级别
           - 可扩展的额外字段

        Args:
            level (str): 日志级别（如'info', 'debug', 'error'）
            message (str): 要记录的消息
            extra (dict | None): 额外的结构化字段，默认包含session_id

        技术实现：
        ==========

        - 动态方法调用: 使用getattr动态调用对应级别的日志方法
        - 栈级别控制: 设置stacklevel=2确保正确的调用栈信息
        - 字段合并: 智能合并默认字段和用户提供的字段
        """

        # 添加控制器标识前缀，便于日志分析
        message = f'[Agent Controller {self.id}] {message}'

        # 初始化额外字段字典
        if extra is None:
            extra = {}

        # 合并默认字段和用户提供的字段
        # session_id是默认包含的重要字段
        extra_merged = {'session_id': self.id, **extra}

        # 动态调用对应级别的日志方法
        # stacklevel=2确保日志显示正确的调用位置
        getattr(logger, level)(message, extra=extra_merged, stacklevel=2)

    def update_state_before_step(self) -> None:
        """
        在Agent步骤执行前更新状态

        这个方法在每次Agent执行步骤之前被调用，负责更新迭代计数器。
        这些计数器用于跟踪Agent的执行进度和控制执行流程。

        技术说明：
        ==========

        1. 全局迭代计数:
           - iteration: 跟踪Agent的总执行步数
           - 用于流量控制和性能分析

        2. 本地迭代计数:
           - local_iteration: 跟踪当前会话的执行步数
           - 用于会话级别的控制和分析

        设计考虑：
        ==========

        - 原子性: 状态更新是原子的
        - 一致性: 确保计数器的一致性
        - 性能: 轻量级的操作，不影响性能
        """

        # 增加全局迭代计数器
        # 跟踪Agent的总执行步数
        self.state.iteration += 1

        # 增加本地迭代计数器
        # 跟踪当前会话的执行步数
        self.state.local_iteration += 1

    async def update_state_after_step(self) -> None:
        """
        在Agent步骤执行后更新状态

        这个方法在每次Agent执行步骤之后被调用，负责更新性能指标和成本信息。
        使用深拷贝确保指标数据不会被Agent的重置操作修改。

        技术实现：
        ==========

        1. 指标更新:
           - 从Agent的LLM中获取最新指标
           - 使用深拷贝避免数据被修改
           - 更新本地指标缓存

        2. 成本跟踪:
           - 跟踪LLM API调用成本
           - 支持预算控制和成本分析
           - 提供成本报告数据

        设计考虑：
        ==========

        - 数据隔离: 使用深拷贝确保数据独立性
        - 性能监控: 实时更新性能指标
        - 成本控制: 支持成本跟踪和预算管理
        """

        # 更新指标，特别是成本信息
        # 使用深拷贝避免被agent._reset()修改
        self.state.local_metrics = copy.deepcopy(self.agent.llm.metrics)

    async def _react_to_exception(
        self,
        e: Exception,
    ) -> None:
        """
        对异常做出反应 - 智能异常处理和状态管理

        这个方法负责处理Agent执行过程中遇到的各种异常，根据异常类型设置相应的状态
        并发送状态消息。它实现了分层的异常处理策略。

        技术架构：
        ==========

        1. 异常分类处理:
           - 认证错误: 处理API密钥等认证问题
           - 服务错误: 处理服务不可用、连接错误等
           - 预算错误: 处理预算超限问题
           - 内容政策错误: 处理内容违规问题
           - 速率限制错误: 处理API调用频率限制

        2. 状态管理:
           - 记录详细的错误信息
           - 设置相应的Agent状态
           - 触发状态回调通知

        3. 错误恢复:
           - 为不同错误类型提供恢复策略
           - 支持自动重试机制
           - 优雅降级处理

        Args:
            e (Exception): 要处理的异常对象

        异常处理策略：
        =============

        - AuthenticationError: 认证失败，需要检查API密钥
        - ServiceUnavailableError/APIConnectionError: 服务问题，可能需要重试
        - InternalServerError: 服务器内部错误，通常是临时问题
        - BadRequestError (ExceededBudget): 预算超限，需要增加预算或停止
        - ContentPolicyViolationError: 内容违规，需要修改输入内容
        - RateLimitError: 速率限制，设置特殊状态等待恢复

        设计特点：
        ==========

        - 精确分类: 根据异常类型进行精确处理
        - 状态一致性: 确保状态转换的一致性
        - 用户友好: 提供清晰的错误信息
        - 可恢复性: 支持从错误中恢复
        """

        # 在设置Agent状态之前存储错误原因
        self.state.last_error = f'{type(e).__name__}: {str(e)}'

        # 如果有状态回调，进行详细的错误分类处理
        if self.status_callback is not None:
            err_id = ''

            # 认证错误 - API密钥无效或过期
            if isinstance(e, AuthenticationError):
                err_id = 'STATUS$ERROR_LLM_AUTHENTICATION'
                self.state.last_error = err_id

            # 服务不可用错误 - 网络或服务问题
            elif isinstance(
                e,
                (
                    ServiceUnavailableError,  # 服务不可用
                    APIConnectionError,  # API连接错误
                    APIError,  # 通用API错误
                ),
            ):
                err_id = 'STATUS$ERROR_LLM_SERVICE_UNAVAILABLE'
                self.state.last_error = err_id

            # 服务器内部错误 - 通常是临时问题
            elif isinstance(e, InternalServerError):
                err_id = 'STATUS$ERROR_LLM_INTERNAL_SERVER_ERROR'
                self.state.last_error = err_id

            # 预算超限错误 - 需要增加预算或停止执行
            elif isinstance(e, BadRequestError) and 'ExceededBudget' in str(e):
                err_id = 'STATUS$ERROR_LLM_OUT_OF_CREDITS'
                self.state.last_error = err_id

            # 内容政策违规错误 - 输入内容不符合平台政策
            elif isinstance(e, ContentPolicyViolationError) or (
                isinstance(e, BadRequestError)
                and 'ContentPolicyViolationError' in str(e)
            ):
                err_id = 'STATUS$ERROR_LLM_CONTENT_POLICY_VIOLATION'
                self.state.last_error = err_id

            # 速率限制错误 - 特殊处理，设置RATE_LIMITED状态
            elif isinstance(e, RateLimitError):
                await self.set_agent_state_to(AgentState.RATE_LIMITED)
                return

            # 调用状态回调通知外部系统
            self.status_callback('error', err_id, self.state.last_error)

        # 在存储错误原因后设置Agent状态为ERROR
        await self.set_agent_state_to(AgentState.ERROR)

    def step(self) -> None:
        """
        启动Agent步骤执行 - 异步任务创建入口

        这是Agent步骤执行的公共入口点，它创建一个异步任务来执行实际的步骤逻辑。
        这种设计允许非阻塞的Agent执行，支持并发处理多个Agent。

        技术实现：
        ==========

        1. 异步任务创建:
           - 使用asyncio.create_task创建异步任务
           - 非阻塞执行，不等待任务完成
           - 支持并发的多Agent执行

        2. 异常处理委托:
           - 将异常处理委托给专门的方法
           - 确保所有异常都被正确捕获和处理
           - 提供统一的错误处理流程

        设计考虑：
        ==========

        - 非阻塞: 不阻塞调用线程
        - 并发支持: 支持多Agent并发执行
        - 异常安全: 确保异常被正确处理
        - 简洁接口: 提供简单的公共接口
        """
        # 创建异步任务执行带异常处理的步骤
        # 这允许非阻塞的Agent执行
        asyncio.create_task(self._step_with_exception_handling())

    async def _step_with_exception_handling(self) -> None:
        """
        带异常处理的Agent步骤执行 - 核心执行包装器

        这个方法是Agent步骤执行的核心包装器，负责捕获和处理执行过程中的所有异常。
        它实现了完善的异常处理策略，确保系统的稳定性和可靠性。

        技术架构：
        ==========

        1. 异常捕获:
           - 捕获所有可能的异常类型
           - 记录详细的错误信息和调用栈
           - 提供完整的错误上下文

        2. 异常分类:
           - 区分已知异常和未知异常
           - 对已知异常直接处理
           - 对未知异常进行包装和转换

        3. 错误报告:
           - 记录详细的错误日志
           - 生成用户友好的错误消息
           - 触发异常处理流程

        异常处理策略：
        =============

        已知异常类型（直接处理）:
        - Timeout: 超时错误
        - APIError: API调用错误
        - BadRequestError: 请求格式错误
        - NotFoundError: 资源未找到
        - InternalServerError: 服务器内部错误
        - AuthenticationError: 认证错误
        - RateLimitError: 速率限制错误
        - ContentPolicyViolationError: 内容政策违规
        - LLMContextWindowExceedError: 上下文窗口超限

        未知异常类型（包装处理）:
        - 包装为RuntimeError
        - 提供用户友好的错误消息
        - 记录警告日志便于调试

        设计特点：
        ==========

        - 全面覆盖: 捕获所有可能的异常
        - 分类处理: 根据异常类型采用不同策略
        - 用户友好: 提供清晰的错误信息
        - 调试支持: 记录详细的调试信息
        """
        try:
            # 执行实际的Agent步骤逻辑
            await self._step()
        except Exception as e:
            # 记录详细的错误信息，包括会话ID和完整的调用栈
            self.log(
                'error',
                f'Error while running the agent (session ID: {self.id}): {e}. '
                f'Traceback: {traceback.format_exc()}',
            )

            # 默认的错误报告，提供用户友好的消息
            reported = RuntimeError(
                f'There was an unexpected error while running the agent: {e.__class__.__name__}. '
                f'You can refresh the page or ask the agent to try again.'
            )

            # 检查是否为已知的异常类型
            # 对于已知异常，直接使用原始异常对象
            if (
                isinstance(e, Timeout)  # 超时错误
                or isinstance(e, APIError)  # API调用错误
                or isinstance(e, BadRequestError)  # 请求格式错误
                or isinstance(e, NotFoundError)  # 资源未找到
                or isinstance(e, InternalServerError)  # 服务器内部错误
                or isinstance(e, AuthenticationError)  # 认证错误
                or isinstance(e, RateLimitError)  # 速率限制错误
                or isinstance(e, ContentPolicyViolationError)  # 内容政策违规
                or isinstance(e, LLMContextWindowExceedError)  # 上下文窗口超限
            ):
                # 对于已知异常，直接使用原始异常对象
                reported = e
            else:
                # 对于未知异常类型，记录警告日志
                self.log(
                    'warning',
                    f'Unknown exception type while running the agent: {type(e).__name__}.',
                )

            # 调用异常处理方法
            await self._react_to_exception(reported)

    def should_step(self, event: Event) -> bool:
        """
        判断Agent是否应该基于事件执行步骤 - 智能事件过滤和决策

        这个方法实现了智能的事件过滤逻辑，决定Agent是否应该对特定事件做出响应。
        它是事件驱动架构中的关键决策点，确保Agent只在适当的时候执行。

        技术架构：
        ==========

        1. 委托检查:
           - 如果存在委托Agent，当前Agent暂停执行
           - 实现Agent层级管理和任务委托

        2. 动作事件处理:
           - 用户消息: 总是触发Agent执行
           - Agent消息: 根据状态决定是否执行
           - 委托动作: 触发委托流程
           - 压缩动作: 触发内存管理

        3. 观察事件处理:
           - 回忆观察: 触发Agent执行
           - 状态变化观察: 不触发执行
           - 空观察: 根据原因决定
           - 其他观察: 触发Agent执行

        Args:
            event (Event): 要评估的事件

        Returns:
            bool: True表示Agent应该执行步骤，False表示不应该

        决策逻辑：
        ==========

        委托优先级:
        - 如果有委托Agent正在执行，当前Agent暂停

        动作事件决策:
        - 用户消息 → 总是执行
        - Agent消息 → 根据状态决定
        - 委托动作 → 执行委托流程
        - 压缩动作 → 执行内存管理
        - 其他动作 → 不执行

        观察事件决策:
        - 回忆观察 → 执行
        - 状态变化观察 → 不执行
        - 空观察 → 不执行
        - 其他观察 → 执行

        设计特点：
        ==========

        - 智能过滤: 只响应相关事件
        - 委托支持: 支持多Agent协作
        - 状态感知: 根据Agent状态决策
        - 性能优化: 避免不必要的执行
        """

        # 委托优先级检查
        # 如果有委托Agent正在执行，当前Agent应该暂停
        if self.delegate is not None:
            return False

        # 处理动作事件
        if isinstance(event, Action):
            # 用户消息总是触发Agent执行
            if isinstance(event, MessageAction) and event.source == EventSource.USER:
                return True

            # Agent消息根据状态决定是否执行
            if (
                isinstance(event, MessageAction)
                and self.get_agent_state() != AgentState.AWAITING_USER_INPUT
            ):
                # TODO: 这个逻辑比较脆弱，但目前没有更好的检查方式
                return True

            # 委托动作触发委托流程
            if isinstance(event, AgentDelegateAction):
                return True

            # 压缩动作触发内存管理
            if isinstance(event, CondensationAction):
                return True

            # 其他动作不触发执行
            return False

        # 处理观察事件
        if isinstance(event, Observation):
            # 回忆观察（由RecallAction引起）触发执行
            if (
                isinstance(event, NullObservation)
                and event.cause is not None
                and event.cause
                > 0  # NullObservation的cause > 0表示RecallAction，而不是0（用户消息）
            ):
                return True

            # 状态变化观察和空观察不触发执行
            if isinstance(event, AgentStateChangedObservation) or isinstance(
                event, NullObservation
            ):
                return False

            # 其他观察事件触发Agent执行
            return True

        # 未知事件类型不触发执行
        return False

    def on_event(self, event: Event) -> None:
        """
        事件流回调处理器 - 事件驱动架构的核心入口

        这是事件流的回调方法，负责处理所有传入的事件。它是事件驱动架构的核心入口点，
        实现了事件的接收、过滤、路由和处理逻辑。

        技术架构：
        ==========

        1. 事件接收:
           - 从事件流接收所有类型的事件
           - 实现观察者模式的回调接口
           - 支持异步事件处理

        2. 事件过滤:
           - 使用should_step方法进行智能过滤
           - 只处理相关的事件
           - 提高系统性能和响应性

        3. 事件路由:
           - 根据事件类型路由到相应的处理逻辑
           - 支持不同类型事件的专门处理
           - 实现事件处理的分层架构

        4. 异步执行:
           - 触发异步的Agent步骤执行
           - 非阻塞的事件处理
           - 支持高并发的事件处理

        Args:
            event (Event): 要处理的传入事件

        处理流程：
        ==========

        1. 事件接收和日志记录
        2. 事件过滤（should_step检查）
        3. 异步步骤执行触发
        4. 错误处理和状态管理

        设计特点：
        ==========

        - 事件驱动: 实现完整的事件驱动架构
        - 智能过滤: 只处理相关事件
        - 异步处理: 非阻塞的事件处理
        - 错误安全: 完善的错误处理机制
        - 高性能: 优化的事件处理流程

        使用场景：
        ==========

        - 用户消息处理
        - 环境观察响应
        - Agent间通信
        - 状态变化通知
        - 系统事件处理
        """

        # 委托Agent事件转发逻辑
        # ======================

        # 如果有委托Agent且未完成或出错，将事件转发给它
        if self.delegate is not None:
            delegate_state = self.delegate.get_agent_state()
            if delegate_state not in (
                AgentState.FINISHED,  # 已完成
                AgentState.ERROR,  # 出错
                AgentState.REJECTED,  # 被拒绝
            ):
                # 将事件转发给委托Agent并跳过父Agent处理
                asyncio.get_event_loop().run_until_complete(
                    self.delegate._on_event(event)
                )
                return
            else:
                # 委托Agent已完成或出错，结束委托
                self.end_delegate()
                return

        # 只有在没有活跃委托时才继续父Agent处理
        asyncio.get_event_loop().run_until_complete(self._on_event(event))

    async def _on_event(self, event: Event) -> None:
        """
        内部事件处理方法 - 核心事件处理逻辑

        这是内部的事件处理方法，负责实际的事件处理逻辑，包括事件过滤、
        历史记录更新、事件分发和Agent步骤触发。

        技术实现：
        ==========

        1. 隐藏事件过滤:
           - 跳过标记为隐藏的事件
           - 减少不必要的处理开销
           - 保持历史记录的清洁

        2. 历史记录管理:
           - 使用事件过滤器决定是否记录
           - 维护Agent的执行历史
           - 支持状态恢复和分析

        3. 事件分发:
           - 根据事件类型分发到专门的处理器
           - 支持动作和观察的不同处理逻辑
           - 实现事件处理的多态性

        4. 步骤触发:
           - 使用should_step方法决定是否触发执行
           - 记录步骤触发的调试信息
           - 启动异步的Agent执行

        Args:
            event (Event): 要处理的事件对象

        处理流程：
        ==========

        1. 隐藏事件检查和过滤
        2. 事件历史记录更新
        3. 事件类型分发处理
        4. 步骤触发决策和执行

        设计特点：
        ==========

        - 高效过滤: 快速跳过不相关事件
        - 类型安全: 基于事件类型的分发
        - 异步处理: 非阻塞的事件处理
        - 调试友好: 详细的调试日志
        """

        # 1. 隐藏事件过滤
        # ================

        # 如果事件被标记为隐藏，跳过处理
        # 隐藏事件通常是内部管理事件，不需要Agent处理
        if hasattr(event, 'hidden') and event.hidden:
            return

        # 2. 历史记录管理
        # ================

        # 如果事件未被过滤器排除，将其添加到历史记录
        # 这维护了Agent的完整执行历史
        if self.agent_history_filter.include(event):
            self.state.history.append(event)

        # 3. 事件类型分发
        # ================

        # 根据事件类型分发到专门的处理器
        if isinstance(event, Action):
            # 处理动作事件（Agent或用户的动作）
            await self._handle_action(event)
        elif isinstance(event, Observation):
            # 处理观察事件（环境的响应）
            await self._handle_observation(event)

        # 4. 步骤触发决策
        # ================

        # 使用智能决策逻辑判断是否应该触发Agent步骤
        should_step = self.should_step(event)
        if should_step:
            # 记录步骤触发的调试信息
            self.log(
                'debug',
                f'Stepping agent after event: {type(event).__name__}',
                extra={'msg_type': 'STEPPING_AGENT'},
            )
            await self._step_with_exception_handling()
        elif isinstance(event, MessageAction) and event.source == EventSource.USER:
            # If we received a user message but aren't stepping, log why
            self.log(
                'warning',
                f'Not stepping agent after user message. Current state: {self.get_agent_state()}',
                extra={'msg_type': 'NOT_STEPPING_AFTER_USER_MESSAGE'},
            )

    async def _handle_action(self, action: Action) -> None:
        """Handles an Action from the agent or delegate."""
        if isinstance(action, ChangeAgentStateAction):
            await self.set_agent_state_to(action.agent_state)  # type: ignore
        elif isinstance(action, MessageAction):
            await self._handle_message_action(action)
        elif isinstance(action, AgentDelegateAction):
            await self.start_delegate(action)
            assert self.delegate is not None
            # Post a MessageAction with the task for the delegate
            if 'task' in action.inputs:
                self.event_stream.add_event(
                    MessageAction(content='TASK: ' + action.inputs['task']),
                    EventSource.USER,
                )
                await self.delegate.set_agent_state_to(AgentState.RUNNING)
            return

        elif isinstance(action, AgentFinishAction):
            self.state.outputs = action.outputs
            self.state.metrics.merge(self.state.local_metrics)
            await self.set_agent_state_to(AgentState.FINISHED)
        elif isinstance(action, AgentRejectAction):
            self.state.outputs = action.outputs
            self.state.metrics.merge(self.state.local_metrics)
            await self.set_agent_state_to(AgentState.REJECTED)

    async def _handle_observation(self, observation: Observation) -> None:
        """Handles observation from the event stream.

        Args:
            observation (observation): The observation to handle.
        """
        observation_to_print = copy.deepcopy(observation)
        if len(observation_to_print.content) > self.agent.llm.config.max_message_chars:
            observation_to_print.content = truncate_content(
                observation_to_print.content, self.agent.llm.config.max_message_chars
            )
        # Use info level if LOG_ALL_EVENTS is set
        log_level = 'info' if os.getenv('LOG_ALL_EVENTS') in ('true', '1') else 'debug'
        self.log(
            log_level, str(observation_to_print), extra={'msg_type': 'OBSERVATION'}
        )

        if observation.llm_metrics is not None:
            self.agent.llm.metrics.merge(observation.llm_metrics)

        # this happens for runnable actions and microagent actions
        if self._pending_action and self._pending_action.id == observation.cause:
            if self.state.agent_state == AgentState.AWAITING_USER_CONFIRMATION:
                return

            self._pending_action = None

            if self.state.agent_state == AgentState.USER_CONFIRMED:
                await self.set_agent_state_to(AgentState.RUNNING)
            if self.state.agent_state == AgentState.USER_REJECTED:
                await self.set_agent_state_to(AgentState.AWAITING_USER_INPUT)
            return
        elif isinstance(observation, ErrorObservation):
            if self.state.agent_state == AgentState.ERROR:
                self.state.metrics.merge(self.state.local_metrics)

    async def _handle_message_action(self, action: MessageAction) -> None:
        """Handles message actions from the event stream.

        Args:
            action (MessageAction): The message action to handle.
        """
        if action.source == EventSource.USER:
            # Use info level if LOG_ALL_EVENTS is set
            log_level = (
                'info' if os.getenv('LOG_ALL_EVENTS') in ('true', '1') else 'debug'
            )
            self.log(
                log_level,
                str(action),
                extra={'msg_type': 'ACTION', 'event_source': EventSource.USER},
            )
            # Extend max iterations when the user sends a message (only in non-headless mode)
            if self._initial_max_iterations is not None and not self.headless_mode:
                self.state.max_iterations = (
                    self.state.iteration + self._initial_max_iterations
                )
                if (
                    self.state.traffic_control_state == TrafficControlState.THROTTLING
                    or self.state.traffic_control_state == TrafficControlState.PAUSED
                ):
                    self.state.traffic_control_state = TrafficControlState.NORMAL
                self.log(
                    'debug',
                    f'Extended max iterations to {self.state.max_iterations} after user message',
                )
            # try to retrieve microagents relevant to the user message
            # set pending_action while we search for information

            # if this is the first user message for this agent, matters for the microagent info type
            first_user_message = self._first_user_message()
            is_first_user_message = (
                action.id == first_user_message.id if first_user_message else False
            )
            recall_type = (
                RecallType.WORKSPACE_CONTEXT
                if is_first_user_message
                else RecallType.KNOWLEDGE
            )

            recall_action = RecallAction(query=action.content, recall_type=recall_type)
            self._pending_action = recall_action
            # this is source=USER because the user message is the trigger for the microagent retrieval
            self.event_stream.add_event(recall_action, EventSource.USER)

            if self.get_agent_state() != AgentState.RUNNING:
                await self.set_agent_state_to(AgentState.RUNNING)

        elif action.source == EventSource.AGENT:
            # If the agent is waiting for a response, set the appropriate state
            if action.wait_for_response:
                await self.set_agent_state_to(AgentState.AWAITING_USER_INPUT)

    def _reset(self) -> None:
        """Resets the agent controller."""
        # Runnable actions need an Observation
        # make sure there is an Observation with the tool call metadata to be recognized by the agent
        # otherwise the pending action is found in history, but it's incomplete without an obs with tool result
        if self._pending_action and hasattr(self._pending_action, 'tool_call_metadata'):
            # find out if there already is an observation with the same tool call metadata
            found_observation = False
            for event in self.state.history:
                if (
                    isinstance(event, Observation)
                    and event.tool_call_metadata
                    == self._pending_action.tool_call_metadata
                ):
                    found_observation = True
                    break

            # make a new ErrorObservation with the tool call metadata
            if not found_observation:
                obs = ErrorObservation(
                    content=ERROR_ACTION_NOT_EXECUTED,
                    error_id=ERROR_ACTION_NOT_EXECUTED_ID,
                )
                obs.tool_call_metadata = self._pending_action.tool_call_metadata
                obs._cause = self._pending_action.id  # type: ignore[attr-defined]
                self.event_stream.add_event(obs, EventSource.AGENT)

        # NOTE: RecallActions don't need an ErrorObservation upon reset, as long as they have no tool calls

        # reset the pending action, this will be called when the agent is STOPPED or ERROR
        self._pending_action = None
        self.agent.reset()

    async def set_agent_state_to(self, new_state: AgentState) -> None:
        """Updates the agent's state and handles side effects. Can emit events to the event stream.

        Args:
            new_state (AgentState): The new state to set for the agent.
        """
        self.log(
            'info',
            f'Setting agent({self.agent.name}) state from {self.state.agent_state} to {new_state}',
        )

        if new_state == self.state.agent_state:
            return

        if new_state in (AgentState.STOPPED, AgentState.ERROR):
            # sync existing metrics BEFORE resetting the agent
            await self.update_state_after_step()
            self.state.metrics.merge(self.state.local_metrics)
            self._reset()
        elif (
            new_state == AgentState.RUNNING
            and self.state.agent_state == AgentState.PAUSED
            # TODO: do we really need both THROTTLING and PAUSED states, or can we clean up one of them completely?
            and self.state.traffic_control_state == TrafficControlState.THROTTLING
        ):
            # user intends to interrupt traffic control and let the task resume temporarily
            self.state.traffic_control_state = TrafficControlState.PAUSED
            # User has chosen to deliberately continue - lets double the max iterations
            if (
                self.state.iteration is not None
                and self.state.max_iterations is not None
                and self._initial_max_iterations is not None
                and not self.headless_mode
            ):
                if self.state.iteration >= self.state.max_iterations:
                    self.state.max_iterations += self._initial_max_iterations

            if (
                self.state.metrics.accumulated_cost is not None
                and self.max_budget_per_task is not None
                and self._initial_max_budget_per_task is not None
            ):
                if self.state.metrics.accumulated_cost >= self.max_budget_per_task:
                    self.max_budget_per_task += self._initial_max_budget_per_task
        elif self._pending_action is not None and (
            new_state in (AgentState.USER_CONFIRMED, AgentState.USER_REJECTED)
        ):
            if hasattr(self._pending_action, 'thought'):
                self._pending_action.thought = ''  # type: ignore[union-attr]
            if new_state == AgentState.USER_CONFIRMED:
                confirmation_state = ActionConfirmationStatus.CONFIRMED
            else:
                confirmation_state = ActionConfirmationStatus.REJECTED
            self._pending_action.confirmation_state = confirmation_state  # type: ignore[attr-defined]
            self._pending_action._id = None  # type: ignore[attr-defined]
            self.event_stream.add_event(self._pending_action, EventSource.AGENT)

        self.state.agent_state = new_state

        # Create observation with reason field if it's an error state
        reason = ''
        if new_state == AgentState.ERROR:
            reason = self.state.last_error

        self.event_stream.add_event(
            AgentStateChangedObservation('', self.state.agent_state, reason),
            EventSource.ENVIRONMENT,
        )

    def get_agent_state(self) -> AgentState:
        """Returns the current state of the agent.

        Returns:
            AgentState: The current state of the agent.
        """
        return self.state.agent_state

    async def start_delegate(self, action: AgentDelegateAction) -> None:
        """Start a delegate agent to handle a subtask.

        OpenHands is a multi-agentic system. A `task` is a conversation between
        OpenHands (the whole system) and the user, which might involve one or more inputs
        from the user. It starts with an initial input (typically a task statement) from
        the user, and ends with either an `AgentFinishAction` initiated by the agent, a
        stop initiated by the user, or an error.

        A `subtask` is a conversation between an agent and the user, or another agent. If a `task`
        is conducted by a single agent, then it's also a `subtask`. Otherwise, a `task` consists of
        multiple `subtasks`, each executed by one agent.

        Args:
            action (AgentDelegateAction): The action containing information about the delegate agent to start.
        """
        agent_cls: type[Agent] = Agent.get_cls(action.agent)
        agent_config = self.agent_configs.get(action.agent, self.agent.config)
        llm_config = self.agent_to_llm_config.get(action.agent, self.agent.llm.config)
        llm = LLM(config=llm_config, retry_listener=self._notify_on_llm_retry)
        delegate_agent = agent_cls(llm=llm, config=agent_config)
        state = State(
            session_id=self.id.removesuffix('-delegate'),
            inputs=action.inputs or {},
            local_iteration=0,
            iteration=self.state.iteration,
            max_iterations=self.state.max_iterations,
            delegate_level=self.state.delegate_level + 1,
            # global metrics should be shared between parent and child
            metrics=self.state.metrics,
            # start on top of the stream
            start_id=self.event_stream.get_latest_event_id() + 1,
        )
        self.log(
            'debug',
            f'start delegate, creating agent {delegate_agent.name} using LLM {llm}',
        )

        # Create the delegate with is_delegate=True so it does NOT subscribe directly
        self.delegate = AgentController(
            sid=self.id + '-delegate',
            agent=delegate_agent,
            event_stream=self.event_stream,
            max_iterations=self.state.max_iterations,
            max_budget_per_task=self.max_budget_per_task,
            agent_to_llm_config=self.agent_to_llm_config,
            agent_configs=self.agent_configs,
            initial_state=state,
            is_delegate=True,
            headless_mode=self.headless_mode,
        )

    def end_delegate(self) -> None:
        """Ends the currently active delegate (e.g., if it is finished or errored).

        so that this controller can resume normal operation.
        """
        if self.delegate is None:
            return

        delegate_state = self.delegate.get_agent_state()

        # update iteration that is shared across agents
        self.state.iteration = self.delegate.state.iteration

        # close the delegate controller before adding new events
        asyncio.get_event_loop().run_until_complete(self.delegate.close())

        if delegate_state in (AgentState.FINISHED, AgentState.REJECTED):
            # retrieve delegate result
            delegate_outputs = (
                self.delegate.state.outputs if self.delegate.state else {}
            )

            # prepare delegate result observation
            # TODO: replace this with AI-generated summary (#2395)
            formatted_output = ', '.join(
                f'{key}: {value}' for key, value in delegate_outputs.items()
            )
            content = (
                f'{self.delegate.agent.name} finishes task with {formatted_output}'
            )
        else:
            # delegate state is ERROR
            # emit AgentDelegateObservation with error content
            delegate_outputs = (
                self.delegate.state.outputs if self.delegate.state else {}
            )
            content = (
                f'{self.delegate.agent.name} encountered an error during execution.'
            )

        content = f'Delegated agent finished with result:\n\n{content}'

        # emit the delegate result observation
        obs = AgentDelegateObservation(outputs=delegate_outputs, content=content)

        # associate the delegate action with the initiating tool call
        for event in reversed(self.state.history):
            if isinstance(event, AgentDelegateAction):
                delegate_action = event
                obs.tool_call_metadata = delegate_action.tool_call_metadata
                break

        self.event_stream.add_event(obs, EventSource.AGENT)

        # unset delegate so parent can resume normal handling
        self.delegate = None

    async def _step(self) -> None:
        """Executes a single step of the parent or delegate agent. Detects stuck agents and limits on the number of iterations and the task budget."""
        if self.get_agent_state() != AgentState.RUNNING:
            self.log(
                'debug',
                f'Agent not stepping because state is {self.get_agent_state()} (not RUNNING)',
                extra={'msg_type': 'STEP_BLOCKED_STATE'},
            )
            return

        if self._pending_action:
            action_id = getattr(self._pending_action, 'id', 'unknown')
            action_type = type(self._pending_action).__name__
            self.log(
                'debug',
                f'Agent not stepping because of pending action: {action_type} (id={action_id})',
                extra={'msg_type': 'STEP_BLOCKED_PENDING_ACTION'},
            )
            return

        self.log(
            'debug',
            f'LEVEL {self.state.delegate_level} LOCAL STEP {self.state.local_iteration} GLOBAL STEP {self.state.iteration}',
            extra={'msg_type': 'STEP'},
        )

        stop_step = False
        if self.state.iteration >= self.state.max_iterations:
            stop_step = await self._handle_traffic_control(
                'iteration', self.state.iteration, self.state.max_iterations
            )
        if self.max_budget_per_task is not None:
            current_cost = self.state.metrics.accumulated_cost
            if current_cost > self.max_budget_per_task:
                stop_step = await self._handle_traffic_control(
                    'budget', current_cost, self.max_budget_per_task
                )
        if stop_step:
            logger.warning('Stopping agent due to traffic control')
            return

        if self._is_stuck():
            await self._react_to_exception(
                AgentStuckInLoopError('Agent got stuck in a loop')
            )
            return

        self.update_state_before_step()
        action: Action = NullAction()

        if self._replay_manager.should_replay():
            # in replay mode, we don't let the agent to proceed
            # instead, we replay the action from the replay trajectory
            action = self._replay_manager.step()
        else:
            try:
                action = self.agent.step(self.state)
                if action is None:
                    raise LLMNoActionError('No action was returned')
                action._source = EventSource.AGENT  # type: ignore [attr-defined]
            except (
                LLMMalformedActionError,
                LLMNoActionError,
                LLMResponseError,
                FunctionCallValidationError,
                FunctionCallNotExistsError,
            ) as e:
                self.event_stream.add_event(
                    ErrorObservation(
                        content=str(e),
                    ),
                    EventSource.AGENT,
                )
                return
            except (ContextWindowExceededError, BadRequestError, OpenAIError) as e:
                # FIXME: this is a hack until a litellm fix is confirmed
                # Check if this is a nested context window error
                # We have to rely on string-matching because LiteLLM doesn't consistently
                # wrap the failure in a ContextWindowExceededError
                error_str = str(e).lower()
                if (
                    'contextwindowexceedederror' in error_str
                    or 'prompt is too long' in error_str
                    or 'input length and `max_tokens` exceed context limit' in error_str
                    or 'please reduce the length of either one'
                    in error_str  # For OpenRouter context window errors
                    or isinstance(e, ContextWindowExceededError)
                ):
                    if self.agent.config.enable_history_truncation:
                        self._handle_long_context_error()
                        return
                    else:
                        raise LLMContextWindowExceedError()
                else:
                    raise e

        if action.runnable:
            if self.state.confirmation_mode and (
                type(action) is CmdRunAction or type(action) is IPythonRunCellAction
            ):
                action.confirmation_state = (
                    ActionConfirmationStatus.AWAITING_CONFIRMATION
                )
            self._pending_action = action

        if not isinstance(action, NullAction):
            if (
                hasattr(action, 'confirmation_state')
                and action.confirmation_state
                == ActionConfirmationStatus.AWAITING_CONFIRMATION
            ):
                await self.set_agent_state_to(AgentState.AWAITING_USER_CONFIRMATION)

            # Create and log metrics for frontend display
            self._prepare_metrics_for_frontend(action)

            self.event_stream.add_event(action, action._source)  # type: ignore [attr-defined]

        await self.update_state_after_step()

        log_level = 'info' if LOG_ALL_EVENTS else 'debug'
        self.log(log_level, str(action), extra={'msg_type': 'ACTION'})

    def _notify_on_llm_retry(self, retries: int, max: int) -> None:
        if self.status_callback is not None:
            msg_id = 'STATUS$LLM_RETRY'
            self.status_callback(
                'info', msg_id, f'Retrying LLM request, {retries} / {max}'
            )

    async def _handle_traffic_control(
        self, limit_type: str, current_value: float, max_value: float
    ) -> bool:
        """Handles agent state after hitting the traffic control limit.

        Args:
            limit_type (str): The type of limit that was hit.
            current_value (float): The current value of the limit.
            max_value (float): The maximum value of the limit.
        """
        stop_step = False
        if self.state.traffic_control_state == TrafficControlState.PAUSED:
            self.log(
                'debug', 'Hitting traffic control, temporarily resume upon user request'
            )
            self.state.traffic_control_state = TrafficControlState.NORMAL
        else:
            self.state.traffic_control_state = TrafficControlState.THROTTLING
            # Format values as integers for iterations, keep decimals for budget
            if limit_type == 'iteration':
                current_str = str(int(current_value))
                max_str = str(int(max_value))
            else:
                current_str = f'{current_value:.2f}'
                max_str = f'{max_value:.2f}'

            if self.headless_mode:
                e = RuntimeError(
                    f'Agent reached maximum {limit_type} in headless mode. '
                    f'Current {limit_type}: {current_str}, max {limit_type}: {max_str}'
                )
                await self._react_to_exception(e)
            else:
                e = RuntimeError(
                    f'Agent reached maximum {limit_type}. '
                    f'Current {limit_type}: {current_str}, max {limit_type}: {max_str}. '
                )
                # FIXME: this isn't really an exception--we should have a different path
                await self._react_to_exception(e)
            stop_step = True
        return stop_step

    @property
    def _pending_action(self) -> Action | None:
        """Get the current pending action with time tracking.

        Returns:
            Action | None: The current pending action, or None if there isn't one.
        """
        if self._pending_action_info is None:
            return None

        action, timestamp = self._pending_action_info
        current_time = time.time()
        elapsed_time = current_time - timestamp

        # Log if the pending action has been active for a long time (but don't clear it)
        if elapsed_time > 60.0:  # 1 minute - just for logging purposes
            action_id = getattr(action, 'id', 'unknown')
            action_type = type(action).__name__
            self.log(
                'warning',
                f'Pending action active for {elapsed_time:.2f}s: {action_type} (id={action_id})',
                extra={'msg_type': 'PENDING_ACTION_TIMEOUT'},
            )

        return action

    @_pending_action.setter
    def _pending_action(self, action: Action | None) -> None:
        """Set or clear the pending action with timestamp and logging.

        Args:
            action: The action to set as pending, or None to clear.
        """
        if action is None:
            if self._pending_action_info is not None:
                prev_action, timestamp = self._pending_action_info
                action_id = getattr(prev_action, 'id', 'unknown')
                action_type = type(prev_action).__name__
                elapsed_time = time.time() - timestamp
                self.log(
                    'debug',
                    f'Cleared pending action after {elapsed_time:.2f}s: {action_type} (id={action_id})',
                    extra={'msg_type': 'PENDING_ACTION_CLEARED'},
                )
            self._pending_action_info = None
        else:
            action_id = getattr(action, 'id', 'unknown')
            action_type = type(action).__name__
            self.log(
                'debug',
                f'Set pending action: {action_type} (id={action_id})',
                extra={'msg_type': 'PENDING_ACTION_SET'},
            )
            self._pending_action_info = (action, time.time())

    def get_state(self) -> State:
        """Returns the current running state object.

        Returns:
            State: The current state object.
        """
        return self.state

    def set_initial_state(
        self,
        state: State | None,
        max_iterations: int,
        confirmation_mode: bool = False,
    ) -> None:
        """Sets the initial state for the agent, either from the previous session, or from a parent agent, or by creating a new one.

        Args:
            state: The state to initialize with, or None to create a new state.
            max_iterations: The maximum number of iterations allowed for the task.
            confirmation_mode: Whether to enable confirmation mode.
        """
        # state can come from:
        # - the previous session, in which case it has history
        # - from a parent agent, in which case it has no history
        # - None / a new state

        # If state is None, we create a brand new state and still load the event stream so we can restore the history
        if state is None:
            self.state = State(
                session_id=self.id.removesuffix('-delegate'),
                inputs={},
                max_iterations=max_iterations,
                confirmation_mode=confirmation_mode,
            )
            self.state.start_id = 0

            self.log(
                'info',
                f'AgentController {self.id} - created new state. start_id: {self.state.start_id}',
            )
        else:
            self.state = state

            if self.state.start_id <= -1:
                self.state.start_id = 0

            self.log(
                'info',
                f'AgentController {self.id} initializing history from event {self.state.start_id}',
            )

        # Always load from the event stream to avoid losing history
        self._init_history()

    def get_trajectory(self, include_screenshots: bool = False) -> list[dict]:
        # state history could be partially hidden/truncated before controller is closed
        assert self._closed
        return [
            event_to_trajectory(event, include_screenshots)
            for event in self.state.history
        ]

    def _init_history(self) -> None:
        """Initializes the agent's history from the event stream.

        The history is a list of events that:
        - Excludes events of types listed in self.filter_out
        - Excludes events with hidden=True attribute
        - For delegate events (between AgentDelegateAction and AgentDelegateObservation):
            - Excludes all events between the action and observation
            - Includes the delegate action and observation themselves
        """
        # define range of events to fetch
        # delegates start with a start_id and initially won't find any events
        # otherwise we're restoring a previous session
        start_id = self.state.start_id if self.state.start_id >= 0 else 0
        end_id = (
            self.state.end_id
            if self.state.end_id >= 0
            else self.event_stream.get_latest_event_id()
        )

        # sanity check
        if start_id > end_id + 1:
            self.log(
                'warning',
                f'start_id {start_id} is greater than end_id + 1 ({end_id + 1}). History will be empty.',
            )
            self.state.history = []
            return

        events: list[Event] = []

        # Get rest of history
        events_to_add = list(
            self.event_stream.search_events(
                start_id=start_id,
                end_id=end_id,
                reverse=False,
                filter=self.agent_history_filter,
            )
        )
        events.extend(events_to_add)

        # Find all delegate action/observation pairs
        delegate_ranges: list[tuple[int, int]] = []
        delegate_action_ids: list[int] = []  # stack of unmatched delegate action IDs

        for event in events:
            if isinstance(event, AgentDelegateAction):
                delegate_action_ids.append(event.id)
                # Note: we can get agent=event.agent and task=event.inputs.get('task','')
                # if we need to track these in the future

            elif isinstance(event, AgentDelegateObservation):
                # Match with most recent unmatched delegate action
                if not delegate_action_ids:
                    self.log(
                        'warning',
                        f'Found AgentDelegateObservation without matching action at id={event.id}',
                    )
                    continue

                action_id = delegate_action_ids.pop()
                delegate_ranges.append((action_id, event.id))

        # Filter out events between delegate action/observation pairs
        if delegate_ranges:
            filtered_events: list[Event] = []
            current_idx = 0

            for start_id, end_id in sorted(delegate_ranges):
                # Add events before delegate range
                filtered_events.extend(
                    event for event in events[current_idx:] if event.id < start_id
                )

                # Add delegate action and observation
                filtered_events.extend(
                    event for event in events if event.id in (start_id, end_id)
                )

                # Update index to after delegate range
                current_idx = next(
                    (i for i, e in enumerate(events) if e.id > end_id), len(events)
                )

            # Add any remaining events after last delegate range
            filtered_events.extend(events[current_idx:])

            self.state.history = filtered_events
        else:
            self.state.history = events

        # make sure history is in sync
        self.state.start_id = start_id

    def _handle_long_context_error(self) -> None:
        # When context window is exceeded, keep roughly half of agent interactions
        current_view = View.from_events(self.state.history)
        kept_events = self._apply_conversation_window(current_view.events)
        kept_event_ids = {e.id for e in kept_events}

        self.log(
            'info',
            f'Context window exceeded. Keeping events with IDs: {kept_event_ids}',
        )

        # The events to forget are those that are not in the kept set
        forgotten_event_ids = {e.id for e in self.state.history} - kept_event_ids

        if len(kept_event_ids) == 0:
            self.log(
                'warning',
                'No events kept after applying conversation window. This should not happen.',
            )

        # verify that the first event id in kept_event_ids is the same as the start_id
        if len(kept_event_ids) > 0 and self.state.history[0].id not in kept_event_ids:
            self.log(
                'warning',
                f'First event after applying conversation window was not kept: {self.state.history[0].id} not in {kept_event_ids}',
            )

        # Add an error event to trigger another step by the agent
        self.event_stream.add_event(
            CondensationAction(
                forgotten_events_start_id=min(forgotten_event_ids)
                if forgotten_event_ids
                else 0,
                forgotten_events_end_id=max(forgotten_event_ids)
                if forgotten_event_ids
                else 0,
            ),
            EventSource.AGENT,
        )

    def _apply_conversation_window(self, history: list[Event]) -> list[Event]:
        """Cuts history roughly in half when context window is exceeded.

        It preserves action-observation pairs and ensures that the system message,
        the first user message, and its associated recall observation are always included
        at the beginning of the context window.

        The algorithm:
        1. Identify essential initial events: System Message, First User Message, Recall Observation.
        2. Determine the slice of recent events to potentially keep.
        3. Validate the start of the recent slice for dangling observations.
        4. Combine essential events and validated recent events, ensuring essentials come first.

        Args:
            events: List of events to filter

        Returns:
            Filtered list of events keeping newest half while preserving pairs and essential initial events.
        """
        # Handle empty history
        if not history:
            return []
        # 1. Identify essential initial events
        system_message: SystemMessageAction | None = None
        first_user_msg: MessageAction | None = None
        recall_action: RecallAction | None = None
        recall_observation: Observation | None = None

        # Find System Message (should be the first event, if it exists)
        system_message = next(
            (e for e in history if isinstance(e, SystemMessageAction)), None
        )
        assert (
            system_message is None
            or isinstance(system_message, SystemMessageAction)
            and system_message.id == history[0].id
        )

        # Find First User Message in the history, which MUST exist
        first_user_msg = self._first_user_message(history)
        if first_user_msg is None:
            # If not found in history, try the event stream
            first_user_msg = self._first_user_message()
            if first_user_msg is None:
                raise RuntimeError('No first user message found in the event stream.')
            self.log(
                'warning',
                'First user message not found in history. Using cached version from event stream.',
            )

        # Find the first user message index in the history
        first_user_msg_index = -1
        for i, event in enumerate(history):
            if isinstance(event, MessageAction) and event.source == EventSource.USER:
                first_user_msg_index = i
                break

        # Find Recall Action and Observation related to the First User Message
        # Look for RecallAction after the first user message
        for i in range(first_user_msg_index + 1, len(history)):
            event = history[i]
            if (
                isinstance(event, RecallAction)
                and event.query == first_user_msg.content
            ):
                # Found RecallAction, now look for its Observation
                recall_action = event
                for j in range(i + 1, len(history)):
                    obs_event = history[j]
                    # Check for Observation caused by this RecallAction
                    if (
                        isinstance(obs_event, Observation)
                        and obs_event.cause == recall_action.id
                    ):
                        recall_observation = obs_event
                        break  # Found the observation, stop inner loop
                break  # Found the recall action (and maybe obs), stop outer loop

        essential_events: list[Event] = []
        if system_message:
            essential_events.append(system_message)
        # Only include first user message if history is not empty
        if history:
            essential_events.append(first_user_msg)
            # Include recall action and observation if both exist
            if recall_action and recall_observation:
                essential_events.append(recall_action)
                essential_events.append(recall_observation)
            # Include recall action without observation for backward compatibility
            elif recall_action:
                essential_events.append(recall_action)

        # 2. Determine the slice of recent events to potentially keep
        num_non_essential_events = len(history) - len(essential_events)
        # Keep roughly half of the non-essential events, minimum 1
        num_recent_to_keep = max(1, num_non_essential_events // 2)

        # Calculate the starting index for the recent slice
        slice_start_index = len(history) - num_recent_to_keep
        slice_start_index = max(0, slice_start_index)  # Ensure index is not negative
        recent_events_slice = history[slice_start_index:]

        # 3. Validate the start of the recent slice for dangling observations
        # IMPORTANT: Most observations in history are tool call results, which cannot be without their action, or we get an LLM API error
        first_valid_event_index = 0
        for i, event in enumerate(recent_events_slice):
            if isinstance(event, Observation):
                first_valid_event_index += 1
            else:
                break
        # If all events in the slice are dangling observations, we need to keep at least one
        if first_valid_event_index == len(recent_events_slice):
            self.log(
                'warning',
                'All recent events are dangling observations, which we truncate. This means the agent has only the essential first events. This should not happen.',
            )

        # Adjust the recent_events_slice if dangling observations were found at the start
        if first_valid_event_index < len(recent_events_slice):
            validated_recent_events = recent_events_slice[first_valid_event_index:]
            if first_valid_event_index > 0:
                self.log(
                    'debug',
                    f'Removed {first_valid_event_index} dangling observation(s) from the start of recent event slice.',
                )
        else:
            validated_recent_events = []

        # 4. Combine essential events and validated recent events
        events_to_keep: list[Event] = essential_events + validated_recent_events
        self.log('debug', f'History truncated. Kept {len(events_to_keep)} events.')

        return events_to_keep

    def _is_stuck(self) -> bool:
        """Checks if the agent or its delegate is stuck in a loop.

        Returns:
            bool: True if the agent is stuck, False otherwise.
        """
        # check if delegate stuck
        if self.delegate and self.delegate._is_stuck():
            return True

        return self._stuck_detector.is_stuck(self.headless_mode)

    def _prepare_metrics_for_frontend(self, action: Action) -> None:
        """Create a minimal metrics object for frontend display and log it.

        To avoid performance issues with long conversations, we only keep:
        - accumulated_cost: The current total cost
        - accumulated_token_usage: Accumulated token statistics across all API calls

        This includes metrics from both the agent's LLM and the condenser's LLM if it exists.

        Args:
            action: The action to attach metrics to
        """
        # Get metrics from agent LLM
        agent_metrics = self.agent.llm.metrics

        # Get metrics from condenser LLM if it exists
        condenser_metrics: TokenUsage | None = None
        if hasattr(self.agent, 'condenser') and hasattr(self.agent.condenser, 'llm'):
            condenser_metrics = self.agent.condenser.llm.metrics

        # Create a new minimal metrics object with just what the frontend needs
        metrics = Metrics(model_name=agent_metrics.model_name)

        # Set accumulated cost (sum of agent and condenser costs)
        metrics.accumulated_cost = agent_metrics.accumulated_cost
        if condenser_metrics:
            metrics.accumulated_cost += condenser_metrics.accumulated_cost

        # Set accumulated token usage (sum of agent and condenser token usage)
        # Use a deep copy to ensure we don't modify the original object
        metrics._accumulated_token_usage = (
            agent_metrics.accumulated_token_usage.model_copy(deep=True)
        )
        if condenser_metrics:
            metrics._accumulated_token_usage = (
                metrics._accumulated_token_usage
                + condenser_metrics.accumulated_token_usage
            )

        action.llm_metrics = metrics

        # Log the metrics information for debugging
        # Get the latest usage directly from the agent's metrics
        latest_usage = None
        if self.agent.llm.metrics.token_usages:
            latest_usage = self.agent.llm.metrics.token_usages[-1]

        accumulated_usage = self.agent.llm.metrics.accumulated_token_usage
        self.log(
            'debug',
            f'Action metrics - accumulated_cost: {metrics.accumulated_cost}, '
            f'latest tokens (prompt/completion/cache_read/cache_write): '
            f'{latest_usage.prompt_tokens if latest_usage else 0}/'
            f'{latest_usage.completion_tokens if latest_usage else 0}/'
            f'{latest_usage.cache_read_tokens if latest_usage else 0}/'
            f'{latest_usage.cache_write_tokens if latest_usage else 0}, '
            f'accumulated tokens (prompt/completion): '
            f'{accumulated_usage.prompt_tokens}/'
            f'{accumulated_usage.completion_tokens}',
            extra={'msg_type': 'METRICS'},
        )

    def __repr__(self) -> str:
        pending_action_info = '<none>'
        if (
            hasattr(self, '_pending_action_info')
            and self._pending_action_info is not None
        ):
            action, timestamp = self._pending_action_info
            action_id = getattr(action, 'id', 'unknown')
            action_type = type(action).__name__
            elapsed_time = time.time() - timestamp
            pending_action_info = (
                f'{action_type}(id={action_id}, elapsed={elapsed_time:.2f}s)'
            )

        return (
            f'AgentController(id={getattr(self, "id", "<uninitialized>")}, '
            f'agent={getattr(self, "agent", "<uninitialized>")!r}, '
            f'event_stream={getattr(self, "event_stream", "<uninitialized>")!r}, '
            f'state={getattr(self, "state", "<uninitialized>")!r}, '
            f'delegate={getattr(self, "delegate", "<uninitialized>")!r}, '
            f'_pending_action={pending_action_info})'
        )

    def _is_awaiting_observation(self) -> bool:
        events = self.event_stream.get_events(reverse=True)
        for event in events:
            if isinstance(event, AgentStateChangedObservation):
                result = event.agent_state == AgentState.RUNNING
                return result
        return False

    def _first_user_message(
        self, events: list[Event] | None = None
    ) -> MessageAction | None:
        """Get the first user message for this agent.

        For regular agents, this is the first user message from the beginning (start_id=0).
        For delegate agents, this is the first user message after the delegate's start_id.

        Args:
            events: Optional list of events to search through. If None, uses the event stream.

        Returns:
            MessageAction | None: The first user message, or None if no user message found
        """
        # If events list is provided, search through it
        if events is not None:
            return next(
                (
                    e
                    for e in events
                    if isinstance(e, MessageAction) and e.source == EventSource.USER
                ),
                None,
            )

        # Otherwise, use the original event stream logic with caching
        # Return cached message if any
        if self._cached_first_user_message is not None:
            return self._cached_first_user_message

        # Find the first user message
        self._cached_first_user_message = next(
            (
                e
                for e in self.event_stream.get_events(
                    start_id=self.state.start_id,
                )
                if isinstance(e, MessageAction) and e.source == EventSource.USER
            ),
            None,
        )
        return self._cached_first_user_message
