"""
OpenHands Agent 抽象基类模块

技术栈:
- Python 3.12+ (核心语言)
- ABC (Abstract Base Classes) - Python抽象基类
- LiteLLM - 多LLM提供商统一接口
- Pydantic - 数据验证和序列化
- 类型注解 - TYPE_CHECKING, 前向引用

架构说明:
这个模块定义了OpenHands中所有智能代理的抽象基类。Agent是整个系统的核心抽象，
定义了代理的基本接口和生命周期管理。所有具体的代理实现都必须继承这个基类。

设计模式:
- 抽象工厂模式: 通过注册机制创建不同类型的Agent
- 模板方法模式: 定义了Agent的基本执行流程
- 策略模式: 不同的Agent实现不同的执行策略
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

# 类型检查时导入，避免循环导入
if TYPE_CHECKING:
    from openhands.controller.state.state import State
    from openhands.core.config import AgentConfig
    from openhands.events.action import Action
    from openhands.events.action.message import SystemMessageAction
    from openhands.utils.prompt import PromptManager

# LiteLLM - 统一的LLM接口，支持OpenAI、Anthropic、Google等多种LLM
from litellm import ChatCompletionToolParam

# 核心异常类
from openhands.core.exceptions import (
    AgentAlreadyRegisteredError,  # Agent已注册异常
    AgentNotRegisteredError,  # Agent未注册异常
)
from openhands.core.logger import openhands_logger as logger
from openhands.events.event import EventSource  # 事件源枚举
from openhands.llm.llm import LLM  # LLM统一接口
from openhands.runtime.plugins import PluginRequirement  # 插件需求定义


class Agent(ABC):
    """
    OpenHands智能代理抽象基类

    这个抽象基类定义了所有智能代理的通用接口，用于执行特定指令并支持人机交互。
    它跟踪执行状态并维护交互历史记录。

    核心职责:
    1. 定义Agent的基本接口和生命周期
    2. 管理LLM交互和工具调用
    3. 处理系统消息和prompt管理
    4. 提供Agent注册和发现机制
    5. 支持MCP (Model Context Protocol) 工具集成

    设计原则:
    - 抽象与具体分离: 基类定义接口，子类实现具体逻辑
    - 可扩展性: 通过注册机制支持新的Agent类型
    - 工具化: 通过工具系统提供各种能力
    - 状态管理: 明确的执行状态和生命周期
    """

    # 标记是否为已弃用的Agent类型
    DEPRECATED = False

    # 类级别的Agent注册表，用于Agent工厂模式
    # 键: Agent名称, 值: Agent类型
    _registry: dict[str, type['Agent']] = {}

    # 沙箱插件需求列表，定义Agent运行所需的插件
    # 例如: JupyterRequirement, AgentSkillsRequirement等
    sandbox_plugins: list[PluginRequirement] = []

    def __init__(
        self,
        llm: LLM,  # 语言模型实例，用于生成响应
        config: 'AgentConfig',  # Agent配置，包含各种参数设置
    ):
        """
        初始化Agent实例

        Args:
            llm: 语言模型实例，负责生成Agent的响应
            config: Agent配置对象，包含运行参数和设置
        """
        self.llm = llm  # LLM实例，核心推理引擎
        self.config = config  # 配置对象，包含各种设置
        self._complete = False  # 任务完成标志
        self._prompt_manager: 'PromptManager' | None = None  # Prompt管理器，延迟初始化

        # MCP (Model Context Protocol) 工具字典
        # 键: 工具名称, 值: 工具参数定义
        self.mcp_tools: dict[str, ChatCompletionToolParam] = {}

        # 可用工具列表，包含所有Agent可以调用的工具
        self.tools: list = []

    @property
    def prompt_manager(self) -> 'PromptManager':
        """
        获取Prompt管理器实例

        Prompt管理器负责加载和管理Agent的系统提示词，包括:
        - 系统消息模板
        - 工具使用说明
        - 微代理集成
        - 多语言支持

        Returns:
            PromptManager: Prompt管理器实例

        Raises:
            ValueError: 如果Prompt管理器未初始化
        """
        if self._prompt_manager is None:
            raise ValueError(f'Prompt manager not initialized for agent {self.name}')
        return self._prompt_manager

    def get_system_message(self) -> 'SystemMessageAction | None':
        """
        生成系统消息Action

        系统消息是Agent与LLM交互的第一条消息，包含:
        1. Agent的角色定义和能力描述
        2. 可用工具列表和使用说明
        3. 任务执行指导原则
        4. 输出格式要求

        这个消息会被添加到事件流的开头，为整个对话设定上下文。

        技术实现:
        - 使用Jinja2模板引擎渲染系统消息
        - 集成MCP工具定义
        - 支持微代理扩展

        Returns:
            SystemMessageAction: 包含系统消息内容和工具定义的Action
            None: 如果生成系统消息时发生错误
        """
        # 延迟导入避免循环依赖
        from openhands.events.action.message import SystemMessageAction

        try:
            # 检查Prompt管理器是否已初始化
            if not self.prompt_manager:
                logger.warning(
                    f'[{self.name}] Prompt manager not initialized before getting system message'
                )
                return None

            # 从Prompt管理器获取渲染后的系统消息
            system_message = self.prompt_manager.get_system_message()

            # 获取Agent可用的工具列表
            # 这些工具定义了Agent可以执行的操作
            tools = getattr(self, 'tools', None)

            # 创建系统消息Action
            system_message_action = SystemMessageAction(
                content=system_message,  # 系统消息内容
                tools=tools,  # 可用工具列表
                agent_class=self.name,  # Agent类名，用于标识
            )

            # 设置事件源为Agent
            system_message_action._source = EventSource.AGENT  # type: ignore

            return system_message_action

        except Exception as e:
            # 记录错误但不抛出异常，保证系统稳定性
            logger.warning(f'[{self.name}] Failed to generate system message: {e}')
            return None

    @property
    def complete(self) -> bool:
        """
        检查当前指令执行是否完成

        这个属性用于判断Agent是否已经完成了当前分配的任务。
        Controller会根据这个状态决定是否继续执行或结束会话。

        Returns:
            bool: True表示执行完成，False表示仍在执行中
        """
        return self._complete

    @abstractmethod
    def step(self, state: 'State') -> 'Action':
        """
        执行一步操作 (抽象方法)

        这是Agent的核心方法，定义了Agent的执行逻辑。每次调用都会：
        1. 分析当前状态 (State)
        2. 决定下一步要执行的操作
        3. 返回相应的Action

        执行流程:
        State → Agent.step() → Action → Runtime.execute() → Observation → State.update()

        子类必须实现这个方法来定义具体的执行策略，例如:
        - CodeActAgent: 基于代码执行的策略
        - BrowsingAgent: 基于网页浏览的策略
        - ReadonlyAgent: 只读模式的策略

        Args:
            state: 当前系统状态，包含历史事件、环境信息等

        Returns:
            Action: 下一步要执行的操作，如命令执行、消息发送等
        """
        pass

    def reset(self) -> None:
        """
        重置Agent的执行状态

        这个方法用于清理Agent的状态，为重新开始任务做准备。
        通常在以下情况调用:
        1. 开始新的任务会话
        2. 从错误状态恢复
        3. 清理资源准备销毁

        重置操作包括:
        - 清除完成标志
        - 重置LLM状态 (清除对话历史)
        - 清理内部缓存 (TODO: 实现历史清理)
        """
        # TODO: 实现完整的历史清理逻辑
        self._complete = False

        # 重置LLM状态，清除对话历史和缓存
        if self.llm:
            self.llm.reset()

    @property
    def name(self) -> str:
        """
        获取Agent的名称

        Agent名称用于:
        1. 日志记录和调试
        2. 配置管理和路由
        3. 注册表查找
        4. 用户界面显示

        Returns:
            str: Agent的类名，如 'CodeActAgent', 'BrowsingAgent'
        """
        return self.__class__.__name__

    @classmethod
    def register(cls, name: str, agent_cls: type['Agent']) -> None:
        """
        在注册表中注册Agent类 (工厂模式)

        这个方法实现了Agent的注册机制，支持动态添加新的Agent类型。
        注册后的Agent可以通过名称进行实例化，实现了插件化的架构。

        使用场景:
        1. 系统启动时自动注册内置Agent
        2. 插件系统动态注册新Agent
        3. 测试环境注册Mock Agent

        技术实现:
        - 使用类级别字典作为注册表
        - 支持重复注册检查
        - 线程安全的注册机制

        Args:
            name: Agent的注册名称，用于后续查找和实例化
            agent_cls: 要注册的Agent类，必须继承自Agent基类

        Raises:
            AgentAlreadyRegisteredError: 如果名称已经被注册
        """
        if name in cls._registry:
            raise AgentAlreadyRegisteredError(name)
        cls._registry[name] = agent_cls

    @classmethod
    def get_cls(cls, name: str) -> type['Agent']:
        """
        从注册表中获取Agent类

        这个方法用于根据名称查找已注册的Agent类，是Agent工厂的核心方法。
        通常在以下场景使用:
        1. 根据配置文件创建Agent实例
        2. 用户界面选择Agent类型
        3. API接口动态创建Agent

        Args:
            name: 要查找的Agent名称

        Returns:
            type['Agent']: 对应的Agent类

        Raises:
            AgentNotRegisteredError: 如果指定名称的Agent未注册
        """
        if name not in cls._registry:
            raise AgentNotRegisteredError(name)
        return cls._registry[name]

    @classmethod
    def list_agents(cls) -> list[str]:
        """
        获取所有已注册的Agent名称列表

        这个方法用于发现系统中可用的Agent类型，常用于:
        1. 用户界面显示可选Agent列表
        2. API文档生成
        3. 系统诊断和调试
        4. 配置验证

        Returns:
            list[str]: 所有已注册Agent的名称列表

        Raises:
            AgentNotRegisteredError: 如果没有任何Agent被注册
        """
        if not bool(cls._registry):
            raise AgentNotRegisteredError()
        return list(cls._registry.keys())

    def set_mcp_tools(self, mcp_tools: list[dict]) -> None:
        """
        设置Agent的MCP工具集

        MCP (Model Context Protocol) 是一个标准化的工具调用协议，
        允许Agent与外部系统和服务进行交互。这个方法用于动态配置
        Agent可以使用的工具集合。

        工具类型包括:
        1. 文件操作工具 (读取、写入、编辑)
        2. 命令执行工具 (bash, python)
        3. 网络访问工具 (HTTP请求、浏览器)
        4. 数据处理工具 (JSON、CSV处理)
        5. 外部API工具 (GitHub、Slack等)

        技术实现:
        - 使用LiteLLM的ChatCompletionToolParam格式
        - 支持工具去重和冲突检测
        - 动态更新工具列表

        Args:
            mcp_tools: MCP工具定义列表，每个工具包含function定义和参数
        """
        logger.info(
            f'Setting {len(mcp_tools)} MCP tools for agent {self.name}: '
            f'{[tool["function"]["name"] for tool in mcp_tools]}'
        )

        # 遍历并添加每个工具
        for tool in mcp_tools:
            # 转换为标准的ChatCompletionToolParam格式
            _tool = ChatCompletionToolParam(**tool)

            # 检查工具是否已存在，避免重复添加
            if _tool['function']['name'] in self.mcp_tools:
                logger.warning(
                    f'Tool {_tool["function"]["name"]} already exists, skipping'
                )
                continue

            # 添加到工具字典和列表
            self.mcp_tools[_tool['function']['name']] = _tool
            self.tools.append(_tool)

        logger.info(
            f'Tools updated for agent {self.name}, total {len(self.tools)}: '
            f'{[tool["function"]["name"] for tool in self.tools]}'
        )
