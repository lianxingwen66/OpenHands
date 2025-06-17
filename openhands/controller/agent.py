"""
OpenHands Agent 抽象基类模块

这个模块定义了OpenHands系统中所有AI Agent的基础抽象类和核心接口。

技术栈：
- Python ABC (抽象基类) - 确保接口一致性
- 类型提示 (Type Hints) - 提供静态类型检查
- 注册表模式 (Registry Pattern) - 管理Agent类型注册
- 工厂模式 (Factory Pattern) - 动态创建Agent实例
- LiteLLM - 统一的LLM接口库
- MCP (Model Context Protocol) - 工具调用协议

核心设计理念：
1. 抽象化：定义所有Agent必须实现的基本接口
2. 可扩展性：支持注册新的Agent类型
3. 工具集成：内置MCP工具支持
4. 状态管理：与State对象紧密集成
5. 提示管理：集成PromptManager进行提示词管理

Agent生命周期：
1. 初始化：配置LLM和Agent参数
2. 系统消息：生成初始系统提示
3. 步骤执行：根据状态生成下一个动作
4. 重置：清理状态准备下一轮对话
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.controller.state.state import State
    from openhands.core.config import AgentConfig
    from openhands.events.action import Action
    from openhands.events.action.message import SystemMessageAction
    from openhands.utils.prompt import PromptManager
from litellm import ChatCompletionToolParam

from openhands.core.exceptions import (
    AgentAlreadyRegisteredError,
    AgentNotRegisteredError,
)
from openhands.core.logger import openhands_logger as logger
from openhands.events.event import EventSource
from openhands.llm.llm import LLM
from openhands.runtime.plugins import PluginRequirement


class Agent(ABC):
    """
    OpenHands Agent 抽象基类

    这是一个通用的Agent接口，专门用于执行特定指令并允许在执行过程中与人类进行交互。
    它跟踪执行状态并维护交互历史记录。

    技术架构：
    - 抽象基类设计：确保所有Agent实现一致的接口
    - 注册表模式：支持动态注册和发现Agent类型
    - 插件系统：支持沙箱插件扩展功能
    - 工具集成：内置MCP工具调用支持
    - 状态管理：跟踪Agent的完成状态

    核心属性：
    - DEPRECATED: 标记Agent是否已弃用
    - _registry: 全局Agent类型注册表
    - sandbox_plugins: 沙箱插件需求列表
    - llm: 大语言模型实例
    - config: Agent配置对象
    - _complete: 执行完成状态标志
    - _prompt_manager: 提示词管理器
    - mcp_tools: MCP工具字典
    - tools: 工具列表

    设计模式：
    1. 模板方法模式：定义Agent执行的基本流程
    2. 策略模式：不同Agent实现不同的执行策略
    3. 观察者模式：与事件系统集成
    4. 工厂模式：通过注册表创建Agent实例
    """

    # 标记此Agent类型是否已弃用
    # 弃用的Agent仍可使用但会显示警告信息
    DEPRECATED = False

    # 全局Agent类型注册表，用于动态创建Agent实例
    # 键为Agent名称，值为Agent类型
    _registry: dict[str, type['Agent']] = {}

    # 沙箱插件需求列表，定义Agent运行所需的插件
    # 这些插件会在运行时环境中自动安装和配置
    sandbox_plugins: list[PluginRequirement] = []

    def __init__(
        self,
        llm: LLM,
        config: 'AgentConfig',
    ):
        """
        初始化Agent实例

        Args:
            llm (LLM): 大语言模型实例，用于生成响应和执行推理
            config (AgentConfig): Agent配置对象，包含各种运行参数

        技术细节：
        - llm: 使用LiteLLM统一接口，支持多种LLM提供商
        - config: 包含模型参数、行为设置、安全配置等
        - _complete: 用于标记当前任务是否完成
        - _prompt_manager: 延迟初始化，管理系统提示和用户提示
        - mcp_tools: 存储MCP协议的工具定义
        - tools: 存储所有可用工具的列表
        """
        # 大语言模型实例，负责生成文本和执行推理
        self.llm = llm

        # Agent配置对象，包含行为参数和设置
        self.config = config

        # 执行完成状态标志，用于判断当前任务是否已完成
        self._complete = False

        # 提示词管理器，延迟初始化以避免循环依赖
        self._prompt_manager: 'PromptManager' | None = None

        # MCP工具字典，键为工具名称，值为工具定义
        # MCP (Model Context Protocol) 是用于工具调用的标准协议
        self.mcp_tools: dict[str, ChatCompletionToolParam] = {}

        # 所有可用工具的列表，包括MCP工具和其他工具
        self.tools: list = []

    @property
    def prompt_manager(self) -> 'PromptManager':
        """
        获取提示词管理器实例

        提示词管理器负责管理系统提示、用户提示和微代理提示。
        它确保提示词的一致性和可维护性。

        Returns:
            PromptManager: 提示词管理器实例

        Raises:
            ValueError: 当提示词管理器未初始化时抛出

        技术说明：
        - 延迟初始化模式：避免循环依赖问题
        - 单例模式：每个Agent实例只有一个PromptManager
        - 异常安全：提供清晰的错误信息
        """
        if self._prompt_manager is None:
            raise ValueError(f'Prompt manager not initialized for agent {self.name}')
        return self._prompt_manager

    def get_system_message(self) -> 'SystemMessageAction | None':
        """
        生成系统消息动作

        返回包含系统消息和工具定义的SystemMessageAction对象。
        这将作为事件流中的第一条消息添加。

        技术流程：
        1. 检查提示词管理器是否已初始化
        2. 从提示词管理器获取系统消息内容
        3. 收集可用的工具列表
        4. 创建SystemMessageAction对象
        5. 设置事件源为AGENT

        Returns:
            SystemMessageAction: 包含系统消息内容和工具的动作对象
            None: 如果生成系统消息时发生错误

        异常处理：
        - 捕获所有异常并记录警告日志
        - 返回None而不是抛出异常，确保系统稳定性

        设计考虑：
        - 循环导入：在方法内部导入以避免循环依赖
        - 容错性：即使失败也不影响Agent的其他功能
        - 工具集成：自动包含Agent的所有可用工具
        """
        # 在此处导入以避免循环导入问题
        from openhands.events.action.message import SystemMessageAction

        try:
            # 检查提示词管理器是否已初始化
            if not self.prompt_manager:
                logger.warning(
                    f'[{self.name}] Prompt manager not initialized before getting system message'
                )
                return None

            # 从提示词管理器获取系统消息内容
            system_message = self.prompt_manager.get_system_message()

            # 获取可用的工具列表（如果存在）
            tools = getattr(self, 'tools', None)

            # 创建系统消息动作对象
            system_message_action = SystemMessageAction(
                content=system_message, tools=tools, agent_class=self.name
            )
            # 设置事件源属性为AGENT
            system_message_action._source = EventSource.AGENT  # type: ignore

            return system_message_action
        except Exception as e:
            # 记录警告日志但不抛出异常，确保系统稳定性
            logger.warning(f'[{self.name}] Failed to generate system message: {e}')
            return None

    @property
    def complete(self) -> bool:
        """
        指示当前指令执行是否完成

        这个属性用于检查Agent是否已完成当前分配的任务。
        控制器使用此属性来决定是否继续执行或结束会话。

        Returns:
            bool: True表示执行完成，False表示仍在执行中

        使用场景：
        - 任务完成检查：控制器定期检查此属性
        - 状态转换：用于决定Agent状态转换
        - 资源清理：完成后可以释放相关资源

        设计说明：
        - 只读属性：外部代码不能直接修改完成状态
        - 线程安全：读取操作是原子的
        - 状态一致性：与内部_complete字段保持同步
        """
        return self._complete

    @abstractmethod
    def step(self, state: 'State') -> 'Action':
        """
        执行单步推理并生成下一个动作

        这是Agent的核心方法，负责根据当前状态生成下一个要执行的动作。
        每个Agent子类必须实现此方法来定义特定的执行逻辑。

        Args:
            state (State): 当前的Agent状态，包含历史记录、配置和上下文信息

        Returns:
            Action: 要执行的下一个动作对象

        实现要求：
        1. 分析当前状态和历史记录
        2. 使用LLM进行推理和决策
        3. 生成适当的动作对象
        4. 处理异常情况

        技术考虑：
        - 状态分析：充分利用state中的信息
        - LLM调用：合理使用self.llm进行推理
        - 动作生成：确保生成的动作是有效和安全的
        - 错误处理：妥善处理各种异常情况

        设计模式：
        - 模板方法模式：定义执行框架，子类实现具体逻辑
        - 策略模式：不同Agent实现不同的推理策略
        """
        pass

    def reset(self) -> None:
        """
        重置Agent的执行状态并清理历史记录

        此方法用于准备Agent重新开始指令执行或在销毁前进行清理。
        重置操作确保Agent可以从干净的状态开始新的任务。

        重置操作包括：
        1. 将完成状态标志设为False
        2. 重置LLM的内部状态
        3. 清理历史记录（TODO：待实现）

        使用场景：
        - 新任务开始：在开始新任务前重置状态
        - 错误恢复：在发生错误后重置到初始状态
        - 资源清理：在Agent销毁前进行清理

        技术细节：
        - 状态重置：确保所有状态变量回到初始值
        - LLM重置：清理LLM的缓存和状态
        - 内存管理：释放不必要的内存占用

        注意事项：
        - 线程安全：重置操作应该是原子的
        - 异常处理：重置过程中的异常不应影响系统稳定性
        - 完整性：确保重置后Agent处于一致的状态
        """
        # TODO: 清理历史记录
        # 将来需要实现历史记录的清理逻辑
        self._complete = False

        # 重置LLM的内部状态（如果LLM实例存在）
        if self.llm:
            self.llm.reset()

    @property
    def name(self) -> str:
        """
        获取Agent的名称

        Agent名称基于其类名，用于标识和日志记录。
        这个名称在注册表中用作键，也用于配置和调试。

        Returns:
            str: Agent的类名作为其名称

        使用场景：
        - 日志记录：在日志中标识Agent类型
        - 注册表查找：作为注册表的键
        - 配置匹配：匹配特定Agent的配置
        - 调试信息：提供清晰的Agent标识

        设计说明：
        - 自动生成：基于类名自动生成，无需手动维护
        - 唯一性：类名确保了名称的唯一性
        - 一致性：所有Agent实例使用相同的命名规则
        """
        return self.__class__.__name__

    @classmethod
    def register(cls, name: str, agent_cls: type['Agent']) -> None:
        """
        在注册表中注册Agent类

        这个类方法用于将Agent类注册到全局注册表中，使其可以通过名称动态创建。
        注册表模式允许系统在运行时发现和实例化不同类型的Agent。

        Args:
            name (str): 注册Agent类时使用的名称标识符
            agent_cls (Type['Agent']): 要注册的Agent类

        Raises:
            AgentAlreadyRegisteredError: 如果名称已经被注册则抛出此异常

        技术特点：
        - 类方法：作为类的静态方法，不需要实例即可调用
        - 全局注册表：所有Agent类共享同一个注册表
        - 名称唯一性：确保每个名称只对应一个Agent类
        - 动态发现：支持运行时发现可用的Agent类型

        使用场景：
        - 插件系统：动态加载和注册新的Agent类型
        - 配置驱动：根据配置文件选择Agent类型
        - 工厂模式：通过名称创建Agent实例

        设计考虑：
        - 线程安全：注册操作应该是原子的
        - 异常安全：重复注册时提供清晰的错误信息
        - 可扩展性：支持无限数量的Agent类型注册
        """
        if name in cls._registry:
            raise AgentAlreadyRegisteredError(name)
        cls._registry[name] = agent_cls

    @classmethod
    def get_cls(cls, name: str) -> type['Agent']:
        """
        从注册表中检索Agent类

        根据名称从全局注册表中获取对应的Agent类，用于动态创建Agent实例。
        这是工厂模式的核心实现，支持基于配置的Agent创建。

        Args:
            name (str): 要检索的Agent类的名称

        Returns:
            type['Agent']: 注册在指定名称下的Agent类

        Raises:
            AgentNotRegisteredError: 如果名称未注册则抛出此异常

        技术特点：
        - 类方法：无需实例即可调用
        - 类型安全：返回正确的Agent类型
        - 异常处理：未找到时提供清晰的错误信息
        - 动态查找：支持运行时类型解析

        使用场景：
        - Agent工厂：根据配置创建特定类型的Agent
        - 插件加载：动态加载和实例化Agent插件
        - 类型验证：检查Agent类型是否可用

        设计模式：
        - 工厂模式：根据名称创建对象
        - 注册表模式：集中管理可用类型
        - 单例模式：全局共享的注册表
        """
        if name not in cls._registry:
            raise AgentNotRegisteredError(name)
        return cls._registry[name]

    @classmethod
    def list_agents(cls) -> list[str]:
        """
        检索注册表中所有Agent名称的列表

        返回当前已注册的所有Agent类型名称，用于发现可用的Agent类型。
        这对于用户界面、配置验证和调试非常有用。

        Returns:
            list[str]: 所有已注册Agent的名称列表

        Raises:
            AgentNotRegisteredError: 如果没有注册任何Agent则抛出此异常

        技术特点：
        - 类方法：无需实例即可调用
        - 动态列表：反映当前注册状态
        - 异常安全：空注册表时提供清晰的错误信息
        - 只读操作：不修改注册表状态

        使用场景：
        - 用户界面：显示可用的Agent类型选项
        - 配置验证：检查配置中的Agent类型是否有效
        - 调试工具：列出系统中可用的Agent类型
        - 文档生成：自动生成Agent类型文档

        设计考虑：
        - 性能：列表生成操作应该是高效的
        - 一致性：返回的列表应该反映当前状态
        - 可用性：提供友好的错误信息
        """
        if not bool(cls._registry):
            raise AgentNotRegisteredError()
        return list(cls._registry.keys())

    def set_mcp_tools(self, mcp_tools: list[dict]) -> None:
        """
        为Agent设置MCP工具列表

        MCP (Model Context Protocol) 是用于工具调用的标准协议。
        此方法将MCP工具定义添加到Agent的工具集合中，使Agent能够调用外部工具。

        Args:
            mcp_tools (list[dict]): MCP工具定义的列表，每个工具包含函数名称和参数定义

        技术流程：
        1. 记录设置的工具数量和名称
        2. 遍历每个工具定义
        3. 将字典转换为ChatCompletionToolParam对象
        4. 检查工具名称是否已存在，避免重复
        5. 将工具添加到mcp_tools字典和tools列表
        6. 记录最终的工具配置状态

        特性：
        - 重复检测：避免添加同名工具
        - 类型转换：自动转换为标准的工具参数格式
        - 日志记录：详细记录工具设置过程
        - 状态维护：同时更新字典和列表结构

        使用场景：
        - 工具集成：集成外部API和服务
        - 功能扩展：动态添加Agent能力
        - 插件系统：支持工具插件的热加载

        技术考虑：
        - 线程安全：工具设置操作应该是原子的
        - 内存管理：避免重复存储相同的工具
        - 错误处理：妥善处理格式错误的工具定义
        - 性能优化：高效的重复检测机制
        """
        logger.info(
            f'Setting {len(mcp_tools)} MCP tools for agent {self.name}: {[tool["function"]["name"] for tool in mcp_tools]}'
        )

        for tool in mcp_tools:
            # 将字典转换为标准的ChatCompletionToolParam对象
            _tool = ChatCompletionToolParam(**tool)

            # 检查工具名称是否已存在，避免重复添加
            if _tool['function']['name'] in self.mcp_tools:
                logger.warning(
                    f'Tool {_tool["function"]["name"]} already exists, skipping'
                )
                continue

            # 将工具添加到MCP工具字典中
            self.mcp_tools[_tool['function']['name']] = _tool

            # 同时添加到通用工具列表中
            self.tools.append(_tool)

        # 记录最终的工具配置状态
        logger.info(
            f'Tools updated for agent {self.name}, total {len(self.tools)}: {[tool["function"]["name"] for tool in self.tools]}'
        )
