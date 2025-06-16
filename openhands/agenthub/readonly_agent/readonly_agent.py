"""
OpenHands ReadOnlyAgent - 只读模式智能代理

技术栈:
- Python 3.12+ (核心语言)
- LiteLLM - 多LLM提供商统一接口
- CodeAct架构 - 继承自CodeActAgent
- 只读工具集 - 安全的文件系统访问
- Function Calling - 受限的函数调用机制
- Jinja2 - 模板引擎 (通过PromptManager)

架构说明:
ReadOnlyAgent是CodeActAgent的安全特化版本，专门设计用于
只读操作。它限制了Agent的操作权限，只允许执行不会修改
系统状态的安全操作，适用于代码探索和研究场景。

核心能力:
1. 代码库探索 - 安全地浏览项目结构
2. 文件内容查看 - 读取文件但不修改
3. 模式搜索 - 使用grep等工具搜索代码
4. 文件匹配 - 使用glob模式匹配文件
5. 网页读取 - 获取网络信息
6. 思考推理 - 分析和总结信息

安全特性:
- 无文件写入权限
- 无命令执行权限
- 无系统修改能力
- 只读文件系统访问
- 受限的网络访问

应用场景:
- 代码库分析和理解
- 安全的代码审查
- 研究和学习
- 信息收集和整理
- 代码模式识别

设计模式:
- 继承模式: 继承CodeActAgent基础功能
- 装饰器模式: 添加安全限制
- 策略模式: 只读操作策略
- 代理模式: 安全的操作代理
"""

import os
from typing import TYPE_CHECKING

# 类型检查时导入，避免循环导入
if TYPE_CHECKING:
    from litellm import ChatCompletionToolParam

    from openhands.events.action import Action
    from openhands.llm.llm import ModelResponse

# 继承自CodeActAgent
from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent

# 只读代理专用函数调用模块
from openhands.agenthub.readonly_agent import (
    function_calling as readonly_function_calling,
)

# 核心框架组件
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import LLM
from openhands.utils.prompt import PromptManager


class ReadOnlyAgent(CodeActAgent):
    """
    ReadOnlyAgent - 只读模式智能代理

    CodeActAgent的安全特化版本，专门用于只读操作。
    通过限制可用工具集，确保Agent无法修改系统状态，
    提供安全的代码探索和研究环境。

    核心特性:
    1. 安全限制 - 只允许只读操作
    2. 代码探索 - 深入理解代码库结构
    3. 模式搜索 - 高效的代码搜索能力
    4. 信息收集 - 从多种来源收集信息
    5. 分析推理 - 基于收集的信息进行分析

    可用工具:
    - grep: 文本搜索和模式匹配
    - glob: 文件路径模式匹配
    - view: 文件内容查看
    - think: 思考和推理
    - finish: 任务完成
    - web_read: 网页内容读取

    安全保证:
    - 无文件修改能力
    - 无命令执行权限
    - 无系统状态改变
    - 受控的网络访问

    使用建议:
    1. 探索代码库结构时使用
    2. 搜索特定模式或代码时使用
    3. 进行研究而不做修改时使用
    4. 准备修改时切换到常规CodeActAgent
    """

    VERSION = '1.0'
    """
    The ReadOnlyAgent is a specialized version of CodeActAgent that only uses read-only tools.

    This agent is designed for safely exploring codebases without making any changes.
    It only has access to tools that don't modify the system: grep, glob, view, think, finish, web_read.

    Use this agent when you want to:
    1. Explore a codebase to understand its structure
    2. Search for specific patterns or code
    3. Research without making any changes

    When you're ready to make changes, switch to the regular CodeActAgent.
    """

    def __init__(
        self,
        llm: LLM,
        config: AgentConfig,
    ) -> None:
        """Initializes a new instance of the ReadOnlyAgent class.

        Parameters:
        - llm (LLM): The llm to be used by this agent
        - config (AgentConfig): The configuration for this agent
        """
        # Initialize the CodeActAgent class; some of it is overridden with class methods
        super().__init__(llm, config)

        logger.debug(
            f'TOOLS loaded for ReadOnlyAgent: {", ".join([tool.get("function").get("name") for tool in self.tools])}'
        )

    @property
    def prompt_manager(self) -> PromptManager:
        # Set up our own prompt manager
        if self._prompt_manager is None:
            self._prompt_manager = PromptManager(
                prompt_dir=os.path.join(os.path.dirname(__file__), 'prompts'),
            )
        return self._prompt_manager

    def _get_tools(self) -> list['ChatCompletionToolParam']:
        # Override the tools to only include read-only tools
        # Get the read-only tools from our own function_calling module
        return readonly_function_calling.get_tools()

    def set_mcp_tools(self, mcp_tools: list[dict]) -> None:
        """Sets the list of MCP tools for the agent.

        Args:
        - mcp_tools (list[dict]): The list of MCP tools.
        """
        logger.warning(
            'ReadOnlyAgent does not support MCP tools. MCP tools will be ignored by the agent.'
        )

    def response_to_actions(self, response: 'ModelResponse') -> list['Action']:
        return readonly_function_calling.response_to_actions(
            response, mcp_tool_names=list(self.mcp_tools.keys())
        )
