"""
OpenHands LocAgent - 基于位置的操作代理

技术栈:
- Python 3.12+ (核心语言)
- LiteLLM - 多LLM提供商统一接口
- Function Calling - LLM函数调用机制
- CodeAct架构 - 继承自CodeActAgent
- 工具系统 - 专门的位置相关工具集

架构说明:
LocAgent是基于CodeActAgent的专门化代理，专注于处理与位置、
坐标、空间相关的操作任务。它扩展了CodeActAgent的功能，
添加了专门的位置处理工具和函数调用能力。

核心能力:
1. 位置坐标处理
2. 空间关系分析
3. 地理信息操作
4. 坐标系转换
5. 位置相关的代码执行
6. 空间数据可视化

应用场景:
- 地理信息系统(GIS)开发
- 地图应用开发
- 位置服务集成
- 空间数据分析
- 导航系统开发

设计模式:
- 继承模式: 继承CodeActAgent的核心功能
- 装饰器模式: 扩展位置相关功能
- 策略模式: 不同的位置处理策略
- 工厂模式: 位置工具的创建和管理
"""

from typing import TYPE_CHECKING

# 位置代理专用函数调用模块
import openhands.agenthub.loc_agent.function_calling as locagent_function_calling

# 继承自CodeActAgent
from openhands.agenthub.codeact_agent import CodeActAgent
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import LLM

# 类型检查时导入，避免循环导入
if TYPE_CHECKING:
    from openhands.events.action import Action
    from openhands.llm.llm import ModelResponse


class LocAgent(CodeActAgent):
    """
    LocAgent - 基于位置的操作代理

    继承自CodeActAgent，专门处理与位置、坐标、空间相关的任务。
    通过专门的工具集和函数调用机制，提供强大的位置处理能力。

    核心特性:
    1. 位置数据处理 - 支持多种坐标系统
    2. 空间关系分析 - 计算距离、方向、区域
    3. 地理编码 - 地址与坐标的相互转换
    4. 地图操作 - 地图显示、标记、路径规划
    5. 空间查询 - 基于位置的数据查询
    6. 可视化 - 位置数据的图形化展示

    工具集成:
    - 地理信息处理工具
    - 坐标转换工具
    - 地图服务API
    - 空间计算工具
    - 可视化工具

    继承优势:
    - 保留CodeAct的代码执行能力
    - 扩展位置相关的专门功能
    - 统一的操作接口
    - 完整的工具生态系统
    """

    VERSION = '1.0'

    def __init__(
        self,
        llm: LLM,
        config: AgentConfig,
    ) -> None:
        """Initializes a new instance of the LocAgent class.

        Parameters:
        - llm (LLM): The llm to be used by this agent
        - config (AgentConfig): The configuration for the agent
        """
        super().__init__(llm, config)

        self.tools = locagent_function_calling.get_tools()
        logger.debug(
            f'TOOLS loaded for LocAgent: {", ".join([tool.get("function").get("name") for tool in self.tools])}'
        )

    def response_to_actions(self, response: 'ModelResponse') -> list['Action']:
        return locagent_function_calling.response_to_actions(
            response,
            mcp_tool_names=list(self.mcp_tools.keys()),
        )
