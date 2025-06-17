"""
OpenHands 动作解析器模块

这个模块定义了用于解析大语言模型(LLM)响应并将其转换为可执行动作的抽象接口和异常类。

技术栈：
- Python ABC (抽象基类) - 定义接口规范
- 策略模式 (Strategy Pattern) - 支持多种解析策略
- 责任链模式 (Chain of Responsibility) - 按顺序尝试不同解析器

核心概念：
1. ResponseParser: 响应解析器的顶层抽象，负责将LLM的原始响应转换为Action对象
2. ActionParser: 具体的动作解析器，每个解析器负责处理特定格式的动作字符串
3. ActionParseError: 解析失败时抛出的异常

设计模式：
- 抽象工厂模式：ResponseParser作为工厂创建Action对象
- 策略模式：不同的ActionParser实现不同的解析策略
- 责任链模式：多个ActionParser按顺序尝试解析

使用场景：
- 解析LLM生成的JSON格式动作
- 解析LLM生成的自然语言动作描述
- 处理不同Agent类型的特定动作格式
- 错误处理和格式验证
"""

from abc import ABC, abstractmethod
from typing import Any

from openhands.events.action import Action


class ActionParseError(Exception):
    """
    动作解析异常类

    当LLM的响应无法被解析为有效的Action对象时抛出此异常。
    这通常发生在以下情况：
    1. LLM响应格式不正确
    2. 缺少必需的字段
    3. 字段类型不匹配
    4. 动作类型不被支持

    技术特点：
    - 继承自Python标准Exception类
    - 提供详细的错误信息用于调试
    - 支持错误信息的字符串表示
    """

    def __init__(self, error: str):
        """
        初始化动作解析异常

        Args:
            error (str): 详细的错误描述信息
        """
        self.error = error

    def __str__(self) -> str:
        """
        返回异常的字符串表示

        Returns:
            str: 错误信息字符串
        """
        return self.error


class ResponseParser(ABC):
    """
    响应解析器抽象基类

    这是一个通用的响应解析器接口，专门用于将LLM的响应解析为Action对象。

    技术架构：
    - 使用ABC (Abstract Base Class) 确保接口一致性
    - 维护ActionParser列表，支持多种解析策略
    - 采用责任链模式，按顺序尝试不同的解析器

    设计原则：
    1. 单一职责：专注于响应解析功能
    2. 开闭原则：可扩展新的解析器而不修改现有代码
    3. 依赖倒置：依赖抽象而非具体实现

    使用流程：
    1. parse_response(): 从原始响应中提取动作字符串
    2. parse_action(): 将动作字符串解析为Action对象
    3. parse(): 完整的解析流程，组合上述两步
    """

    def __init__(
        self,
    ) -> None:
        """
        初始化响应解析器

        注意：action_parsers列表的顺序很重要，解析器会按顺序尝试解析
        """
        # 需要注意self.action_parsers中项目的顺序
        # 解析器会按照列表顺序依次尝试解析，直到成功或全部失败
        self.action_parsers: list[ActionParser] = []

    @abstractmethod
    def parse(self, response: Any) -> Action:
        """
        从LLM响应中解析出Action对象

        这是主要的解析入口点，将原始响应转换为可执行的Action。

        Args:
            response: LLM的响应，可以是字符串或字典格式

        Returns:
            Action: 解析得到的动作对象

        Raises:
            ActionParseError: 当响应无法被解析时抛出
        """
        pass

    @abstractmethod
    def parse_response(self, response: Any) -> str:
        """
        从LLM响应中提取动作字符串

        这个方法负责从LLM的原始响应中提取出包含动作信息的字符串部分。
        不同的LLM可能有不同的响应格式，这个方法需要适配具体的格式。

        Args:
            response: LLM的响应，可以是字符串或字典格式

        Returns:
            str: 提取出的动作字符串

        Raises:
            ActionParseError: 当无法从响应中提取动作字符串时抛出
        """
        pass

    @abstractmethod
    def parse_action(self, action_str: str) -> Action:
        """
        将动作字符串解析为Action对象

        这个方法使用注册的ActionParser列表，按顺序尝试解析动作字符串。

        Args:
            action_str (str): 包含动作信息的字符串

        Returns:
            Action: 解析得到的动作对象

        Raises:
            ActionParseError: 当所有解析器都无法解析时抛出
        """
        pass


class ActionParser(ABC):
    """
    动作解析器抽象基类

    这是一个通用的动作解析器接口，专门用于将特定格式的动作字符串解析为Action对象。

    技术特点：
    - 使用策略模式，每个具体解析器实现特定的解析策略
    - 支持条件检查，确保解析器只处理它能理解的格式
    - 职责单一，每个解析器专注于一种动作格式

    实现要求：
    1. check_condition(): 必须能够快速判断是否能解析给定的字符串
    2. parse(): 必须能够将字符串解析为有效的Action对象

    常见的实现类型：
    - JSONActionParser: 解析JSON格式的动作
    - MarkdownActionParser: 解析Markdown格式的动作
    - PlainTextActionParser: 解析纯文本格式的动作
    """

    @abstractmethod
    def check_condition(self, action_str: str) -> bool:
        """
        检查动作字符串是否可以被此解析器解析

        这个方法用于快速判断当前解析器是否能够处理给定的动作字符串。
        应该尽可能快速和准确，避免复杂的解析逻辑。

        Args:
            action_str (str): 待检查的动作字符串

        Returns:
            bool: True表示可以解析，False表示不能解析

        设计建议：
        - 使用简单的字符串匹配或正则表达式
        - 检查关键字或格式标识符
        - 避免实际的解析操作，只做格式检查
        """
        pass

    @abstractmethod
    def parse(self, action_str: str) -> Action:
        """
        将动作字符串解析为Action对象

        这个方法执行实际的解析工作，将字符串转换为具体的Action对象。
        调用此方法前应该先调用check_condition()确认可以解析。

        Args:
            action_str (str): 来自LLM响应的动作字符串

        Returns:
            Action: 解析得到的动作对象

        Raises:
            ActionParseError: 当解析失败时抛出

        实现建议：
        - 使用try-catch处理解析异常
        - 提供详细的错误信息
        - 验证解析结果的有效性
        """
        pass
