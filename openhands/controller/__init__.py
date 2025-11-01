"""
OpenHands Controller 模块

这个模块是OpenHands系统的核心控制层，负责管理AI Agent的生命周期、状态转换和任务执行。

技术栈：
- Python 3.8+ (核心语言)
- asyncio (异步编程框架)
- 事件驱动架构 (Event-driven Architecture)
- 观察者模式 (Observer Pattern)
- 状态机模式 (State Machine Pattern)

主要组件：
- AgentController: 主控制器，管理Agent的完整生命周期
- Agent: Agent抽象基类，定义所有AI Agent的基本接口
- State: 状态管理器，维护Agent的运行状态和历史数据
- ActionParser: 动作解析器，解析LLM响应为可执行动作
- ReplayManager: 重放管理器，支持历史轨迹的重放功能
- StuckDetector: 循环检测器，防止Agent陷入无限循环

架构特点：
1. 多Agent支持：支持Agent委托和多层级Agent协作
2. 事件流处理：基于事件流的异步消息传递
3. 状态持久化：支持会话状态的保存和恢复
4. 错误恢复：完善的异常处理和错误恢复机制
5. 性能监控：内置指标收集和性能监控
"""

from openhands.controller.agent_controller import AgentController

__all__ = [
    'AgentController',
]
