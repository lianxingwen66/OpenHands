"""
OpenHands DummyAgent - 测试用虚拟代理

技术栈:
- Python 3.12+ (核心语言)
- TypedDict - 类型化字典，提供类型安全
- 事件序列化 - 事件对象的字典转换
- 预定义操作序列 - 确定性的测试流程

架构说明:
DummyAgent是专门用于端到端测试的虚拟代理。它不进行任何LLM调用，
而是按照预定义的操作序列确定性地执行，用于验证系统的基础功能。

核心特性:
1. 确定性执行 - 每次运行都产生相同的结果
2. 无LLM依赖 - 不需要外部LLM服务
3. 完整测试覆盖 - 涵盖所有主要操作类型
4. 观察验证 - 验证期望的观察结果
5. 快速测试 - 无网络延迟的快速执行

测试场景:
- 消息发送和接收
- 命令执行和输出
- 文件读写操作
- 网页浏览功能
- Agent状态变化
- 任务完成流程

设计模式:
- 模板方法模式: 预定义的执行模板
- 状态模式: Agent状态管理
- 观察者模式: 观察结果验证
"""

from typing import TypedDict

# 核心框架导入
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.schema import AgentState

# 事件系统 - Action类型
from openhands.events.action import (
    Action,  # 基础Action类
    AgentFinishAction,  # 任务完成Action
    AgentRejectAction,  # 任务拒绝Action
    BrowseInteractiveAction,  # 交互式浏览Action
    BrowseURLAction,  # URL浏览Action
    CmdRunAction,  # 命令执行Action
    FileReadAction,  # 文件读取Action
    FileWriteAction,  # 文件写入Action
    MessageAction,  # 消息Action
)

# 事件系统 - Observation类型
from openhands.events.observation import (
    AgentStateChangedObservation,  # Agent状态变化观察
    BrowserOutputObservation,  # 浏览器输出观察
    CmdOutputMetadata,  # 命令输出元数据
    CmdOutputObservation,  # 命令输出观察
    FileReadObservation,  # 文件读取观察
    FileWriteObservation,  # 文件写入观察
    Observation,  # 基础观察类
)

# 事件序列化
from openhands.events.serialization.event import event_to_dict
from openhands.llm.llm import LLM

# 已知问题记录
"""
FIXME: 发现的一些问题
* FileWrites 似乎在文件末尾添加了意外的换行符
* Browser 功能不工作
"""

# 类型定义：Action和Observation的组合
ActionObs = TypedDict(
    'ActionObs', {'action': Action, 'observations': list[Observation]}
)
"""
ActionObs类型定义
用于表示一个操作及其对应的观察结果的组合。
这种结构化的定义使得测试用例更加清晰和类型安全。
"""


class DummyAgent(Agent):
    """
    DummyAgent - 测试用虚拟代理

    专门用于端到端测试的Agent实现。它按照预定义的操作序列
    确定性地执行，不进行任何LLM调用，确保测试结果的可重复性。

    核心职责:
    1. 提供确定性的测试执行流程
    2. 验证系统各组件的基础功能
    3. 模拟真实Agent的操作序列
    4. 检查观察结果的正确性

    测试覆盖:
    - 消息处理
    - 命令执行
    - 文件操作
    - 网页浏览
    - 状态管理
    - 任务完成

    使用场景:
    - 集成测试
    - 回归测试
    - 性能基准测试
    - 系统验证
    """

    VERSION = '1.0'
    """
    DummyAgent版本号

    版本历史:
    - 1.0: 初始版本，包含基础测试操作序列
    """

    def __init__(self, llm: LLM, config: AgentConfig):
        """
        初始化DummyAgent

        设置预定义的测试操作序列，每个操作都包含期望的观察结果。
        这些操作覆盖了OpenHands系统的主要功能点。

        Args:
            llm: LLM实例（虽然不会被使用）
            config: Agent配置
        """
        super().__init__(llm, config)
        self.steps: list[ActionObs] = [
            {
                'action': MessageAction('Time to get started!'),
                'observations': [],
            },
            {
                'action': CmdRunAction(command='echo "foo"'),
                'observations': [CmdOutputObservation('foo', command='echo "foo"')],
            },
            {
                'action': FileWriteAction(
                    content='echo "Hello, World!"', path='hello.sh'
                ),
                'observations': [
                    FileWriteObservation(
                        content='echo "Hello, World!"', path='hello.sh'
                    )
                ],
            },
            {
                'action': FileReadAction(path='hello.sh'),
                'observations': [
                    FileReadObservation('echo "Hello, World!"\n', path='hello.sh')
                ],
            },
            {
                'action': CmdRunAction(command='bash hello.sh'),
                'observations': [
                    CmdOutputObservation(
                        'bash: hello.sh: No such file or directory',
                        command='bash workspace/hello.sh',
                        metadata=CmdOutputMetadata(exit_code=127),
                    )
                ],
            },
            {
                'action': BrowseURLAction(url='https://google.com'),
                'observations': [
                    BrowserOutputObservation(
                        '<html><body>Simulated Google page</body></html>',
                        url='https://google.com',
                        screenshot='',
                        trigger_by_action='',
                    ),
                ],
            },
            {
                'action': BrowseInteractiveAction(
                    browser_actions='goto("https://google.com")'
                ),
                'observations': [
                    BrowserOutputObservation(
                        '<html><body>Simulated Google page after interaction</body></html>',
                        url='https://google.com',
                        screenshot='',
                        trigger_by_action='',
                    ),
                ],
            },
            {
                'action': AgentRejectAction(),
                'observations': [AgentStateChangedObservation('', AgentState.REJECTED)],
            },
            {
                'action': AgentFinishAction(
                    outputs={}, thought='Task completed', action='finish'
                ),
                'observations': [AgentStateChangedObservation('', AgentState.FINISHED)],
            },
        ]

    def step(self, state: State) -> Action:
        if state.iteration >= len(self.steps):
            return AgentFinishAction()

        current_step = self.steps[state.iteration]
        action = current_step['action']

        if state.iteration > 0:
            prev_step = self.steps[state.iteration - 1]

            if 'observations' in prev_step and prev_step['observations']:
                expected_observations = prev_step['observations']
                hist_events = state.view[-len(expected_observations) :]

                if len(hist_events) < len(expected_observations):
                    print(
                        f'Warning: Expected {len(expected_observations)} observations, but got {len(hist_events)}'
                    )

                for i in range(min(len(expected_observations), len(hist_events))):
                    hist_obs = event_to_dict(hist_events[i])
                    expected_obs = event_to_dict(expected_observations[i])

                    # Remove dynamic fields for comparison
                    for obs in [hist_obs, expected_obs]:
                        obs.pop('id', None)
                        obs.pop('timestamp', None)
                        obs.pop('cause', None)
                        obs.pop('source', None)

                    if hist_obs != expected_obs:
                        print(
                            f'Warning: Observation mismatch. Expected {expected_obs}, got {hist_obs}'
                        )

        return action
