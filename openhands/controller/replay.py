"""
OpenHands 重放管理器模块

这个模块实现了轨迹重放功能，允许系统重新执行之前记录的Agent行为序列。
重放功能对于调试、测试、演示和评估非常重要。

技术栈：
- Python 类型提示 - 提供静态类型检查
- 事件序列化 - 支持事件的持久化和恢复
- 状态机模式 - 管理重放的不同阶段
- 过滤器模式 - 选择性重放特定类型的事件

核心概念：
1. 轨迹重放：按顺序重新执行历史动作序列
2. 事件过滤：只重放相关的事件，忽略环境事件
3. 状态管理：跟踪重放进度和状态
4. 确定性执行：确保重放结果的一致性

使用场景：
- 调试分析：重现问题场景进行调试
- 回归测试：验证系统行为的一致性
- 演示展示：重放成功的执行案例
- 性能评估：分析Agent的执行效率

设计考虑：
- 确定性：重放结果应该是可预测的
- 过滤性：只重放必要的事件
- 容错性：处理不完整或损坏的轨迹
- 性能：高效的事件处理和状态管理
"""

from __future__ import annotations

from openhands.core.logger import openhands_logger as logger
from openhands.events.action.action import Action
from openhands.events.action.message import MessageAction
from openhands.events.event import Event, EventSource
from openhands.events.observation.empty import NullObservation
from openhands.events.serialization.event import event_from_dict


class ReplayManager:
    """
    重放管理器 - 管理给定轨迹的重放会话生命周期

    重放管理器跟踪事件列表，重放动作，并忽略消息和观察。
    它确保轨迹的确定性重放，用于调试、测试和演示。

    技术架构：
    - 事件过滤：只保留可重放的事件类型
    - 状态跟踪：维护重放进度和模式状态
    - 确定性保证：处理非确定性因素
    - 异常处理：处理不完整或损坏的轨迹

    重要注意事项：
    如果出现以下情况，可能会产生意外或错误的结果：
    1) 任何动作是非确定性的，或者
    2) 重放会话前的初始状态与轨迹的初始状态不同

    设计原则：
    - 确定性：确保重放结果的可预测性
    - 过滤性：只重放相关的事件
    - 完整性：维护事件序列的完整性
    - 可靠性：处理各种边界情况
    """

    def __init__(self, events: list[Event] | None):
        """
        初始化重放管理器

        Args:
            events: 要重放的事件列表，可以为None表示无重放

        技术流程：
        1. 过滤事件：移除环境事件和空观察
        2. 处理消息动作：调整等待响应的设置
        3. 初始化状态：设置重放模式和索引

        过滤规则：
        - 忽略ENVIRONMENT事件：这些事件不是由用户或Agent发出的
        - 忽略NullObservation：这些是空观察，不需要重放
        - 调整MessageAction：移除中间消息的wait_for_response标志
        """
        replay_events = []

        # 遍历所有输入事件并进行过滤
        for event in events or []:
            # 忽略环境事件，因为它们不是由用户或Agent发出的，不应该被重放
            if event.source == EventSource.ENVIRONMENT:
                continue

            # 忽略空观察事件
            if isinstance(event, NullObservation):
                continue

            replay_events.append(event)

        # 如果有重放事件，进行预处理
        if replay_events:
            logger.info(f'Replay events loaded, events length = {len(replay_events)}')

            # 处理除最后一个事件外的所有事件
            for index in range(len(replay_events) - 1):
                event = replay_events[index]

                # 对于等待响应的消息动作，需要特殊处理
                if isinstance(event, MessageAction) and event.wait_for_response:
                    # 对于不是最后一个事件的等待响应消息，我们将wait_for_response设为False
                    # 因为响应已经包含在下一个事件中，我们不希望用户干扰重放过程
                    logger.info(
                        'Replay events contains wait_for_response message action, ignoring wait_for_response'
                    )
                    event.wait_for_response = False

        # 设置重放状态
        self.replay_events = replay_events  # 过滤后的重放事件列表
        self.replay_mode = bool(replay_events)  # 是否处于重放模式
        self.replay_index = 0  # 当前重放事件的索引

    def _replayable(self) -> bool:
        """
        检查当前索引位置的事件是否可以重放

        一个事件可以重放需要满足以下条件：
        1. 重放事件列表不为空
        2. 当前索引在有效范围内
        3. 当前事件是Action类型（只重放动作，不重放观察）

        Returns:
            bool: True表示当前事件可以重放，False表示不可重放

        技术说明：
        - 类型检查：确保只重放Action类型的事件
        - 边界检查：防止索引越界
        - 状态验证：确保重放状态的一致性
        """
        return (
            self.replay_events is not None
            and self.replay_index < len(self.replay_events)
            and isinstance(self.replay_events[self.replay_index], Action)
        )

    def should_replay(self) -> bool:
        """
        判断控制器是否处于轨迹重放模式且重放尚未完成

        注意：重放完成后，用户和Agent可以继续进行消息/动作交互。
        此方法还会将"replay_index"移动到下一个动作（如果适用）。

        Returns:
            bool: True表示应该继续重放，False表示不需要重放

        技术流程：
        1. 检查是否处于重放模式
        2. 跳过非动作事件（如观察、消息等）
        3. 找到下一个可重放的动作
        4. 返回是否找到可重放的事件

        设计考虑：
        - 自动跳过：自动跳过不可重放的事件
        - 状态更新：更新重放索引到正确位置
        - 完成检测：自动检测重放是否已完成
        """
        # 如果不在重放模式，直接返回False
        if not self.replay_mode:
            return False

        assert self.replay_events is not None

        # 跳过所有不可重放的事件，直到找到下一个动作或到达末尾
        while self.replay_index < len(self.replay_events) and not self._replayable():
            self.replay_index += 1

        # 返回是否找到可重放的事件
        return self._replayable()

    def step(self) -> Action:
        """
        执行重放的下一步，返回下一个要重放的动作

        这个方法从重放事件列表中获取当前索引位置的动作，
        并将索引前进到下一个位置。

        Returns:
            Action: 要重放的下一个动作对象

        Raises:
            AssertionError: 如果重放事件列表为空或当前事件不是动作类型

        技术说明：
        - 断言检查：确保重放状态的有效性
        - 类型安全：确保返回的是Action类型
        - 状态更新：自动前进到下一个事件

        使用前提：
        - 必须先调用should_replay()确认有可重放的事件
        - 重放事件列表必须已正确初始化
        - 当前索引必须指向有效的动作事件
        """
        assert self.replay_events is not None
        event = self.replay_events[self.replay_index]
        assert isinstance(event, Action)
        self.replay_index += 1
        return event

    @staticmethod
    def get_replay_events(trajectory: list[dict]) -> list[Event]:
        """
        从轨迹字典列表中获取重放事件列表

        这是一个静态方法，用于将序列化的轨迹数据转换为可重放的事件列表。
        它处理事件的反序列化和过滤，确保只包含可重放的事件。

        Args:
            trajectory: 轨迹字典列表，每个字典代表一个序列化的事件

        Returns:
            list[Event]: 过滤后的重放事件列表

        Raises:
            ValueError: 如果输入不是列表类型

        技术流程：
        1. 验证输入类型
        2. 遍历轨迹中的每个项目
        3. 反序列化事件对象
        4. 过滤环境事件
        5. 清理事件ID以避免冲突
        6. 构建重放事件列表

        过滤规则：
        - 忽略ENVIRONMENT事件：这些事件不应该被重放
        - 清理事件ID：避免与事件流中的现有事件冲突

        使用场景：
        - 从文件加载轨迹：读取保存的轨迹文件
        - 网络传输：接收远程轨迹数据
        - 测试数据：处理测试用的轨迹数据
        """
        # 验证输入类型
        if not isinstance(trajectory, list):
            raise ValueError(
                f'Expected a list in {trajectory}, got {type(trajectory).__name__}'
            )

        replay_events = []

        # 遍历轨迹中的每个项目
        for item in trajectory:
            # 从字典反序列化事件对象
            event = event_from_dict(item)

            # 忽略环境事件，因为它们不是由用户或Agent发出的，不应该被重放
            if event.source == EventSource.ENVIRONMENT:
                continue

            # 清理事件ID，因为不能将带有_id的事件添加到事件流中
            # 这避免了与现有事件的ID冲突
            event._id = None  # type: ignore[attr-defined]

            replay_events.append(event)

        return replay_events
