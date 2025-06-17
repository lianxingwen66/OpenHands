"""
OpenHands 循环检测器模块

这个模块实现了Agent循环检测功能，防止Agent陷入无限循环或重复执行相同的动作。
循环检测对于确保Agent的稳定性和效率至关重要。

技术栈：
- Python 类型提示 - 提供静态类型检查
- 模式识别算法 - 检测各种循环模式
- 事件分析 - 分析事件序列中的重复模式
- 状态机模式 - 管理检测状态和逻辑

核心概念：
1. 循环检测：识别Agent行为中的重复模式
2. 多种检测算法：支持不同类型的循环检测
3. 可配置参数：支持调整检测敏感度
4. 模式分类：区分不同类型的循环模式

检测场景：
- 动作-观察循环：重复执行相同的动作并得到相同的结果
- 动作-错误循环：重复执行导致错误的动作
- Agent独白：Agent重复发送相同的消息
- 上下文窗口错误：重复的上下文窗口溢出错误

设计特点：
- 智能识别：区分正常重复和异常循环
- 多模式支持：支持交互式和无头模式
- 性能优化：高效的模式匹配算法
- 可扩展性：易于添加新的检测算法
"""

from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger
from openhands.events.action.action import Action
from openhands.events.action.commands import IPythonRunCellAction
from openhands.events.action.empty import NullAction
from openhands.events.action.message import MessageAction
from openhands.events.event import Event, EventSource
from openhands.events.observation import (
    CmdOutputObservation,
    IPythonRunCellObservation,
)
from openhands.events.observation.agent import AgentCondensationObservation
from openhands.events.observation.empty import NullObservation
from openhands.events.observation.error import ErrorObservation
from openhands.events.observation.observation import Observation


class StuckDetector:
    """
    Agent循环检测器

    这个类负责检测Agent是否陷入各种类型的循环模式。它分析Agent的行为历史，
    识别重复的动作-观察模式，并在检测到循环时发出警告。

    技术架构：
    - 多算法检测：实现多种循环检测算法
    - 模式识别：识别不同类型的重复模式
    - 状态分析：基于Agent状态进行智能分析
    - 可配置检测：支持不同模式下的检测策略

    检测算法：
    1. 重复动作-观察检测：检测相同动作产生相同观察的循环
    2. 重复动作-错误检测：检测重复执行导致错误的动作
    3. Agent独白检测：检测Agent重复发送相同消息
    4. 动作-观察模式检测：检测交替的动作-观察模式
    5. 上下文窗口错误检测：检测重复的上下文窗口溢出

    设计原则：
    - 准确性：减少误报，确保检测的准确性
    - 效率性：高效的算法，不影响Agent性能
    - 可扩展性：易于添加新的检测算法
    - 可配置性：支持不同场景下的检测策略
    """

    # 语法错误消息模式列表
    # 用于检测Python代码执行中的重复语法错误
    SYNTAX_ERROR_MESSAGES = [
        'SyntaxError: unterminated string literal (detected at line',  # 未终止的字符串字面量
        'SyntaxError: invalid syntax. Perhaps you forgot a comma?',  # 无效语法，可能忘记逗号
        'SyntaxError: incomplete input',  # 不完整的输入
    ]

    def __init__(self, state: State):
        """
        初始化循环检测器

        Args:
            state (State): Agent的状态对象，包含历史记录和配置信息

        技术说明：
        - 状态绑定：检测器与特定的Agent状态绑定
        - 历史分析：基于状态中的历史记录进行分析
        - 动态检测：随着状态更新动态检测循环
        """
        self.state = state

    def is_stuck(self, headless_mode: bool = True) -> bool:
        """
        检查Agent是否陷入循环

        这是主要的循环检测方法，它分析Agent的历史记录，使用多种算法检测不同类型的循环模式。

        Args:
            headless_mode (bool): 匹配AgentController的headless_mode设置
                                True: 考虑所有历史记录（自动化/测试模式）
                                False: 只考虑最后一条用户消息之后的历史（交互模式）

        Returns:
            bool: True表示Agent陷入循环，False表示正常运行

        技术流程：
        1. 根据模式选择要分析的历史范围
        2. 过滤掉不相关的事件（用户消息、空事件）
        3. 检查历史长度是否足够进行循环检测
        4. 提取最近的动作和观察事件
        5. 依次应用各种循环检测算法
        6. 返回检测结果

        检测算法优先级：
        1. 重复动作-观察循环（最常见）
        2. 重复动作-错误循环（代码错误）
        3. Agent独白循环（消息重复）
        4. 动作-观察模式循环（复杂模式）
        5. 上下文窗口错误循环（资源限制）

        设计考虑：
        - 模式适应：支持交互式和无头模式的不同检测策略
        - 性能优化：最小历史长度要求，避免不必要的计算
        - 算法分层：从简单到复杂的检测算法
        - 早期退出：一旦检测到循环立即返回
        """
        if not headless_mode:
            # 在交互模式下，只查看最后一条用户消息之后的历史
            last_user_msg_idx = -1
            for i, event in enumerate(reversed(self.state.history)):
                if (
                    isinstance(event, MessageAction)
                    and event.source == EventSource.USER
                ):
                    last_user_msg_idx = len(self.state.history) - i - 1
                    break

            history_to_check = self.state.history[last_user_msg_idx + 1 :]
        else:
            # 在无头模式下，查看所有历史记录
            history_to_check = self.state.history

        # 过滤掉用户消息和空事件
        filtered_history = [
            event
            for event in history_to_check
            if not (
                # 过滤器在两种模式下都能优雅地工作：
                # - 在无头模式：主动从完整历史中过滤掉用户消息
                # - 在非无头模式：无操作，因为我们已经在最后一条用户消息之后切片
                (isinstance(event, MessageAction) and event.source == EventSource.USER)
                # 历史中可能有一些NullAction或NullObservation，至少目前是这样
                or isinstance(event, (NullAction, NullObservation))
            )
        ]

        # 检测循环至少需要3个动作，否则无需处理
        if len(filtered_history) < 3:
            return False

        # 前几个场景检测3或4个重复步骤
        # 准备最后4个动作和观察，以便检查它们
        last_actions: list[Event] = []
        last_observations: list[Event] = []

        # 从历史的末尾开始检索最后四个动作和观察，无论它们在哪里
        for event in reversed(filtered_history):
            if isinstance(event, Action) and len(last_actions) < 4:
                last_actions.append(event)
            elif isinstance(event, Observation) and len(last_observations) < 4:
                last_observations.append(event)

            if len(last_actions) == 4 and len(last_observations) == 4:
                break

        # 场景1：相同动作，相同观察
        if self._is_stuck_repeating_action_observation(last_actions, last_observations):
            return True

        # 场景2：相同动作，错误结果
        if self._is_stuck_repeating_action_error(last_actions, last_observations):
            return True

        # 场景3：Agent独白
        if self._is_stuck_monologue(filtered_history):
            return True

        # 场景4：最后六步的动作-观察模式
        if len(filtered_history) >= 6:
            if self._is_stuck_action_observation_pattern(filtered_history):
                return True

        # 场景5：上下文窗口错误循环
        if len(filtered_history) >= 10:
            if self._is_stuck_context_window_error(filtered_history):
                return True

        return False

    def _is_stuck_repeating_action_observation(
        self, last_actions: list[Event], last_observations: list[Event]
    ) -> bool:
        """
        检测重复的动作-观察循环

        场景1：相同动作，相同观察
        需要4个动作和4个观察来检测循环

        Args:
            last_actions: 最近的动作列表
            last_observations: 最近的观察列表

        Returns:
            bool: True表示检测到循环

        技术说明：
        - 需要4个完整的动作-观察对来确认循环
        - 使用_eq_no_pid方法比较事件，忽略进程ID等变化
        - 这是最常见的循环类型
        """
        # 场景1：相同动作，相同观察
        # 需要4个动作和4个观察来检测循环

        # 检查4个相同的动作-观察对的循环
        if len(last_actions) == 4 and len(last_observations) == 4:
            actions_equal = all(
                self._eq_no_pid(last_actions[0], action) for action in last_actions
            )
            observations_equal = all(
                self._eq_no_pid(last_observations[0], observation)
                for observation in last_observations
            )

            if actions_equal and observations_equal:
                logger.warning('Action, Observation loop detected')
                return True

        return False

    def _is_stuck_repeating_action_error(
        self, last_actions: list[Event], last_observations: list[Event]
    ) -> bool:
        # scenario 2: same action, errors
        # it takes 3 actions and 3 observations to detect a loop
        # check if the last three actions are the same and result in errors

        if len(last_actions) < 3 or len(last_observations) < 3:
            return False

        # are the last three actions the "same"?
        if all(self._eq_no_pid(last_actions[0], action) for action in last_actions[:3]):
            # and the last three observations are all errors?
            if all(isinstance(obs, ErrorObservation) for obs in last_observations[:3]):
                logger.warning('Action, ErrorObservation loop detected')
                return True
            # or, are the last three observations all IPythonRunCellObservation with SyntaxError?
            elif all(
                isinstance(obs, IPythonRunCellObservation)
                for obs in last_observations[:3]
            ):
                warning = 'Action, IPythonRunCellObservation loop detected'
                for error_message in self.SYNTAX_ERROR_MESSAGES:
                    if error_message.startswith(
                        'SyntaxError: unterminated string literal (detected at line'
                    ):
                        if self._check_for_consistent_line_error(
                            [
                                obs
                                for obs in last_observations[:3]
                                if isinstance(obs, IPythonRunCellObservation)
                            ],
                            error_message,
                        ):
                            logger.warning(warning)
                            return True
                    elif error_message in (
                        'SyntaxError: invalid syntax. Perhaps you forgot a comma?',
                        'SyntaxError: incomplete input',
                    ) and self._check_for_consistent_invalid_syntax(
                        [
                            obs
                            for obs in last_observations[:3]
                            if isinstance(obs, IPythonRunCellObservation)
                        ],
                        error_message,
                    ):
                        logger.warning(warning)
                        return True
        return False

    def _check_for_consistent_invalid_syntax(
        self, observations: list[IPythonRunCellObservation], error_message: str
    ) -> bool:
        first_lines = []
        valid_observations = []

        for obs in observations:
            content = obs.content
            lines = content.strip().split('\n')

            if len(lines) < 6:  # 6 because a real syntax error has at least 6 lines
                return False

            line1 = lines[0].strip()
            if not line1.startswith('Cell In[1], line'):
                return False

            first_lines.append(line1)  # Store the first line of each observation

            # Check last three lines
            if (
                lines[-1].startswith('[Jupyter Python interpreter:')
                and lines[-2].startswith('[Jupyter current working directory:')
                and error_message in lines[-3]
            ):
                valid_observations.append(obs)

        # Check if:
        # 1. All first lines are identical
        # 2. We have exactly 3 valid observations
        # 3. The error message line is identical in all valid observations
        return (
            len(set(first_lines)) == 1
            and len(valid_observations) == 3
            and len(
                set(
                    obs.content.strip().split('\n')[:-2][-1]
                    for obs in valid_observations
                )
            )
            == 1
        )

    def _check_for_consistent_line_error(
        self, observations: list[IPythonRunCellObservation], error_message: str
    ) -> bool:
        error_lines = []

        for obs in observations:
            content = obs.content
            lines = content.strip().split('\n')

            if len(lines) < 3:
                return False

            last_lines = lines[-3:]

            # Check if the last two lines are our own
            if not (
                last_lines[-2].startswith('[Jupyter current working directory:')
                and last_lines[-1].startswith('[Jupyter Python interpreter:')
            ):
                return False

            # Check for the error message in the 3rd-to-last line
            if error_message in last_lines[-3]:
                error_lines.append(last_lines[-3])

        # Check if we found the error message in all 3 observations
        # and the 3rd-to-last line is identical across all occurrences
        return len(error_lines) == 3 and len(set(error_lines)) == 1

    def _is_stuck_monologue(self, filtered_history: list[Event]) -> bool:
        # scenario 3: monologue
        # check for repeated MessageActions with source=AGENT
        # see if the agent is engaged in a good old monologue, telling itself the same thing over and over
        agent_message_actions = [
            (i, event)
            for i, event in enumerate(filtered_history)
            if isinstance(event, MessageAction) and event.source == EventSource.AGENT
        ]

        # last three message actions will do for this check
        if len(agent_message_actions) >= 3:
            last_agent_message_actions = agent_message_actions[-3:]

            if all(
                (last_agent_message_actions[0][1] == action[1])
                for action in last_agent_message_actions
            ):
                # check if there are any observations between the repeated MessageActions
                # then it's not yet a loop, maybe it can recover
                start_index = last_agent_message_actions[0][0]
                end_index = last_agent_message_actions[-1][0]

                has_observation_between = False
                for event in filtered_history[start_index + 1 : end_index]:
                    if isinstance(event, Observation):
                        has_observation_between = True
                        break

                if not has_observation_between:
                    logger.warning('Repeated MessageAction with source=AGENT detected')
                    return True
        return False

    def _is_stuck_action_observation_pattern(
        self, filtered_history: list[Event]
    ) -> bool:
        # scenario 4: action, observation pattern on the last six steps
        # check if the agent repeats the same (Action, Observation)
        # every other step in the last six steps
        last_six_actions: list[Event] = []
        last_six_observations: list[Event] = []

        # the end of history is most interesting
        for event in reversed(filtered_history):
            if isinstance(event, Action) and len(last_six_actions) < 6:
                last_six_actions.append(event)
            elif isinstance(event, Observation) and len(last_six_observations) < 6:
                last_six_observations.append(event)

            if len(last_six_actions) == 6 and len(last_six_observations) == 6:
                break

        # this pattern is every other step, like:
        # (action_1, obs_1), (action_2, obs_2), (action_1, obs_1), (action_2, obs_2),...
        if len(last_six_actions) == 6 and len(last_six_observations) == 6:
            actions_equal = (
                # action_0 == action_2 == action_4
                self._eq_no_pid(last_six_actions[0], last_six_actions[2])
                and self._eq_no_pid(last_six_actions[0], last_six_actions[4])
                # action_1 == action_3 == action_5
                and self._eq_no_pid(last_six_actions[1], last_six_actions[3])
                and self._eq_no_pid(last_six_actions[1], last_six_actions[5])
            )
            observations_equal = (
                # obs_0 == obs_2 == obs_4
                self._eq_no_pid(last_six_observations[0], last_six_observations[2])
                and self._eq_no_pid(last_six_observations[0], last_six_observations[4])
                # obs_1 == obs_3 == obs_5
                and self._eq_no_pid(last_six_observations[1], last_six_observations[3])
                and self._eq_no_pid(last_six_observations[1], last_six_observations[5])
            )

            if actions_equal and observations_equal:
                logger.warning('Action, Observation pattern detected')
                return True
        return False

    def _is_stuck_context_window_error(self, filtered_history: list[Event]) -> bool:
        """Detects if we're stuck in a loop of context window errors.

        This happens when we repeatedly get context window errors and try to trim,
        but the trimming doesn't work, causing us to get more context window errors.
        The pattern is repeated AgentCondensationObservation events without any other
        events between them.

        Args:
            filtered_history: List of filtered events to check

        Returns:
            bool: True if we detect a context window error loop
        """
        # Look for AgentCondensationObservation events
        condensation_events = [
            (i, event)
            for i, event in enumerate(filtered_history)
            if isinstance(event, AgentCondensationObservation)
        ]

        # Need at least 10 condensation events to detect a loop
        if len(condensation_events) < 10:
            return False

        # Get the last 10 condensation events
        last_condensation_events = condensation_events[-10:]

        # Check if there are any non-condensation events between them
        for i in range(len(last_condensation_events) - 1):
            start_idx = last_condensation_events[i][0]
            end_idx = last_condensation_events[i + 1][0]

            # Look for any non-condensation events between these two
            has_other_events = False
            for event in filtered_history[start_idx + 1 : end_idx]:
                if not isinstance(event, AgentCondensationObservation):
                    has_other_events = True
                    break

            if not has_other_events:
                logger.warning(
                    'Context window error loop detected - repeated condensation events'
                )
                return True

        return False

    def _eq_no_pid(self, obj1: Event, obj2: Event) -> bool:
        if isinstance(obj1, IPythonRunCellAction) and isinstance(
            obj2, IPythonRunCellAction
        ):
            # for loop detection on edit actions, ignore the thought, compare some code
            # the code should have at least 3 lines, to avoid simple one-liners
            if (
                'edit_file_by_replace(' in obj1.code
                and 'edit_file_by_replace(' in obj2.code
            ):
                return (
                    len(obj1.code.split('\n')) > 2
                    and obj1.code.split('\n')[:3] == obj2.code.split('\n')[:3]
                )
            else:
                # default comparison
                return obj1 == obj2
        elif isinstance(obj1, CmdOutputObservation) and isinstance(
            obj2, CmdOutputObservation
        ):
            # for loop detection, ignore command_id, which is the pid
            return obj1.command == obj2.command and obj1.exit_code == obj2.exit_code
        else:
            # this is the default comparison
            return obj1 == obj2
