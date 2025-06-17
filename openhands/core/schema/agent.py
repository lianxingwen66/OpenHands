"""
OpenHands Agent状态管理模块

技术栈:
- Python 3.12+ (核心语言)
- Enum - Python枚举类型，提供类型安全的状态定义
- str - 字符串混入，使枚举值可序列化

架构说明:
这个模块定义了Agent的生命周期状态，用于跟踪Agent的执行状态和控制流程。
状态管理是Agent系统的重要组成部分，确保系统的可预测性和可控性。

设计原则:
- 状态明确: 每个状态都有明确的含义和转换条件
- 可序列化: 状态可以在网络传输和持久化存储
- 类型安全: 使用枚举避免魔法字符串
- 扩展性: 支持添加新的状态类型
"""

from enum import Enum


class AgentState(str, Enum):
    """
    Agent状态枚举

    定义了Agent在执行过程中的所有可能状态，用于:
    1. 控制Agent的执行流程
    2. 用户界面状态显示
    3. 系统监控和调试
    4. 错误处理和恢复

    状态转换图:
    LOADING → RUNNING → [AWAITING_USER_INPUT | PAUSED | FINISHED | ERROR]
                    ↓
    [AWAITING_USER_CONFIRMATION] → [USER_CONFIRMED | USER_REJECTED]
                    ↓
    [STOPPED | REJECTED | RATE_LIMITED]
    """

    LOADING = 'loading'
    """
    Agent正在加载中

    初始状态，Agent正在进行以下操作:
    - 初始化LLM连接
    - 加载配置和prompt模板
    - 设置工具和插件
    - 准备运行时环境

    转换条件: 加载完成 → RUNNING
    """

    RUNNING = 'running'
    """
    Agent正在运行中

    Agent处于活跃执行状态，正在:
    - 处理用户指令
    - 执行Action并等待Observation
    - 与LLM进行交互
    - 分析和决策下一步操作

    转换条件:
    - 需要用户输入 → AWAITING_USER_INPUT
    - 任务完成 → FINISHED
    - 发生错误 → ERROR
    - 用户暂停 → PAUSED
    """

    AWAITING_USER_INPUT = 'awaiting_user_input'
    """
    等待用户输入

    Agent需要用户提供额外信息才能继续执行:
    - 需要澄清任务需求
    - 请求用户确认操作
    - 等待用户提供必要参数
    - 询问下一步指示

    转换条件: 用户提供输入 → RUNNING
    """

    PAUSED = 'paused'
    """
    Agent已暂停

    用户主动暂停了Agent的执行:
    - 用户点击暂停按钮
    - 系统资源不足时自动暂停
    - 等待外部条件满足

    转换条件: 用户恢复 → RUNNING
    """

    STOPPED = 'stopped'
    """
    Agent已停止

    Agent执行被终止:
    - 用户主动停止
    - 系统强制停止
    - 超时停止
    - 资源耗尽停止

    转换条件: 重新启动 → LOADING
    """

    FINISHED = 'finished'
    """
    任务已完成

    Agent成功完成了分配的任务:
    - 所有目标都已达成
    - Agent主动调用finish操作
    - 任务验证通过

    转换条件: 新任务开始 → LOADING
    """

    REJECTED = 'rejected'
    """
    Agent拒绝执行任务

    Agent因以下原因拒绝任务:
    - 任务超出能力范围
    - 任务违反安全策略
    - 任务描述不清晰
    - 缺少必要权限

    转换条件: 修改任务 → LOADING
    """

    ERROR = 'error'
    """
    执行过程中发生错误

    Agent遇到无法恢复的错误:
    - LLM API调用失败
    - 运行时环境异常
    - 工具执行错误
    - 系统资源问题

    转换条件: 错误恢复 → RUNNING 或重启 → LOADING
    """

    AWAITING_USER_CONFIRMATION = 'awaiting_user_confirmation'
    """
    等待用户确认

    Agent需要用户确认才能执行潜在危险操作:
    - 删除重要文件
    - 执行系统级命令
    - 访问敏感数据
    - 进行不可逆操作

    转换条件:
    - 用户确认 → USER_CONFIRMED
    - 用户拒绝 → USER_REJECTED
    """

    USER_CONFIRMED = 'user_confirmed'
    """
    用户已确认操作

    用户确认了Agent的操作请求:
    - 确认执行危险操作
    - 批准访问权限
    - 同意继续执行

    转换条件: 继续执行 → RUNNING
    """

    USER_REJECTED = 'user_rejected'
    """
    用户拒绝了操作

    用户拒绝了Agent的操作请求:
    - 拒绝执行危险操作
    - 不同意访问权限
    - 要求修改方案

    转换条件:
    - 重新规划 → RUNNING
    - 等待新指示 → AWAITING_USER_INPUT
    """

    RATE_LIMITED = 'rate_limited'
    """
    Agent受到速率限制

    由于API调用频率过高被限制:
    - LLM API速率限制
    - 外部服务限制
    - 系统资源保护

    转换条件: 等待限制解除 → RUNNING
    """
