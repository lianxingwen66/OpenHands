"""
OpenHands CodeAct Agent Bash工具模块

技术栈:
- Python 3.12+ (核心语言)
- LiteLLM - 函数调用工具定义
- Bash Shell - 命令执行环境
- 进程管理 - 长时间运行进程的处理
- 超时控制 - 命令执行时间限制

架构说明:
Bash工具是CodeActAgent的核心工具之一，提供了在持久化shell会话中
执行bash命令的能力。它支持复杂的进程管理、超时控制和交互式操作。

核心功能:
1. 命令执行 - 在持久化shell会话中执行bash命令
2. 进程管理 - 处理长时间运行的进程
3. 超时控制 - 软超时和硬超时机制
4. 交互式操作 - 与运行中的进程交互
5. 输出处理 - 处理和截断命令输出
6. 环境持久化 - 保持环境变量和工作目录

设计模式:
- 工厂模式: 根据需求创建不同描述详细程度的工具
- 策略模式: 不同的命令执行策略
- 状态模式: 进程状态管理
"""

import sys

# LiteLLM工具定义类型
from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

# 工具名称常量
from openhands.llm.tool_names import EXECUTE_BASH_TOOL_NAME

_DETAILED_BASH_DESCRIPTION = """Execute a bash command in the terminal within a persistent shell session.

### Command Execution
* One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, use `&&` or `;` to chain them together.
* Persistent session: Commands execute in a persistent shell session where environment variables, virtual environments, and working directory persist between commands.
* Timeout: Commands have a soft timeout of 10 seconds, once that's reached, you have the option to continue or interrupt the command (see section below for details)

### Running and Interacting with Processes
* Long running commands: For commands that may run indefinitely, run them in the background and redirect output to a file, e.g. `python3 app.py > server.log 2>&1 &`. For commands that need to run for a specific duration, like "sleep", you can set the "timeout" argument to specify a hard timeout in seconds.
* Interact with running process: If a bash command returns exit code `-1`, this means the process is not yet finished. By setting `is_input` to `true`, you can:
  - Send empty `command` to retrieve additional logs
  - Send text (set `command` to the text) to STDIN of the running process
  - Send control commands like `C-c` (Ctrl+C), `C-d` (Ctrl+D), or `C-z` (Ctrl+Z) to interrupt the process

### Best Practices
* Directory verification: Before creating new directories or files, first verify the parent directory exists and is the correct location.
* Directory management: Try to maintain working directory by using absolute paths and avoiding excessive use of `cd`.

### Output Handling
* Output truncation: If the output exceeds a maximum length, it will be truncated before being returned.
"""

_SHORT_BASH_DESCRIPTION = """Execute a bash command in the terminal.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`. For commands that need to run for a specific duration, you can set the "timeout" argument to specify a hard timeout in seconds.
* Interact with running process: If a bash command returns exit code `-1`, this means the process is not yet finished. By setting `is_input` to `true`, the assistant can interact with the running process and send empty `command` to retrieve any additional logs, or send additional text (set `command` to the text) to STDIN of the running process, or send command like `C-c` (Ctrl+C), `C-d` (Ctrl+D), `C-z` (Ctrl+Z) to interrupt the process.
* One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, you can use `&&` or `;` to chain them together."""


def refine_prompt(prompt: str):
    if sys.platform == 'win32':
        return prompt.replace('bash', 'powershell')
    return prompt


def create_cmd_run_tool(
    use_short_description: bool = False,
) -> ChatCompletionToolParam:
    description = (
        _SHORT_BASH_DESCRIPTION if use_short_description else _DETAILED_BASH_DESCRIPTION
    )
    return ChatCompletionToolParam(
        type='function',
        function=ChatCompletionToolParamFunctionChunk(
            name=EXECUTE_BASH_TOOL_NAME,
            description=refine_prompt(description),
            parameters={
                'type': 'object',
                'properties': {
                    'command': {
                        'type': 'string',
                        'description': refine_prompt(
                            'The bash command to execute. Can be empty string to view additional logs when previous exit code is `-1`. Can be `C-c` (Ctrl+C) to interrupt the currently running process. Note: You can only execute one bash command at a time. If you need to run multiple commands sequentially, you can use `&&` or `;` to chain them together.'
                        ),
                    },
                    'is_input': {
                        'type': 'string',
                        'description': refine_prompt(
                            'If True, the command is an input to the running process. If False, the command is a bash command to be executed in the terminal. Default is False.'
                        ),
                        'enum': ['true', 'false'],
                    },
                    'timeout': {
                        'type': 'number',
                        'description': 'Optional. Sets a hard timeout in seconds for the command execution. If not provided, the command will use the default soft timeout behavior.',
                    },
                },
                'required': ['command'],
            },
        ),
    )
