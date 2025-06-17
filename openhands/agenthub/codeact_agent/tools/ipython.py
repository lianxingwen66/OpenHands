"""
OpenHands CodeAct Agent IPython工具模块

技术栈:
- Python 3.12+ (核心语言)
- IPython - 交互式Python环境
- Jupyter - Notebook执行环境
- LiteLLM - 函数调用工具定义
- Magic Commands - IPython魔法命令支持

架构说明:
IPython工具提供了在隔离的IPython环境中执行Python代码的能力。
这个工具特别适合数据分析、科学计算和快速原型开发。

核心功能:
1. Python代码执行 - 在IPython环境中执行代码
2. 变量管理 - 管理IPython会话中的变量
3. 包导入 - 支持动态包导入和安装
4. 魔法命令 - 支持IPython魔法命令（如%pip）
5. 环境隔离 - IPython环境与终端环境隔离
6. 结果展示 - 格式化显示执行结果

特性:
- 支持所有Python语法和库
- 支持matplotlib等可视化库
- 支持pandas等数据处理库
- 支持机器学习库
- 支持魔法命令进行环境管理

设计模式:
- 单例模式: IPython环境的单一实例
- 命令模式: 代码执行作为命令
- 观察者模式: 执行结果的观察和处理
"""

# LiteLLM工具定义类型
from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

# IPython工具描述
_IPYTHON_DESCRIPTION = """Run a cell of Python code in an IPython environment.
* The assistant should define variables and import packages before using them.
* The variable defined in the IPython environment will not be available outside the IPython environment (e.g., in terminal).
"""

# IPython工具定义
IPythonTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='execute_ipython_cell',
        description=_IPYTHON_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'code': {
                    'type': 'string',
                    'description': 'The Python code to execute. Supports magic commands like %pip.',
                },
            },
            'required': ['code'],
        },
    ),
)
