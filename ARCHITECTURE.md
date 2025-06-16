# OpenHands 架构文档

## 整体技术栈

### 后端技术栈
- **Python 3.12+**: 主要编程语言
- **FastAPI**: Web框架，用于REST API服务
- **Pydantic**: 数据验证和序列化
- **SQLAlchemy**: ORM数据库操作
- **LiteLLM**: 多LLM提供商统一接口
- **Docker**: 容器化运行时环境
- **Poetry**: Python依赖管理
- **Pytest**: 单元测试框架
- **AsyncIO**: 异步编程支持
- **WebSocket**: 实时通信
- **Jinja2**: 模板引擎（用于prompt管理）

### 前端技术栈
- **React 18**: 前端框架
- **TypeScript**: 类型安全的JavaScript
- **Vite**: 构建工具和开发服务器
- **TanStack Query**: 数据获取和缓存管理
- **Tailwind CSS**: 原子化CSS框架
- **React Router**: 客户端路由
- **Vitest**: 前端测试框架
- **Playwright**: 端到端测试

### 基础设施技术栈
- **Docker Compose**: 多容器编排
- **E2B**: 安全代码执行沙箱
- **Modal**: 云计算平台集成
- **GitHub Actions**: CI/CD流水线
- **Pre-commit**: 代码质量检查

## 核心架构组件

### 1. Agent系统 (`openhands/controller/`)
```
Agent (抽象基类)
├── CodeActAgent        # 主要实现，基于代码执行
├── BrowsingAgent       # 网页浏览专用
├── ReadonlyAgent       # 只读模式
└── LocAgent           # 基于位置的操作
```

### 2. 事件系统 (`openhands/events/`)
```
Event (基类)
├── Action             # Agent发出的指令
│   ├── CmdRunAction   # 执行bash命令
│   ├── IPythonRunCellAction # 执行Python代码
│   ├── MessageAction  # 消息通信
│   └── AgentFinishAction # 任务完成
└── Observation        # 环境反馈
    ├── CmdRunObservation # 命令执行结果
    ├── ErrorObservation  # 错误信息
    └── SuccessObservation # 成功信息
```

### 3. 运行时系统 (`openhands/runtime/`)
```
Runtime (抽象基类)
├── LocalRuntime       # 本地执行环境
├── DockerRuntime      # Docker容器环境
├── E2BRuntime         # E2B沙箱环境
└── RemoteRuntime      # 远程执行环境
```

### 4. LLM集成 (`openhands/llm/`)
```
LLM (统一接口)
├── OpenAI GPT系列
├── Anthropic Claude
├── Google Gemini
├── 本地模型 (Ollama, vLLM)
└── Azure OpenAI
```

### 5. 内存管理 (`openhands/memory/`)
```
Memory System
├── ConversationMemory  # 对话历史管理
├── Condenser          # 历史压缩
│   ├── Truncation     # 简单截断策略
│   └── Summarization  # 智能摘要策略
└── View               # 内存视图管理
```

### 6. 微代理系统 (`openhands/microagent/`)
```
Microagent (专门化prompt)
├── KnowledgeMicroagent  # 知识型微代理
├── RepoMicroagent      # 仓库特定微代理
└── TaskMicroagent      # 任务型微代理
```

### 7. 服务器系统 (`openhands/server/`)
```
Server Components
├── FastAPI Application  # 主Web服务
├── WebSocket Handler   # 实时通信
├── Session Management  # 会话管理
├── Authentication     # 用户认证
└── File Management    # 文件操作API
```

### 8. 前端架构 (`frontend/src/`)
```
Frontend Structure
├── components/        # React组件
├── hooks/            # 自定义Hook
│   ├── query/        # TanStack Query hooks
│   └── mutation/     # 数据变更hooks
├── api/              # API客户端
├── types/            # TypeScript类型定义
├── routes/           # 页面路由组件
└── i18n/             # 国际化支持
```

## 数据流架构

### 1. 用户交互流程
```
用户输入 → Frontend → WebSocket → Server → AgentController → Agent → Action → Runtime → Observation → Agent → Response → Frontend
```

### 2. Agent执行循环
```
State → Agent.step() → Action → Runtime.execute() → Observation → State.update() → Agent.step() → ...
```

### 3. 内存管理流程
```
Event Stream → Condenser → Compressed History → ConversationMemory → LLM Context
```

## 安全架构

### 1. 沙箱隔离
- Docker容器隔离
- E2B安全沙箱
- 文件系统权限控制
- 网络访问限制

### 2. 代码执行安全
- 命令白名单/黑名单
- 超时控制
- 资源使用限制
- 恶意代码检测

### 3. 用户数据保护
- 会话隔离
- 数据加密
- 访问控制
- 审计日志

## 扩展性设计

### 1. 插件系统
- Agent插件注册机制
- Runtime插件支持
- 工具扩展接口
- 微代理扩展

### 2. 配置管理
- 分层配置系统
- 环境变量支持
- 动态配置更新
- 配置验证

### 3. 监控和日志
- 结构化日志
- 性能监控
- 错误追踪
- 使用统计

## 部署架构

### 1. 开发环境
```
Docker Compose
├── openhands-app      # 主应用容器
├── openhands-frontend # 前端开发服务器
└── sandbox           # 沙箱运行时
```

### 2. 生产环境
```
Kubernetes/Docker Swarm
├── Load Balancer     # 负载均衡
├── App Instances     # 应用实例
├── Database         # 数据存储
└── Sandbox Pool     # 沙箱资源池
```

## 性能优化

### 1. 前端优化
- 代码分割和懒加载
- TanStack Query缓存
- 虚拟滚动
- 图片优化

### 2. 后端优化
- 异步处理
- 连接池管理
- 缓存策略
- 数据库优化

### 3. 内存优化
- 历史压缩
- 垃圾回收
- 对象池
- 流式处理

这个架构文档为后续的代码注释提供了整体框架参考。
