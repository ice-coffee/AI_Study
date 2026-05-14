# LangChain API 示例

本目录包含 LangChain 框架的 API 使用示例，涵盖 LangChain 的核心功能模块。

## 环境配置

运行示例前，需要设置以下环境变量：

```bash
export ONE_API_KEY="your-api-key"
export BASE_URL="your-base-url"  # 可选，用于兼容 OpenAI API 的服务
```

## 示例列表

| 文件 | 说明 |
|------|------|
| `01_chat_models.py` | 聊天模型调用 - 基础对话、流式输出、批量调用、异步调用 |
| `02_prompts.py` | 提示词模板 - PromptTemplate、ChatPromptTemplate、Few-shot |
| `03_output_parsers.py` | 输出解析器 - JSON 解析、Pydantic 模型解析、列表解析 |
| `04_chains.py` | 链式调用 - LCEL 语法、并行链、流水线构建 |
| `05_rag.py` | RAG 检索增强生成 - 文档加载、向量存储、语义检索 |
| `06_agents.py` | 智能体 - 工具调用、ReAct 模式、自主决策 |
| `07_memory.py` | 对话记忆 - 消息历史、会话管理、滑动窗口 |
| `08_embeddings.py` | 向量嵌入 - 文本向量化、语义搜索、相似度计算 |
| `09_tools.py` | 工具定义 - @tool 装饰器、StructuredTool、参数验证 |

## 运行示例

```bash
# 安装依赖
uv sync

# 运行单个示例
uv run python 01_chat_models.py
```

## 依赖说明

```
langchain>=1.3.0          # LangChain 核心库
langchain-openai          # OpenAI 集成
langchain-community       # 社区集成（向量存储等）
pydantic                  # 数据验证
faiss-cpu                 # 向量存储（可选）
```

## 核心概念

### LCEL (LangChain Expression Language)

LangChain 使用 `|` 操作符连接组件，构建处理流水线：

```python
chain = prompt | llm | parser
result = chain.invoke({"topic": "Python"})
```

### Runnable 接口

所有组件都实现了 Runnable 接口，支持：
- `invoke()`: 单个输入
- `batch()`: 批量输入
- `stream()`: 流式输出
- `ainvoke()`: 异步调用

### 消息类型

- `SystemMessage`: 系统指令
- `HumanMessage`: 用户消息
- `AIMessage`: AI 回复
- `ToolMessage`: 工具调用结果
