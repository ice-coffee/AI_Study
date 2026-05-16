---

# Agents

Agents 将语言模型与[工具](/oss/python/langchain/tools)结合，创建能够推理任务、决定使用哪些工具并迭代地寻求解决方案的系统。

[`create_agent`](https://reference.langchain.com/python/langchain/agents/factory/create_agent) 提供了生产就绪的 agent 实现。

[LLM Agent 通过在循环中运行工具来实现目标](https://simonwillison.net/2025/Sep/18/agents/)。
Agent 会持续运行，直到满足停止条件——即模型发出最终输出或达到迭代限制。

```mermaid theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
%%{
  init: {
    "fontFamily": "monospace",
    "flowchart": {
      "curve": "curve"
    }
  }
}%%
graph TD
  %% Outside the agent
  QUERY([input])
  LLM{model}
  TOOL(tools)
  ANSWER([output])

  %% Main flows (no inline labels)
  QUERY --> LLM
  LLM --"action"--> TOOL
  TOOL --"observation"--> LLM
  LLM --"finish"--> ANSWER

  classDef blueHighlight fill:#E5F4FF,stroke:#006DDD,color:#030710;
  classDef greenHighlight fill:#F6FFDB,stroke:#6E8900,color:#2E3900;
  class QUERY blueHighlight;
  class ANSWER blueHighlight;
  class LLM greenHighlight;
  class TOOL greenHighlight;
```

<Info>
  [`create_agent`](https://reference.langchain.com/python/langchain/agents/factory/create_agent) 使用 [LangGraph](/oss/python/langgraph/overview) 构建基于**图**的 agent 运行时。图由节点（步骤）和边（连接）组成，定义了 agent 如何处理信息。Agent 在图中移动，执行如模型节点（调用模型）、工具节点（执行工具）或中间件等节点。

  了解更多关于 [Graph API](/oss/python/langgraph/graph-api) 的信息。
</Info>

<Tip>
  使用 [LangSmith](https://smith.langchain.com?utm_source=docs\\&utm_medium=cta\\&utm_campaign=langsmith-signup\\&utm_content=oss-langchain-agents) 跟踪循环中的每个步骤、调试工具调用并评估 agent 输出。按照 [tracing quickstart](/langsmith/trace-with-langchain) 进行设置。
</Tip>

## 核心组件

### Model

[model](/oss/python/langchain/models) 是 agent 的推理引擎。可以通过多种方式指定，支持静态和动态模型选择。

#### Static model（静态模型）

静态模型在创建 agent 时配置一次，并在整个执行过程中保持不变。这是最常见和直接的方法。

要从 <Tooltip tip="遵循格式 `provider:model` 的字符串（例如 openai:gpt-5）" cta="查看映射" href="https://reference.langchain.com/python/langchain/models/#langchain.chat_models.init_chat_model(model)">模型标识符字符串</Tooltip>初始化静态模型：

```python wrap theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent

agent = create_agent("openai:gpt-5.4", tools=tools)
```

<Tip>
  模型标识符字符串支持自动推断（例如，`"gpt-5.4"` 将被推断为 `"openai:gpt-5.4"`）。请参阅[参考文档](https://reference.langchain.com/python/langchain/chat_models/base/init_chat_model)以查看模型标识符字符串映射的完整列表。
</Tip>

要更精细地控制模型配置，请使用提供者包直接初始化模型实例。在此示例中，我们使用 [`ChatOpenAI`](https://reference.langchain.com/python/langchain-openai/chat_models/base/ChatOpenAI)。请参阅 [Chat models](/oss/python/integrations/chat)以了解其他可用的聊天模型类。

```python wrap theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5.4",
    temperature=0.1,
    max_tokens=1000,
    timeout=30
    # ... (其他参数)
)
agent = create_agent(model, tools=tools)
```

模型实例为您提供了完整的配置控制权。当需要设置特定的[参数](/oss/python/langchain/models#parameters)（如 `temperature`、`max_tokens`、`timeouts`、`base_url` 和其他提供者特定的设置）时，请使用它们。请参阅[参考文档](https://reference.langchain.com/python/langchain/integrations/providers/all_providers)以查看模型上可用的参数和方法。

#### Dynamic model（动态模型）

动态模型根据当前的<Tooltip tip="agent 的执行环境，包含不可变配置和在整个 agent 执行期间持续的上下文数据（例如，用户 ID、会话详细信息或特定于应用程序的配置）。">运行时</Tooltip>状态和上下文进行选择。这可以实现复杂的路由逻辑和成本优化。

要使用动态模型，请使用 [`@wrap_model_call`](https://reference.langchain.com/python/langchain/agents/middleware/types/wrap_model_call) 装饰器创建中间件，该装饰器修改请求中的模型：

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse


basic_model = ChatOpenAI(model="gpt-5.4-mini")
advanced_model = ChatOpenAI(model="gpt-5.4")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """根据对话复杂度选择模型。"""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # 对于较长的对话使用高级模型
        model = advanced_model
    else:
        model = basic_model

    return handler(request.override(model=model))

agent = create_agent(
    model=basic_model,  # 默认模型
    tools=tools,
    middleware=[dynamic_model_selection]
)
```

<Warning>
  使用结构化输出时，不支持预绑定模型（已调用 [`bind_tools`](https://reference.langchain.com/python/langchain-core/language_models/chat_models/BaseChatModel/bind_tools) 的模型）。如果需要结构化输出的动态模型选择，请确保传递给中间件的模型未预绑定。
</Warning>

<Tip>
  有关模型配置详细信息，请参阅 [Models](/oss/python/langchain/models)。有关动态模型选择模式，请参阅 [Dynamic model in middleware](/oss/python/langchain/middleware#dynamic-model)。
</Tip>

### Tools

工具赋予 agent 采取行动的能力。Agent 通过以下方式超越了仅绑定模型的简单工具调用：

* 按顺序进行多次工具调用（由单个提示触发）
* 在适当时并行调用工具
* 基于先前的结果动态选择工具
* 工具重试逻辑和错误处理
* 跨工具调用的状态持久化

有关更多信息，请参阅 [Tools](/oss/python/langchain/tools)。

#### Static tools（静态工具）

静态工具在创建 agent 时定义，并在整个执行过程中保持不变。这是最常见和直接的方法。

要使用静态工具定义 agent，请将工具列表传递给 agent。

<Tip>
  工具可以指定为普通 Python 函数或<Tooltip tip="可以暂停执行并在稍后恢复的方法">协程</Tooltip>。

  [工具装饰器](/oss/python/langchain/tools#create-tools)可用于自定义工具名称、描述、参数架构和其他属性。
</Tip>

```python wrap theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.tools import tool
from langchain.agents import create_agent


@tool
def search(query: str) -> str:
    """搜索信息。"""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """获取位置的天气信息。"""
    return f"Weather in {location}: Sunny, 72°F"

agent = create_agent(model, tools=[search, get_weather])
```

如果提供了空工具列表，agent 将由单个 LLM 节点组成，而不具备工具调用能力。

#### Dynamic tools（动态工具）

使用动态工具时，agent 可用的工具集在运行时被修改，而不是预先全部定义。并非每个工具都适用于每种情况。太多的工具可能会使模型不堪重负（过载上下文）并增加错误；太少的工具会限制能力。动态工具选择能够基于身份验证状态、用户权限、功能标志或对话阶段来适应可用的工具集。

有两种方法取决于工具是否提前已知：

<Tabs>
  <Tab title="过滤预注册工具">
    当在创建 agent 时知道所有可能的工具时，您可以预先注册它们，并根据状态、权限或上下文动态过滤暴露给模型的工具。

    <Tabs>
      <Tab title="State">
        仅在达到某些对话里程碑后启用高级工具：

        ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
        from langchain.agents import create_agent
        from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
        from typing import Callable

        @wrap_model_call
        def state_based_tools(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            """基于对话 State 过滤工具。"""
            # 从 State 读取：检查用户是否已身份验证
            state = request.state
            is_authenticated = state.get("authenticated", False)
            message_count = len(state["messages"])

            # 仅在身份验证后启用敏感工具
            if not is_authenticated:
                tools = [t for t in request.tools if t.name.startswith("public_")]
                request = request.override(tools=tools)
            elif message_count < 5:
                # 在对话早期限制工具
                tools = [t for t in request.tools if t.name != "advanced_search"]
                request = request.override(tools=tools)

            return handler(request)

        agent = create_agent(
            model="gpt-5.4",
            tools=[public_search, private_search, advanced_search],
            middleware=[state_based_tools]
        )
        ```
      </Tab>

      <Tab title="Store">
        根据用户偏好或 Store 中的功能标志过滤工具：

        ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
        from dataclasses import dataclass
        from langchain.agents import create_agent
        from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
        from typing import Callable
        from langgraph.store.memory import InMemoryStore

        @dataclass
        class Context:
            user_id: str

        @wrap_model_call
        def store_based_tools(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            """基于 Store 偏好过滤工具。"""
            user_id = request.runtime.context.user_id

            # 从 Store 读取：获取用户的启用功能
            store = request.runtime.store
            feature_flags = store.get(("features",), user_id)

            if feature_flags:
                enabled_features = feature_flags.value.get("enabled_tools", [])
                # 仅包含为此用户启用的工具
                tools = [t for t in request.tools if t.name in enabled_features]
                request = request.override(tools=tools)

            return handler(request)

        agent = create_agent(
            model="gpt-5.4",
            tools=[search_tool, analysis_tool, export_tool],
            middleware=[store_based_tools],
            context_schema=Context,
            store=InMemoryStore()
        )
        ```
      </Tab>

      <Tab title="Runtime Context">
        根据 Runtime Context 中的用户权限过滤工具：

        ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
        from dataclasses import dataclass
        from langchain.agents import create_agent
        from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
        from typing import Callable

        @dataclass
        class Context:
            user_role: str

        @wrap_model_call
        def context_based_tools(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            """基于 Runtime Context 权限过滤工具。"""
            # 从 Runtime Context 读取：获取用户角色
            if request.runtime is None or request.runtime.context is None:
                # 如果未提供上下文，默认为 viewer（最严格）
                user_role = "viewer"
            else:
                user_role = request.runtime.context.user_role

            if user_role == "admin":
                # 管理员拥有所有工具
                pass
            elif user_role == "editor":
                # 编辑者不能删除
                tools = [t for t in request.tools if t.name != "delete_data"]
                request = request.override(tools=tools)
            else:
                # 查看者仅获得只读工具
                tools = [t for t in request.tools if t.name.startswith("read_")]
                request = request.override(tools=tools)

            return handler(request)

        agent = create_agent(
            model="gpt-5.4",
            tools=[read_data, write_data, delete_data],
            middleware=[context_based_tools],
            context_schema=Context
        )
        ```
      </Tab>
    </Tabs>

    此方法最适合于：

    * 在编译/启动时知道所有可能的工具
    * 您想要基于权限、功能标志或对话状态进行过滤
    * 工具是静态的，但其可用性是动态的

    有关更多示例，请参阅 [Dynamically selecting tools](/oss/python/langchain/middleware/custom#dynamically-selecting-tools)。
  </Tab>

  <Tab title="运行时工具注册">
    当工具在运行时被发现或创建时（例如，从 MCP 服务器加载、基于用户数据生成或从远程注册表获取），您需要同时注册工具并动态处理其执行。

    这需要两个中间件钩子：

    1. `wrap_model_call` - 将动态工具添加到请求中
    2. `wrap_tool_call` - 处理动态添加的工具的执行

    ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    from langchain.tools import tool
    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentMiddleware, ModelRequest, ToolCallRequest

    # 将在运行时动态添加的工具
    @tool
    def calculate_tip(bill_amount: float, tip_percentage: float = 20.0) -> str:
        """计算账单的小费金额。"""
        tip = bill_amount * (tip_percentage / 100)
        return f"Tip: ${tip:.2f}, Total: ${bill_amount + tip:.2f}"

    class DynamicToolMiddleware(AgentMiddleware):
        """注册和处理动态工具的中间件。"""

        def wrap_model_call(self, request: ModelRequest, handler):
            # 将动态工具添加到请求中
            # 这可以从 MCP 服务器、数据库等加载
            updated = request.override(tools=[*request.tools, calculate_tip])
            return handler(updated)

        def wrap_tool_call(self, request: ToolCallRequest, handler):
            # 处理动态工具的执行
            if request.tool_call["name"] == "calculate_tip":
                return handler(request.override(tool=calculate_tip))
            return handler(request)

    agent = create_agent(
        model="gpt-4o",
        tools=[get_weather],  # 此处仅注册静态工具
        middleware=[DynamicToolMiddleware()],
    )

    # agent 现在可以同时使用 get_weather 和 calculate_tip
    result = agent.invoke({
        "messages": [{"role": "user", "content": "计算 $85 的 20% 小费"}]
    })
    ```

    此方法最适合于：

    * 在运行时发现工具（例如，从 MCP 服务器）
    * 基于用户数据或配置动态生成工具
    * 与外部工具注册表集成

    <Note>
      对于运行时注册的工具，`wrap_tool_call` 钩子是必需的，因为 agent 需要知道如何执行原始工具列表中没有的工具。没有它，agent 将不知道如何调用动态添加的工具。
    </Note>
  </Tab>
</Tabs>

<Tip>
  要了解有关工具的更多信息，请参阅 [Tools](/oss/python/langchain/tools)。
</Tip>

#### Tool error handling（工具错误处理）

要自定义工具错误的处理方式，请使用 [`@wrap_tool_call`](https://reference.langchain.com/python/langchain/agents/middleware/types/wrap_tool_call) 装饰器创建中间件：

```python wrap theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage


@wrap_tool_call
def handle_tool_errors(request, handler):
    """使用自定义消息处理工具执行错误。"""
    try:
        return handler(request)
    except Exception as e:
        # 向模型返回自定义错误消息
        return ToolMessage(
            content=f"Tool error: 请检查您的输入并重试。({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model="gpt-5.4",
    tools=[search, get_weather],
    middleware=[handle_tool_errors]
)
```

当工具失败时，agent 将返回包含自定义错误消息的 [`ToolMessage`](https://reference.langchain.com/python/langchain-core/messages/tool/ToolMessage)：

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
[
    ...
    ToolMessage(
        content="Tool error: 请检查您的输入并重试。(division by zero)",
        tool_call_id="..."
    ),
    ...
]
```

#### Tool use in the ReAct loop（ReAct 循环中的工具使用）

Agent 遵循 ReAct（"推理 + 行动"）模式，在简短推理步骤与针对性工具调用之间交替，并将结果观察反馈给后续决策，直到能够提供最终答案。

<Accordion title="ReAct 循环示例">
  **提示词：** 识别当前最受欢迎的无线耳机并验证可用性。

  ```
  ================================ Human Message =================================

  Find the most popular wireless headphones right now and check if they're in stock
  ```

  * **推理**："流行度是时间敏感的，我需要使用提供的搜索工具。"
  * **行动**：调用 `search_products("wireless headphones")`

  ```
  ================================== Ai Message ==================================
  Tool Calls:
    search_products (call_abc123)
   Call ID: call_abc123
    Args:
      query: wireless headphones
  ```

  ```
  ================================= Tool Message =================================

  Found 5 products matching "wireless headphones". Top 5 results: WH-1000XM5, ...
  ```

  * **推理**："我需要在回答之前确认排名靠前的项目的可用性。"
  * **行动**：调用 `check_inventory("WH-1000XM5")`

  ```
  ================================== Ai Message ==================================
  Tool Calls:
    check_inventory (call_def456)
   Call ID: call_def456
    Args:
      product_id: WH-1000XM5
  ```

  ```
  ================================= Tool Message =================================

  Product WH-1000XM5: 10 units in stock
  ```

  * **推理**："我已经获得了最受欢迎的模型及其库存状态。现在我可以回答用户的问题。"
  * **行动**：生成最终答案

  ```
  ================================== Ai Message ==================================

  I found wireless headphones (model WH-1000XM5) with 10 units in stock...
  ```
</Accordion>

### System prompt（系统提示词）

您可以通过提供提示词来塑造 agent 处理任务的方式。[`system_prompt`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent\(system_prompt\)) 参数可以作为字符串提供：

```python wrap theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
agent = create_agent(
    model,
    tools,
    system_prompt="你是一个有帮助的助手。简明准确。"
)
```

当未提供 [`system_prompt`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent\(system_prompt\)) 时，agent 将直接从消息中推断其任务。

[`system_prompt`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent\(system_prompt\)) 参数接受 `str` 或 [`SystemMessage`](https://reference.langchain.com/python/langchain-core/messages/system/SystemMessage)。使用 `SystemMessage` 可以让您更好地控制提示词结构，这对于提供者特定功能（如 [Anthropic 的提示词缓存](/oss/python/integrations/chat/anthropic#prompt-caching)）很有用：

```python wrap theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage

literary_agent = create_agent(
    model="google_genai:gemini-3.1-pro-preview",
    system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text": "你是一个负责分析文学作品的 AI 助手。",
            },
            {
                "type": "text",
                "text": "<the entire contents of 'Pride and Prejudice'>",
                "cache_control": {"type": "ephemeral"}
            }
        ]
    )
)

result = literary_agent.invoke(
    {"messages": [HumanMessage("分析'傲慢与偏见'的主要主题。")]}
)
```

`cache_control` 字段设置为 `{"type": "ephemeral"}` 会告诉 Anthropic 缓存该内容块，从而减少重复请求的延迟和成本，这些重复请求使用相同的系统提示词。

#### Dynamic system prompt（动态系统提示词）

对于需要根据运行时上下文或 agent 状态修改系统提示词的高级用例，您可以使用 [中间件](/oss/python/langchain/middleware)。

[`@dynamic_prompt`](https://reference.langchain.com/python/langchain/agents/middleware/types/dynamic_prompt) 装饰器创建基于模型请求生成系统提示词的中间件：

```python wrap theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest


class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """基于用户角色生成系统提示词。"""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "你是一个有帮助的助手。"

    if user_role == "expert":
        return f"{base_prompt} 提供详细的技术响应。"
    elif user_role == "beginner":
        return f"{base_prompt} 简单解释概念，避免使用行话。"

    return base_prompt

agent = create_agent(
    model="gpt-5.4",
    tools=[web_search],
    middleware=[user_role_prompt],
    context_schema=Context
)

# 系统提示词将基于上下文动态设置
result = agent.invoke(
    {"messages": [{"role": "user", "content": "解释机器学习"}]},
    context={"user_role": "expert"}
)
```

<Tip>
  有关消息类型和格式的更多详细信息，请参阅 [Messages](/oss/python/langchain/messages)。有关全面的中间件文档，请参阅 [Middleware](/oss/python/langchain/middleware)。
</Tip>

### Name（名称）

为 agent 设置可选的 [`name`](https://reference.langchain.com/python/langchain/agents/factory/create_agent)。这用作在[多 agent 系统](/oss/python/langchain/multi-agent)中将 agent 添加为子图时的节点标识符：

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
agent = create_agent(
    model,
    tools,
    name="research_assistant"
)
```

<Warning>
  首选使用 `snake_case` 作为 agent 名称（例如，使用 `research_assistant` 而不是 `Research Assistant`）。某些模型提供者会拒绝包含空格或特殊字符的名称并报错。仅使用字母数字字符、下划线和连字符可确保与所有提供者的兼容性。这同样适用于[工具名称](/oss/python/langchain/tools)。
</Warning>

## Invocation（调用）

您可以通过向其 [`State`](/oss/python/langgraph/graph-api#state) 传递更新来调用 agent。所有 agent 在其状态中都包含一个[消息序列](/oss/python/langgraph/use-graph-api#messagesstate)；要调用 agent，请传递新消息：

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "旧金山的天气怎么样？"}]}
)
```

要从 agent 流式传输步骤和/或令牌，请参阅 [streaming](/oss/python/langchain/streaming) 指南。

否则，agent 遵循 LangGraph [Graph API](/oss/python/langgraph/use-graph-api) 并支持所有相关方法，如 `stream` 和 `invoke`。

## Advanced concepts（高级概念）

### Structured output（结构化输出）

在某些情况下，您可能希望 agent 以特定格式返回输出。LangChain 通过 [`response_format`](https://reference.langchain.com/python/langchain/agents/factory/create_agent) 参数提供结构化输出策略。

#### ToolStrategy

`ToolStrategy` 使用人工工具调用生成结构化输出。这适用于任何支持工具调用的模型。当提供者原生结构化输出（通过 [`ProviderStrategy`](#providerstrategy)）不可用或不可靠时，应使用 `ToolStrategy`。

```python wrap theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model="gpt-5.4-mini",
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "从以下内容中提取联系信息：John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

#### ProviderStrategy

`ProviderStrategy` 使用模型提供者的原生结构化输出生成。这更可靠，但仅适用于支持原生结构化输出的提供者：

```python wrap theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="gpt-5.4",
    response_format=ProviderStrategy(ContactInfo)
)
```

<Note>
  从 `langchain 1.0` 开始，简单地传递模式（例如，`response_format=ContactInfo`）将在模型支持原生结构化输出时默认为 `ProviderStrategy`。否则，它将回退到 `ToolStrategy`。
</Note>

<Tip>
  要了解结构化输出，请参阅 [Structured output](/oss/python/langchain/structured-output)。
</Tip>

### Memory（记忆）

Agent 通过消息状态自动维护对话历史。您还可以配置 agent 使用自定义状态架构来记住对话期间的额外信息。

存储在状态中的信息可以被视为 agent 的[短期记忆](/oss/python/langchain/short-term-memory)：

自定义状态架构必须作为 `TypedDict` 扩展 [`AgentState`](https://reference.langchain.com/python/langchain/agents/middleware/types/AgentState)。

有两种定义自定义状态的方法：

1. 通过 [中间件](/oss/python/langchain/middleware)（首选）
2. 通过 [`create_agent`](https://reference.langchain.com/python/langchain/agents/factory/create_agent) 上的 [`state_schema`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.AgentMiddleware.state_schema)

#### 通过中间件定义状态

当您的自定义状态需要由附加到所述中间件的特定中间件钩子和工具访问时，请使用中间件来定义自定义状态。

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from typing import Any


class CustomState(AgentState):
    user_preferences: dict

class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState
    tools = [tool1, tool2]

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        ...

agent = create_agent(
    model,
    tools=tools,
    middleware=[CustomMiddleware()]
)

# agent 现在可以跟踪除消息之外的其他状态
result = agent.invoke({
    "messages": [{"role": "user", "content": "我更喜欢技术解释"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})
```

#### 通过 `state_schema` 定义状态

使用 [`state_schema`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.AgentMiddleware.state_schema) 参数作为快捷方式来定义仅在工具中使用的自定义状态。

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import AgentState


class CustomState(AgentState):
    user_preferences: dict

agent = create_agent(
    model,
    tools=[tool1, tool2],
    state_schema=CustomState
)
# agent 现在可以跟踪除消息之外的其他状态
result = agent.invoke({
    "messages": [{"role": "user", "content": "我更喜欢技术解释"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})
```

<Note>
  从 `langchain 1.0` 开始，自定义状态架构**必须**是 `TypedDict` 类型。不再支持 Pydantic 模型和数据类。有关更多详细信息，请参阅 [v1 迁移指南](/oss/python/migrate/langchain-v1#state-type-restrictions)。
</Note>

<Note>
  通过中间件定义自定义状态优于通过 [`create_agent`](https://reference.langchain.com/python/langchain/agents/factory/create_agent) 上的 [`state_schema`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.AgentMiddleware.state_schema) 定义它，因为它允许您将状态扩展概念性地限定于相关的中间件和工具。

  出于向后兼容性的原因，[`create_agent`](https://reference.langchain.com/python/langchain/agents/factory/create_agent) 上仍支持 [`state_schema`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.AgentMiddleware.state_schema)。
</Note>

<Tip>
  要了解有关记忆的更多信息，请参阅 [Memory](/oss/python/concepts/memory)。有关实现跨会话持久存在的长期记忆的信息，请参阅 [Long-term memory](/oss/python/langchain/long-term-memory)。
</Tip>

### Streaming（流式传输）

我们已经了解了如何使用 `invoke` 调用 agent 以获得最终响应。如果 agent 执行多个步骤，这可能需要一段时间。为了显示中间进度，我们可以随着消息的出现流式传输它们。

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.messages import AIMessage, HumanMessage

for chunk in agent.stream({
    "messages": [{"role": "user", "content": "搜索 AI 新闻并总结发现"}]
}, stream_mode="values"):
    # 每个块包含该点的完整状态
    latest_message = chunk["messages"][-1]
    if latest_message.content:
        if isinstance(latest_message, HumanMessage):
            print(f"User: {latest_message.content}")
        elif isinstance(latest_message, AIMessage):
            print(f"Agent: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")
```

<Tip>
  有关流式传输的更多详细信息，请参阅 [Streaming](/oss/python/langchain/streaming)。
</Tip>

### Middleware（中间件）

[中间件](/oss/python/langchain/middleware) 为在不同执行阶段自定义 agent 行为提供了强大的可扩展性。您可以使用中间件来：

* 在调用模型之前处理状态（例如，消息修剪、上下文注入）
* 修改或验证模型的响应（例如，护栏、内容过滤）
* 使用自定义逻辑处理工具执行错误
* 基于状态或上下文实现动态模型选择
* 添加自定义日志记录、监控或分析

中间件无缝集成到 agent 的执行中，允许您在关键点拦截和修改数据流，而无需更改核心 agent 逻辑。

<Tip>
  有关全面的中间件文档，包括 [`@before_model`](https://reference.langchain.com/python/langchain/agents/middleware/types/before_model)、[`@after_model`](https://reference.langchain.com/python/langchain/agents/middleware/types/after_model) 和 [`@wrap_tool_call`](https://reference.langchain.com/python/langchain/agents/middleware/types/wrap_tool_call) 等装饰器，请参阅 [Middleware](/oss/python/langchain/middleware)。
</Tip>

***

<div className="source-links">
  <Callout icon="terminal-2">
    通过 MCP 将这些文档连接到 Claude、VSCode 等，以获得实时答案。
  </Callout>

  <Callout icon="edit">
    在 GitHub 上[编辑此页面](https://github.com/langchain-ai/docs/edit/main/src/oss/langchain/agents.mdx)或[提交问题](https://github.com/langchain-ai/docs/issues/new/choose)。
  </Callout>
</div>