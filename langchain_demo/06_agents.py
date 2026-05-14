#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
LangChain API - Agents（智能体）
================================================================================

模块：langchain.agents
用途：让 LLM 能够自主决定使用哪些工具来完成任务

核心概念：
--------------------------------------------------------------------------------
1. Agent: 智能体
   - 分析输入决定采取什么行动
   - 可以使用工具或返回最终答案

2. Tools: 工具
   - 智能体可以调用的功能
   - 每个工具有名称、描述和执行函数

3. AgentExecutor: 智能体执行器
   - 运行智能体的运行时
   - 处理工具调用循环

Agent 类型：
--------------------------------------------------------------------------------
1. ReAct Agent: 推理+行动模式
   - 先思考，再行动
   - 适合需要多步推理的任务

2. OpenAI Functions Agent: 使用 Function Calling
   - 利用模型的 Function Calling 能力
   - 更可靠的工具调用

3. Structured Chat Agent: 结构化聊天
   - 支持多输入参数的工具
   - 适合复杂工具调用

工具创建方式：
--------------------------------------------------------------------------------
- @tool 装饰器
- Tool 类
- StructuredTool 类
================================================================================
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate


# ==================== 定义工具 ====================

@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息
    
    Args:
        city: 城市名称
    
    Returns:
        天气信息字符串
    """
    # 模拟天气数据
    weather_data = {
        "北京": "晴天，温度 25°C，空气质量良好",
        "上海": "多云，温度 28°C，有轻微雾霾",
        "广州": "小雨，温度 30°C，湿度较高",
        "深圳": "晴天，温度 32°C，紫外线较强",
    }
    return weather_data.get(city, f"未找到{city}的天气信息")


@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式
    
    Args:
        expression: 数学表达式，如 "2 + 3 * 4"
    
    Returns:
        计算结果
    """
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def search_database(query: str) -> str:
    """
    在数据库中搜索信息
    
    Args:
        query: 搜索关键词
    
    Returns:
        搜索结果
    """
    # 模拟数据库
    database = {
        "Python": "Python 是一种高级编程语言，广泛用于 Web 开发、数据科学和 AI。",
        "JavaScript": "JavaScript 是一种脚本语言，主要用于 Web 前端开发。",
        "Go": "Go 是 Google 开发的编程语言，专注于并发和高性能。",
    }
    for key, value in database.items():
        if key.lower() in query.lower():
            return value
    return "未找到相关信息"


# ==================== Agent 示例 ====================

def basic_agent():
    """
    基础智能体示例
    """
    # 初始化 LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    # 定义工具列表
    tools = [get_weather, calculate, search_database]
    
    # 创建提示词
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手，可以使用工具来帮助用户。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),  # 用于显示思考过程
    ])
    
    # 创建智能体
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # 创建执行器
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # 执行任务
    result = agent_executor.invoke({"input": "北京今天的天气怎么样？"})
    print("结果:", result["output"])


def multi_tool_agent():
    """
    多工具协作示例
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    tools = [get_weather, calculate, search_database]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手，可以使用工具来帮助用户。必要时组合使用多个工具。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # 需要使用多个工具的任务
    result = agent_executor.invoke({
        "input": "帮我查一下 Python 的介绍，然后计算 100 * 5 的结果"
    })
    print("结果:", result["output"])


def agent_with_memory():
    """
    带记忆的智能体
    """
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory
    
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    tools = [get_weather, search_database]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手。"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    
    # 创建消息历史
    message_history = ChatMessageHistory()
    
    # 包装为带记忆的执行器
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    # 第一轮对话
    result1 = agent_with_chat_history.invoke(
        {"input": "查一下北京的天气"},
        config={"configurable": {"session_id": "test"}}
    )
    print("第一轮:", result1["output"])
    
    # 第二轮对话（会记住上下文）
    result2 = agent_with_chat_history.invoke(
        {"input": "那个城市我刚才问了？"},
        config={"configurable": {"session_id": "test"}}
    )
    print("第二轮:", result2["output"])


def custom_tool_class():
    """
    使用 Tool 类创建自定义工具
    """
    from langchain_core.tools import Tool, StructuredTool
    from pydantic import BaseModel, Field
    
    # 方式一：使用 Tool 类（简单工具）
    def get_time() -> str:
        """获取当前时间"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    time_tool = Tool(
        name="get_current_time",
        description="获取当前时间",
        func=get_time
    )
    
    # 方式二：使用 StructuredTool（带参数的工具）
    class EmailInput(BaseModel):
        recipient: str = Field(description="收件人邮箱")
        subject: str = Field(description="邮件主题")
        body: str = Field(description="邮件正文")
    
    def send_email(recipient: str, subject: str, body: str) -> str:
        """发送邮件"""
        return f"已发送邮件给 {recipient}\n主题: {subject}\n内容: {body[:50]}..."
    
    email_tool = StructuredTool.from_function(
        func=send_email,
        name="send_email",
        description="发送邮件给指定收件人",
        args_schema=EmailInput
    )
    
    print("工具列表:")
    print(f"  - {time_tool.name}: {time_tool.description}")
    print(f"  - {email_tool.name}: {email_tool.description}")
    
    # 测试工具
    print(f"\n时间工具测试: {time_tool.func()}")
    print(f"\n邮件工具测试: {email_tool.func('test@example.com', 'Hello', 'This is a test email.')}")


if __name__ == "__main__":
    print("=" * 60)
    print("1. 基础智能体")
    print("=" * 60)
    basic_agent()
    
    print("\n" + "=" * 60)
    print("2. 多工具协作")
    print("=" * 60)
    multi_tool_agent()
    
    print("\n" + "=" * 60)
    print("3. 自定义工具类")
    print("=" * 60)
    custom_tool_class()
