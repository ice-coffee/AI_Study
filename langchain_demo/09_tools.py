#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
LangChain API - Tools（工具定义与使用）
================================================================================

模块：langchain_core.tools
用途：定义可被 LLM 调用的工具函数

核心概念：
--------------------------------------------------------------------------------
1. Tool: 工具基类
   - name: 工具名称
   - description: 工具描述（LLM 用于判断何时使用）
   - func: 工具执行函数
   - args_schema: 参数模式（使用 Pydantic 定义）

2. @tool 装饰器:
   - 最简单的工具创建方式
   - 自动从函数签名生成参数模式
   - 使用 docstring 作为描述

3. StructuredTool:
   - 更灵活的工具定义
   - 支持复杂参数类型
   - 可以自定义参数验证

工具设计原则：
--------------------------------------------------------------------------------
1. 单一职责：每个工具只做一件事
2. 清晰描述：让 LLM 能准确判断何时使用
3. 类型提示：使用类型注解帮助生成正确的参数模式
4. 错误处理：工具应该优雅地处理错误
================================================================================
"""

import os
import json
from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, Tool, StructuredTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate


# ==================== 基础工具定义 ====================

@tool
def get_current_weather(
    city: str,
    unit: Literal["celsius", "fahrenheit"] = "celsius"
) -> str:
    """
    获取指定城市的当前天气
    
    Args:
        city: 城市名称，如"北京"、"上海"
        unit: 温度单位，celsius 或 fahrenheit
    
    Returns:
        天气信息字符串
    """
    # 模拟天气数据
    weather_data = {
        "北京": {"temp": 25, "condition": "晴天"},
        "上海": {"temp": 28, "condition": "多云"},
        "广州": {"temp": 32, "condition": "小雨"},
    }
    
    if city not in weather_data:
        return f"未找到城市 {city} 的天气信息"
    
    data = weather_data[city]
    temp = data["temp"]
    if unit == "fahrenheit":
        temp = temp * 9 / 5 + 32
    
    return f"{city}当前天气：{data['condition']}，温度 {temp}°{'F' if unit == 'fahrenheit' else 'C'}"


@tool
def search_web(query: str, max_results: int = 5) -> str:
    """
    在网上搜索信息
    
    Args:
        query: 搜索关键词
        max_results: 返回结果的最大数量
    
    Returns:
        搜索结果
    """
    # 模拟搜索结果
    mock_results = [
        {"title": f"关于 {query} 的介绍", "url": "https://example.com/1"},
        {"title": f"{query} 的最新动态", "url": "https://example.com/2"},
        {"title": f"如何学习 {query}", "url": "https://example.com/3"},
    ]
    
    results = mock_results[:max_results]
    return json.dumps(results, ensure_ascii=False)


@tool
def execute_python_code(code: str) -> str:
    """
    执行 Python 代码并返回结果
    
    注意：此工具仅用于演示，实际应用中需要考虑安全性
    
    Args:
        code: 要执行的 Python 代码
    
    Returns:
        执行结果或错误信息
    """
    try:
        # 创建受限的全局命名空间
        safe_globals = {"__builtins__": {}}
        result = eval(code, safe_globals, {})
        return str(result)
    except Exception as e:
        return f"执行错误: {str(e)}"


# ==================== 结构化工具定义 ====================

class DatabaseQueryInput(BaseModel):
    """数据库查询参数"""
    table: str = Field(description="表名")
    columns: List[str] = Field(default=["*"], description="要查询的列")
    where: Optional[str] = Field(default=None, description="WHERE 条件")
    limit: int = Field(default=10, description="返回结果数量限制")


def query_database(table: str, columns: List[str] = ["*"], 
                   where: Optional[str] = None, limit: int = 10) -> str:
    """查询数据库"""
    # 模拟数据库查询
    mock_data = {
        "users": [
            {"id": 1, "name": "张三", "email": "zhangsan@example.com"},
            {"id": 2, "name": "李四", "email": "lisi@example.com"},
        ],
        "products": [
            {"id": 1, "name": "Python教程", "price": 99},
            {"id": 2, "name": "JavaScript教程", "price": 79},
        ]
    }
    
    if table not in mock_data:
        return f"表 {table} 不存在"
    
    return json.dumps(mock_data[table][:limit], ensure_ascii=False)


# 使用 StructuredTool 创建结构化工具
database_tool = StructuredTool.from_function(
    func=query_database,
    name="query_database",
    description="查询数据库获取数据",
    args_schema=DatabaseQueryInput
)


# ==================== 工具类定义 ====================

class CalculatorTool:
    """计算器工具类"""
    
    @tool
    @staticmethod
    def add(a: float, b: float) -> float:
        """加法运算"""
        return a + b
    
    @tool
    @staticmethod
    def subtract(a: float, b: float) -> float:
        """减法运算"""
        return a - b
    
    @tool
    @staticmethod
    def multiply(a: float, b: float) -> float:
        """乘法运算"""
        return a * b
    
    @tool
    @staticmethod
    def divide(a: float, b: float) -> str:
        """除法运算"""
        if b == 0:
            return "错误：除数不能为零"
        return str(a / b)


# ==================== 示例函数 ====================

def basic_tool_usage():
    """
    基础工具使用示例
    """
    print("直接调用工具:")
    print("-" * 40)
    
    # 直接调用工具函数
    result = get_current_weather.invoke({"city": "北京"})
    print(f"天气查询: {result}")
    
    result = search_web.invoke({"query": "Python教程"})
    print(f"搜索结果: {result}")


def tool_with_agent():
    """
    在 Agent 中使用工具
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    # 定义工具列表
    tools = [get_current_weather, search_web, database_tool]
    
    # 创建提示词
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手，可以使用各种工具帮助用户。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # 创建 Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # 执行任务
    result = agent_executor.invoke({
        "input": "帮我查一下北京的天气，然后搜索一下 Python 教程"
    })
    print("\n最终结果:", result["output"])


def create_tool_from_function():
    """
    从普通函数创建工具
    """
    # 普通函数
    def get_word_count(text: str) -> int:
        """计算文本中的单词数量"""
        return len(text.split())
    
    # 方式一：使用 Tool 类
    word_count_tool = Tool(
        name="word_count",
        description="计算文本中的单词数量",
        func=get_word_count
    )
    
    # 方式二：使用 @tool 装饰器包装已有函数
    @tool
    def word_count(text: str) -> int:
        """计算文本中的单词数量"""
        return len(text.split())
    
    print("创建的工具:")
    print(f"  - {word_count_tool.name}: {word_count_tool.description}")
    print(f"  - {word_count.name}: {word_count.description}")
    
    # 测试
    print(f"\n测试: '{'Hello World Python'}' -> {word_count.invoke({'text': 'Hello World Python'})} 个单词")


def tool_error_handling():
    """
    工具错误处理示例
    """
    @tool
    def safe_divide(a: float, b: float) -> str:
        """
        安全的除法运算，会处理除零错误
        
        Args:
            a: 被除数
            b: 除数
        
        Returns:
            除法结果或错误信息
        """
        try:
            if b == 0:
                return "错误：除数不能为零，请提供非零的除数"
            result = a / b
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算过程中发生错误: {str(e)}"
    
    # 测试正常情况
    print("正常除法:")
    print(safe_divide.invoke({"a": 10, "b": 2}))
    
    # 测试错误情况
    print("\n除零错误:")
    print(safe_divide.invoke({"a": 10, "b": 0}))


def tool_with_validation():
    """
    带参数验证的工具
    """
    class EmailInput(BaseModel):
        """邮件输入参数"""
        to: str = Field(description="收件人邮箱", pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
        subject: str = Field(description="邮件主题", min_length=1, max_length=100)
        body: str = Field(description="邮件正文", min_length=1)
    
    @tool(args_schema=EmailInput)
    def send_email(to: str, subject: str, body: str) -> str:
        """
        发送邮件
        
        Args:
            to: 收件人邮箱
            subject: 邮件主题
            body: 邮件正文
        
        Returns:
            发送结果
        """
        # 模拟发送邮件
        return f"邮件已发送至 {to}\n主题: {subject}"
    
    # 测试
    print("有效邮件:")
    try:
        result = send_email.invoke({
            "to": "test@example.com",
            "subject": "测试邮件",
            "body": "这是一封测试邮件"
        })
        print(result)
    except Exception as e:
        print(f"验证错误: {e}")
    
    print("\n无效邮箱:")
    try:
        result = send_email.invoke({
            "to": "invalid-email",
            "subject": "测试邮件",
            "body": "这是一封测试邮件"
        })
        print(result)
    except Exception as e:
        print(f"验证错误: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("1. 基础工具使用")
    print("=" * 60)
    basic_tool_usage()
    
    print("\n" + "=" * 60)
    print("2. 工具与 Agent")
    print("=" * 60)
    tool_with_agent()
    
    print("\n" + "=" * 60)
    print("3. 从函数创建工具")
    print("=" * 60)
    create_tool_from_function()
    
    print("\n" + "=" * 60)
    print("4. 工具错误处理")
    print("=" * 60)
    tool_error_handling()
    
    print("\n" + "=" * 60)
    print("5. 带验证的工具")
    print("=" * 60)
    tool_with_validation()
