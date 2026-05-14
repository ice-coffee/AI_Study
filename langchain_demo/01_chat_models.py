#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
LangChain API - Chat Models（聊天模型）
================================================================================

模块：langchain.chat_models / langchain_openai
用途：LangChain 的聊天模型接口，支持多种 LLM 提供商的统一调用

核心组件：
--------------------------------------------------------------------------------
1. BaseChatModel: 聊天模型的基类
   - 所有聊天模型的统一接口
   - 支持同步和异步调用

2. ChatOpenAI: OpenAI 聊天模型封装
   - 支持 GPT-4, GPT-3.5 等模型
   - 支持流式输出、Function Calling

3. 消息类型:
   - SystemMessage: 系统消息，设定 AI 行为
   - HumanMessage: 用户消息
   - AIMessage: AI 回复消息
   - ToolMessage: 工具调用结果消息

常用参数：
--------------------------------------------------------------------------------
- model: 模型名称，如 "gpt-4o", "gpt-3.5-turbo"
- temperature: 温度参数，控制随机性 (0-2)
- max_tokens: 最大生成 token 数
- timeout: 请求超时时间
- max_retries: 最大重试次数
- api_key: API 密钥
- base_url: API 基础 URL（用于兼容 OpenAI API 的服务）
================================================================================
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.callbacks import StreamingStdOutCallbackHandler


def basic_chat():
    """
    基础对话示例
    """
    # 初始化模型
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    # 构建消息
    messages = [
        SystemMessage(content="你是一个专业的程序员"),
        HumanMessage(content="解释什么是 REST API")
    ]
    
    # 调用模型
    response = llm.invoke(messages)
    
    print("回复内容:", response.content)
    print("Token 用量:", response.response_metadata.get("token_usage"))


def streaming_chat():
    """
    流式输出示例
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    messages = [HumanMessage(content="写一首关于春天的诗")]
    
    print("流式输出: ")
    llm.invoke(messages)
    print()


def batch_chat():
    """
    批量调用示例 - 同时处理多个请求
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    # 批量处理多个问题
    batch_messages = [
        [HumanMessage(content="什么是 Python？")],
        [HumanMessage(content="什么是 JavaScript？")],
        [HumanMessage(content="什么是 Go？")]
    ]
    
    responses = llm.batch(batch_messages)
    
    for i, response in enumerate(responses):
        print(f"\n问题 {i+1} 的回答:")
        print(response.content[:200] + "...")


def async_chat():
    """
    异步调用示例
    """
    import asyncio
    
    async def run():
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.environ.get("ONE_API_KEY"),
            base_url=os.environ.get("BASE_URL")
        )
        
        response = await llm.ainvoke([HumanMessage(content="你好")])
        print("异步回复:", response.content)
    
    asyncio.run(run())


def with_timeout():
    """
    带超时和重试的调用
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        timeout=10,  # 10秒超时
        max_retries=3,  # 最大重试3次
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    response = llm.invoke([HumanMessage(content="你好")])
    print("回复:", response.content)


def multi_turn_chat():
    """
    多轮对话示例 - 手动管理消息历史
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    # 初始化消息历史
    history = [SystemMessage(content="你是一个友好的助手")]
    
    def chat(user_input: str) -> str:
        # 添加用户消息
        history.append(HumanMessage(content=user_input))
        
        # 调用模型
        response = llm.invoke(history)
        
        # 添加 AI 回复到历史
        history.append(AIMessage(content=response.content))
        
        return response.content
    
    print("第一轮:", chat("我叫张三"))
    print("第二轮:", chat("我叫什么名字？"))  # 模型会记住张三


if __name__ == "__main__":
    print("=" * 60)
    print("1. 基础对话")
    print("=" * 60)
    basic_chat()
    
    print("\n" + "=" * 60)
    print("2. 流式输出")
    print("=" * 60)
    streaming_chat()
    
    print("\n" + "=" * 60)
    print("3. 批量调用")
    print("=" * 60)
    batch_chat()
    
    print("\n" + "=" * 60)
    print("4. 异步调用")
    print("=" * 60)
    async_chat()
    
    print("\n" + "=" * 60)
    print("5. 多轮对话")
    print("=" * 60)
    multi_turn_chat()
