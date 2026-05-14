#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
LangChain API - Prompts（提示词模板）
================================================================================

模块：langchain_core.prompts
用途：构建可复用、参数化的提示词模板

核心组件：
--------------------------------------------------------------------------------
1. PromptTemplate: 字符串模板
   - 用于构建简单的文本提示词
   - 支持变量替换

2. ChatPromptTemplate: 聊天模板
   - 用于构建聊天消息序列
   - 支持多种消息类型

3. MessagesPlaceholder: 消息占位符
   - 用于动态插入消息列表
   - 常用于对话历史

4. SystemMessagePromptTemplate: 系统消息模板
5. HumanMessagePromptTemplate: 人类消息模板
6. AIMessagePromptTemplate: AI消息模板

模板语法：
--------------------------------------------------------------------------------
- {variable}: 变量占位符
- 支持 f-string 和 jinja2 风格
- 可以使用 partial() 部分填充变量
================================================================================
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def basic_prompt_template():
    """
    基础 PromptTemplate 示例
    """
    # 创建模板
    template = PromptTemplate.from_template("给我讲一个关于{topic}的笑话")
    
    # 格式化模板
    prompt = template.format(topic="程序员")
    print("格式化后的提示词:", prompt)
    
    # 与 LLM 配合使用
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    response = llm.invoke(prompt)
    print("回复:", response.content)


def chat_prompt_template():
    """
    ChatPromptTemplate 示例 - 构建聊天消息序列
    """
    # 方式一：from_messages 方法
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}，请用{style}的风格回答"),
        ("human", "{question}")
    ])
    
    # 格式化
    messages = prompt.format_messages(
        role="专业的程序员",
        style="幽默",
        question="什么是 REST API"
    )
    
    print("消息列表:")
    for msg in messages:
        print(f"  {type(msg).__name__}: {msg.content}")
    
    # 调用模型
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    response = llm.invoke(messages)
    print("\n回复:", response.content)


def messages_placeholder():
    """
    MessagesPlaceholder 示例 - 动态插入消息历史
    """
    # 创建带占位符的模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的助手"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # 模拟对话历史
    history = [
        ("human", "我叫张三"),
        ("ai", "你好张三！很高兴认识你")
    ]
    
    # 格式化
    messages = prompt.format_messages(
        history=history,
        input="我叫什么名字？"
    )
    
    print("完整消息列表:")
    for msg in messages:
        print(f"  {type(msg).__name__}: {msg.content}")


def partial_template():
    """
    部分填充模板示例
    """
    # 创建模板
    template = PromptTemplate.from_template(
        "{country}的首都是哪里？请用{language}回答。"
    )
    
    # 部分填充 - 固定 language 变量
    partial_template = template.partial(language="中文")
    
    # 使用部分填充后的模板
    prompt1 = partial_template.format(country="法国")
    prompt2 = partial_template.format(country="日本")
    
    print("提示词1:", prompt1)
    print("提示词2:", prompt2)


def template_with_examples():
    """
    带示例的模板 - Few-shot prompting
    """
    # 创建带示例的提示词
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一个翻译助手，将中文翻译成英文。"),
        ("human", "你好"),
        ("ai", "Hello"),
        ("human", "谢谢"),
        ("ai", "Thank you"),
        ("human", "{text}")
    ])
    
    # 使用模板
    messages = template.format_messages(text="早上好")
    
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    response = llm.invoke(messages)
    print("翻译结果:", response.content)


def template_composition():
    """
    模板组合示例
    """
    # 创建多个子模板
    system_template = SystemMessagePromptTemplate.from_template(
        "你是一个{role}"
    )
    
    human_template = HumanMessagePromptTemplate.from_template(
        "{question}"
    )
    
    # 组合成完整模板
    full_prompt = ChatPromptTemplate.from_messages([
        system_template,
        human_template
    ])
    
    messages = full_prompt.format_messages(
        role="Python专家",
        question="什么是装饰器？"
    )
    
    print("组合后的消息:")
    for msg in messages:
        print(f"  {type(msg).__name__}: {msg.content}")


if __name__ == "__main__":
    print("=" * 60)
    print("1. 基础 PromptTemplate")
    print("=" * 60)
    basic_prompt_template()
    
    print("\n" + "=" * 60)
    print("2. ChatPromptTemplate")
    print("=" * 60)
    chat_prompt_template()
    
    print("\n" + "=" * 60)
    print("3. MessagesPlaceholder")
    print("=" * 60)
    messages_placeholder()
    
    print("\n" + "=" * 60)
    print("4. 部分填充模板")
    print("=" * 60)
    partial_template()
    
    print("\n" + "=" * 60)
    print("5. Few-shot 提示")
    print("=" * 60)
    template_with_examples()
    
    print("\n" + "=" * 60)
    print("6. 模板组合")
    print("=" * 60)
    template_composition()
