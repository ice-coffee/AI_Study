#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
LangChain API - Memory（对话记忆）
================================================================================

模块：langchain_core.messages, langchain_community.chat_message_histories
用途：管理对话历史，实现多轮对话的上下文保持

核心概念：
--------------------------------------------------------------------------------
1. Message History: 消息历史
   - 存储对话中的消息
   - 包括用户消息和 AI 回复

2. 消息类型:
   - HumanMessage: 用户消息
   - AIMessage: AI 回复
   - SystemMessage: 系统消息
   - FunctionMessage: 函数调用结果

3. ChatMessageHistory: 聊天消息历史类
   - 内存中的消息存储
   - 支持 add_user_message, add_ai_message

4. RunnableWithMessageHistory: 带记忆的 Runnable
   - 自动管理对话历史
   - 支持会话隔离

存储后端：
--------------------------------------------------------------------------------
- ChatMessageHistory: 内存存储
- FileChatMessageHistory: 文件存储
- RedisChatMessageHistory: Redis 存储
- SQLChatMessageHistory: 数据库存储
================================================================================
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from config import get_chat_model


def manual_memory():
    """
    手动管理对话历史
    """
    llm = get_chat_model()

    # 手动管理消息列表
    messages = [SystemMessage(content="你是一个友好的助手，会记住用户的偏好。")]
    
    def chat(user_input: str) -> str:
        # 添加用户消息
        messages.append(HumanMessage(content=user_input))
        
        # 调用模型
        response = llm.invoke(messages)
        
        # 添加 AI 回复
        messages.append(AIMessage(content=response.content))
        
        return response.content
    
    # 第一轮对话
    print("用户: 我叫张三，我喜欢 Python 编程")
    print("AI:", chat("我叫张三，我喜欢 Python 编程"))
    
    # 第二轮对话
    print("\n用户: 我叫什么名字？我喜欢什么？")
    print("AI:", chat("我叫什么名字？我喜欢什么？"))
    
    # 显示完整历史
    print("\n完整对话历史:")
    for msg in messages:
        print(f"  {type(msg).__name__}: {msg.content[:50]}...")


def chat_message_history():
    """
    使用 ChatMessageHistory 类
    """
    llm = get_chat_model()

    # 创建消息历史
    history = ChatMessageHistory()
    
    def chat(user_input: str) -> str:
        # 添加用户消息
        history.add_user_message(user_input)
        
        # 获取所有消息并调用模型
        response = llm.invoke(history.messages)
        
        # 添加 AI 回复
        history.add_ai_message(response.content)
        
        return response.content
    
    print("对话开始 (输入 'quit' 退出)")
    print("-" * 40)
    
    # 模拟多轮对话
    questions = ["你好", "我叫李四", "我叫什么名字？"]
    for q in questions:
        print(f"用户: {q}")
        print(f"AI: {chat(q)}\n")


def runnable_with_message_history():
    """
    使用 RunnableWithMessageHistory 自动管理历史
    """
    llm = get_chat_model()

    # 创建带历史占位符的提示词
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的助手"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # 构建链
    chain = prompt | llm | StrOutputParser()
    
    # 存储会话历史
    store = {}
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    # 包装为带记忆的链
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    # 会话 1
    print("=== 会话 1 ===")
    print("用户: 我叫王五")
    print("AI:", chain_with_history.invoke(
        {"input": "我叫王五"},
        config={"configurable": {"session_id": "session1"}}
    ))
    
    print("\n用户: 我叫什么名字？")
    print("AI:", chain_with_history.invoke(
        {"input": "我叫什么名字？"},
        config={"configurable": {"session_id": "session1"}}
    ))
    
    # 会话 2 (不同的 session_id，独立的历史)
    print("\n=== 会话 2 (新会话) ===")
    print("用户: 我叫什么名字？")
    print("AI:", chain_with_history.invoke(
        {"input": "我叫什么名字？"},
        config={"configurable": {"session_id": "session2"}}
    ))


def memory_window():
    """
    滑动窗口记忆 - 只保留最近 N 条消息
    """
    from langchain_core.messages import trim_messages

    llm = get_chat_model()

    # 创建消息历史
    history = ChatMessageHistory()
    
    # 添加多轮对话
    history.add_user_message("你好")
    history.add_ai_message("你好！有什么可以帮助你的吗？")
    history.add_user_message("我叫张三")
    history.add_ai_message("你好张三！很高兴认识你。")
    history.add_user_message("我喜欢编程")
    history.add_ai_message("编程是一项很有趣的技能！你喜欢哪种编程语言？")
    history.add_user_message("我喜欢 Python")
    history.add_ai_message("Python 是一门很棒的语言！")
    
    print(f"原始消息数量: {len(history.messages)}")
    
    # 使用 trim_messages 截断消息
    trimmed_messages = trim_messages(
        history.messages,
        max_tokens=100,  # 最大 token 数
        strategy="last",  # 保留最后的消息
        token_counter=len,  # 简单计数器
        include_system=True,  # 保留系统消息
    )
    
    print(f"截断后消息数量: {len(trimmed_messages)}")
    
    for msg in trimmed_messages:
        print(f"  {type(msg).__name__}: {msg.content[:30]}...")


def summary_memory():
    """
    摘要记忆 - 将历史压缩为摘要
    """
    llm = get_chat_model()

    # 模拟长对话历史
    conversation = [
        ("human", "你好"),
        ("ai", "你好！有什么可以帮助你的吗？"),
        ("human", "我叫张三，我是一名软件工程师"),
        ("ai", "你好张三！作为软件工程师，你一定有很多有趣的经历。"),
        ("human", "我主要用 Python 和 JavaScript"),
        ("ai", "Python 和 JavaScript 都是非常流行的语言！"),
        ("human", "我在做一个机器学习项目"),
        ("ai", "机器学习是一个令人兴奋的领域！你在做什么类型的项目？"),
    ]
    
    # 创建摘要提示词
    summary_prompt = ChatPromptTemplate.from_template(
        "请将以下对话总结为一段简短的摘要：\n\n{conversation}"
    )
    
    # 格式化对话
    conv_text = "\n".join([f"{role}: {content}" for role, content in conversation])
    
    # 生成摘要
    summary_chain = summary_prompt | llm | StrOutputParser()
    summary = summary_chain.invoke({"conversation": conv_text})
    
    print("对话摘要:")
    print(summary)
    
    # 使用摘要作为上下文继续对话
    continue_prompt = ChatPromptTemplate.from_messages([
        ("system", "以下是之前对话的摘要：\n{summary}"),
        ("human", "{input}")
    ])
    
    chain = continue_prompt | llm | StrOutputParser()
    result = chain.invoke({
        "summary": summary,
        "input": "我之前说我叫什么名字？做什么项目？"
    })
    
    print("\n基于摘要的回答:")
    print(result)


if __name__ == "__main__":
    print("=" * 60)
    print("1. 手动管理记忆")
    print("=" * 60)
    manual_memory()
    
    print("\n" + "=" * 60)
    print("2. ChatMessageHistory")
    print("=" * 60)
    chat_message_history()
    
    print("\n" + "=" * 60)
    print("3. RunnableWithMessageHistory")
    print("=" * 60)
    runnable_with_message_history()
    
    print("\n" + "=" * 60)
    print("4. 滑动窗口记忆")
    print("=" * 60)
    memory_window()
    
    print("\n" + "=" * 60)
    print("5. 摘要记忆")
    print("=" * 60)
    summary_memory()
