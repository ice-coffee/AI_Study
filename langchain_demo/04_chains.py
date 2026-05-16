#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
LangChain API - Chains（链式调用）
================================================================================

模块：langchain_core.runnables
用途：将多个组件串联成处理流水线

核心概念：
--------------------------------------------------------------------------------
1. Runnable: 可执行单元接口
   - 所有 LangChain 组件都实现了 Runnable 接口
   - 支持 invoke(), batch(), stream() 等方法

2. LCEL (LangChain Expression Language):
   - 使用 | 操作符连接组件
   - 自动处理数据流转

3. RunnableSequence: 链式序列
   - prompt | llm | parser
   - 按顺序执行每个组件

4. RunnableParallel: 并行执行
   - 同时执行多个分支
   - 结果以字典形式返回

5. RunnablePassthrough: 数据透传
   - 将输入原样传递
   - 用于构建复杂的数据流

常用操作符：
--------------------------------------------------------------------------------
- |: 管道操作符，连接组件
- .assign(): 添加新的输出字段
- .map(): 对列表中的每个元素应用操作
================================================================================
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from config import get_chat_model


def basic_chain():
    """
    基础链式调用示例
    """
    # 定义组件
    prompt = ChatPromptTemplate.from_template("给我讲一个关于{topic}的笑话")
    llm = get_chat_model()
    parser = StrOutputParser()
    
    # 使用 LCEL 构建链
    chain = prompt | llm | parser
    
    # 调用
    result = chain.invoke({"topic": "程序员"})
    print("结果:", result)


def chain_with_multiple_steps():
    """
    多步骤链式调用
    """
    # 第一步：生成问题
    question_prompt = ChatPromptTemplate.from_template(
        "生成一个关于{topic}的问题"
    )
    
    # 第二步：回答问题
    answer_prompt = ChatPromptTemplate.from_template(
        "请回答以下问题：{question}"
    )

    llm = get_chat_model()

    # 构建链：先生成问题，再回答
    question_chain = question_prompt | llm | StrOutputParser()
    answer_chain = answer_prompt | llm | StrOutputParser()
    
    # 组合成完整链
    full_chain = (
        {"question": question_chain}
        | answer_chain
    )
    
    result = full_chain.invoke({"topic": "人工智能"})
    print("最终答案:", result)


def parallel_chain():
    """
    并行执行多个链
    """
    llm = get_chat_model()

    # 定义多个并行任务
    joke_chain = (
        ChatPromptTemplate.from_template("讲一个关于{topic}的笑话")
        | llm
        | StrOutputParser()
    )
    
    fact_chain = (
        ChatPromptTemplate.from_template("说一个关于{topic}的事实")
        | llm
        | StrOutputParser()
    )
    
    # 并行执行
    chain = RunnableParallel(
        joke=joke_chain,
        fact=fact_chain
    )
    
    result = chain.invoke({"topic": "猫"})
    print("笑话:", result["joke"][:100], "...")
    print("事实:", result["fact"][:100], "...")


def chain_with_passthrough():
    """
    使用 RunnablePassthrough 保留原始输入
    """
    llm = get_chat_model()

    prompt = ChatPromptTemplate.from_template(
        "主题: {topic}\n问题: {question}\n请回答："
    )
    
    # 使用 RunnablePassthrough 保留原始输入
    chain = (
        {
            "topic": RunnablePassthrough(),
            "question": lambda x: f"什么是{x}？"
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    result = chain.invoke("Python")
    print("结果:", result[:200], "...")


def chain_with_assign():
    """
    使用 assign 添加中间结果
    """
    llm = get_chat_model()

    # 定义子链
    translation_chain = (
        ChatPromptTemplate.from_template("将以下英文翻译成中文：{text}")
        | llm
        | StrOutputParser()
    )
    
    # 使用 assign 添加翻译结果
    chain = RunnablePassthrough.assign(translation=translation_chain)
    
    result = chain.invoke({"text": "Hello, World!"})
    print("原文:", result["text"])
    print("翻译:", result["translation"])


def fallback_chain():
    """
    回退链 - 当主链失败时使用备用链
    """
    llm = get_chat_model()

    # 主链
    main_chain = (
        ChatPromptTemplate.from_template("用{style}风格介绍{topic}")
        | llm
        | StrOutputParser()
    )
    
    # 备用链
    fallback_chain = (
        ChatPromptTemplate.from_template("介绍{topic}")
        | llm
        | StrOutputParser()
    )
    
    # 设置回退
    chain = main_chain.with_fallbacks([fallback_chain])
    
    result = chain.invoke({"topic": "Python", "style": "幽默"})
    print("结果:", result[:200], "...")


def streaming_chain():
    """
    流式输出链
    """
    llm = get_chat_model()

    chain = (
        ChatPromptTemplate.from_template("写一首关于{topic}的诗")
        | llm
        | StrOutputParser()
    )
    
    print("流式输出: ")
    for chunk in chain.stream({"topic": "春天"}):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("1. 基础链式调用")
    print("=" * 60)
    basic_chain()
    
    print("\n" + "=" * 60)
    print("2. 多步骤链")
    print("=" * 60)
    chain_with_multiple_steps()
    
    print("\n" + "=" * 60)
    print("3. 并行链")
    print("=" * 60)
    parallel_chain()
    
    print("\n" + "=" * 60)
    print("4. Passthrough 链")
    print("=" * 60)
    chain_with_passthrough()
    
    print("\n" + "=" * 60)
    print("5. Assign 链")
    print("=" * 60)
    chain_with_assign()
    
    print("\n" + "=" * 60)
    print("6. 流式链")
    print("=" * 60)
    streaming_chain()
