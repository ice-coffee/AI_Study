#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
LangChain API - Output Parsers（输出解析器）
================================================================================

模块：langchain_core.output_parsers
用途：将 LLM 的输出解析为结构化数据

核心组件：
--------------------------------------------------------------------------------
1. StrOutputParser: 字符串输出解析器
   - 最简单的解析器，直接返回字符串

2. JsonOutputParser: JSON 输出解析器
   - 将输出解析为 JSON 对象
   - 可结合 Pydantic 模型进行验证

3. PydanticOutputParser: Pydantic 模型解析器
   - 将输出解析为 Pydantic 模型实例
   - 自动生成格式说明添加到提示词

4. CommaSeparatedListOutputParser: 逗号分隔列表解析器
   - 将逗号分隔的字符串解析为列表

5. StructuredOutputParser: 结构化输出解析器
   - 解析结构化数据

使用流程：
--------------------------------------------------------------------------------
1. 创建 Pydantic 模型定义输出结构
2. 创建解析器实例
3. 将解析器的格式说明添加到提示词
4. 调用 LLM 获取输出
5. 使用解析器解析输出
================================================================================
"""

import os
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    CommaSeparatedListOutputParser,
)
from langchain.output_parsers import PydanticOutputParser


def str_output_parser():
    """
    StrOutputParser 示例 - 最简单的输出解析
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    # 创建提示词模板
    prompt = ChatPromptTemplate.from_template("给我讲一个关于{topic}的笑话")
    
    # 创建输出解析器
    parser = StrOutputParser()
    
    # 构建链
    chain = prompt | llm | parser
    
    # 调用
    result = chain.invoke({"topic": "程序员"})
    print("解析结果:", result)


def json_output_parser():
    """
    JsonOutputParser 示例 - 解析 JSON 输出
    """
    # 定义数据模型
    class Person(BaseModel):
        name: str = Field(description="人名")
        age: int = Field(description="年龄")
        hobbies: List[str] = Field(description="爱好列表")
    
    # 创建解析器
    parser = JsonOutputParser(pydantic_object=Person)
    
    # 创建提示词（包含格式说明）
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个信息提取助手。"),
        ("human", "{query}\n\n{format_instructions}")
    ])
    
    # 部分填充格式说明
    prompt = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )
    
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    # 构建链
    chain = prompt | llm | parser
    
    # 调用
    result = chain.invoke({"query": "提取张三的信息：25岁，喜欢编程、读书和跑步"})
    print("解析结果:", result)
    print("类型:", type(result))


def pydantic_output_parser():
    """
    PydanticOutputParser 示例 - 使用 Pydantic 模型
    """
    # 定义复杂数据模型
    class Book(BaseModel):
        title: str = Field(description="书名")
        author: str = Field(description="作者")
        summary: str = Field(description="简介")
        rating: float = Field(description="评分，0-10分")
        tags: List[str] = Field(description="标签列表")
    
    # 创建解析器
    parser = PydanticOutputParser(pydantic_object=Book)
    
    # 打印格式说明
    print("格式说明:")
    print(parser.get_format_instructions())
    print()
    
    # 创建提示词
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个图书推荐助手。"),
        ("human", "推荐一本关于{topic}的书\n\n{format_instructions}")
    ])
    
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    # 构建链
    chain = prompt | llm | parser
    
    # 调用
    result = chain.invoke({
        "topic": "人工智能",
        "format_instructions": parser.get_format_instructions()
    })
    
    print("解析结果:")
    print(f"  书名: {result.title}")
    print(f"  作者: {result.author}")
    print(f"  评分: {result.rating}")
    print(f"  标签: {result.tags}")


def list_output_parser():
    """
    CommaSeparatedListOutputParser 示例 - 解析列表
    """
    # 创建解析器
    parser = CommaSeparatedListOutputParser()
    
    # 获取格式说明
    format_instructions = parser.get_format_instructions()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个知识助手。"),
        ("human", "列出5个{topic}\n\n{format_instructions}")
    ])
    
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    # 构建链
    chain = prompt | llm | parser
    
    # 调用
    result = chain.invoke({
        "topic": "编程语言",
        "format_instructions": format_instructions
    })
    
    print("解析结果:", result)
    print("类型:", type(result))


def nested_model_parser():
    """
    嵌套模型解析示例
    """
    # 定义嵌套模型
    class Address(BaseModel):
        city: str = Field(description="城市")
        street: str = Field(description="街道")
        
    class Company(BaseModel):
        name: str = Field(description="公司名称")
        address: Address = Field(description="公司地址")
        employees: int = Field(description="员工数量")
    
    # 创建解析器
    parser = JsonOutputParser(pydantic_object=Company)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个信息提取助手。"),
        ("human", "{query}\n\n{format_instructions}")
    ])
    
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    chain = prompt | llm | parser
    
    result = chain.invoke({
        "query": "百度公司位于北京市海淀区上地十街，有约4万员工",
        "format_instructions": parser.get_format_instructions()
    })
    
    print("解析结果:", result)


if __name__ == "__main__":
    print("=" * 60)
    print("1. StrOutputParser")
    print("=" * 60)
    str_output_parser()
    
    print("\n" + "=" * 60)
    print("2. JsonOutputParser")
    print("=" * 60)
    json_output_parser()
    
    print("\n" + "=" * 60)
    print("3. PydanticOutputParser")
    print("=" * 60)
    pydantic_output_parser()
    
    print("\n" + "=" * 60)
    print("4. ListOutputParser")
    print("=" * 60)
    list_output_parser()
    
    print("\n" + "=" * 60)
    print("5. 嵌套模型解析")
    print("=" * 60)
    nested_model_parser()
