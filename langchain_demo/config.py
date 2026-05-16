#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
LangChain Demo - 配置文件
================================================================================

统一管理所有 LLM 和 Embedding 模型的配置参数

使用方式:
    from config import get_chat_model, get_embeddings

    # 获取聊天模型
    llm = get_chat_model()

    # 获取嵌入模型
    embeddings = get_embeddings()

================================================================================
"""

import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# ================================================================================
# 聊天模型配置
# ================================================================================

# 默认聊天模型配置（用于大部分示例）
DEFAULT_CHAT_MODEL = {
    "model": "gpt-4o",
    "temperature": 0.7,
    "api_key": os.environ.get("ONE_API_KEY"),
    "base_url": os.environ.get("BASE_URL"),
}

# GLM 聊天模型配置（用于特定示例）
GLM_CHAT_MODEL = {
    "model": "glm-5",
    "temperature": 0.7,
    "api_key": os.environ.get("ONE_API_KEY"),
    "base_url": os.environ.get("BASE_URL"),
}

# Agent 专用配置（低温度，更确定性）
AGENT_CHAT_MODEL = {
    "model": "gpt-4o",
    "temperature": 0,
    "api_key": os.environ.get("ONE_API_KEY"),
    "base_url": os.environ.get("BASE_URL"),
}

# ================================================================================
# 嵌入模型配置
# ================================================================================

# 小型嵌入模型（更快速，更少 token）
SMALL_EMBEDDINGS = {
    "model": "text-embedding-3-small",
    "api_key": os.environ.get("ONE_API_KEY"),
    "base_url": os.environ.get("BASE_URL"),
}

# 大型嵌入模型（更精确，更多 token）
LARGE_EMBEDDINGS = {
    "model": "text-embedding-3-large",
    "api_key": os.environ.get("ONE_API_KEY"),
    "base_url": os.environ.get("BASE_URL"),
}

# ================================================================================
# 工具函数
# ================================================================================

def get_chat_model(config_name: str = "default", **kwargs) -> ChatOpenAI:
    """
    获取聊天模型实例

    Args:
        config_name: 配置名称，可选值:
            - "default": 默认配置 (gpt-4o)
            - "glm": GLM 配置 (glm-5)
            - "agent": Agent 配置 (gpt-4o, 低温度)
        **kwargs: 可选参数，用于覆盖配置中的值
            - streaming: 是否启用流式输出
            - callbacks: 回调函数列表
            - timeout: 超时时间（秒）
            - max_retries: 最大重试次数
            - 其他 ChatOpenAI 支持的参数

    Returns:
        ChatOpenAI: 聊天模型实例
    """
    configs = {
        "default": DEFAULT_CHAT_MODEL,
        "glm": GLM_CHAT_MODEL,
        "agent": AGENT_CHAT_MODEL,
    }
    config = configs.get(config_name, DEFAULT_CHAT_MODEL).copy()
    config.update(kwargs)
    return ChatOpenAI(**config)


def get_embeddings_model(size: str = "small") -> OpenAIEmbeddings:
    """
    获取嵌入模型实例

    Args:
        size: 嵌入模型大小，可选值:
            - "small": text-embedding-3-small
            - "large": text-embedding-3-large

    Returns:
        OpenAIEmbeddings: 嵌入模型实例
    """
    sizes = {
        "small": SMALL_EMBEDDINGS,
        "large": LARGE_EMBEDDINGS,
    }
    config = sizes.get(size, SMALL_EMBEDDINGS).copy()
    return OpenAIEmbeddings(**config)