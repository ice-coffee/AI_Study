#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
OpenAI Demo - 配置文件
================================================================================

统一管理所有 OpenAI API 的配置参数

使用方式:
    from config import create_client, MODELS

    # 创建客户端
    client = create_client()

    # 使用模型配置
    model = MODELS["chat"]["default"]  # gpt-4o

================================================================================
"""

import os
from openai import OpenAI


# ================================================================================
# 模型配置
# ================================================================================

# 聊天模型配置
CHAT_MODELS = {
    "default": "gpt-4o",
    "mini": "gpt-4o-mini",
    "glm": "glm-5",
}

# 文本补全模型
COMPLETION_MODELS = {
    "default": "gpt-3.5-turbo-instruct",
}

# 嵌入模型
EMBEDDING_MODELS = {
    "small": "text-embedding-3-small",
    "large": "text-embedding-3-large",
}

# 图像生成模型
IMAGE_MODELS = {
    "dalle3": "dall-e-3",
    "dalle2": "dall-e-2",
}

# 音频模型
AUDIO_MODELS = {
    "whisper": "whisper-1",
    "tts1": "tts-1",
    "tts1_hd": "tts-1-hd",
}

# 审核模型
MODERATION_MODELS = {
    "latest": "text-moderation-latest",
    "omni": "omni-moderation-latest",
}

# 微调模型
FINE_TUNE_MODELS = {
    "default": "gpt-3.5-turbo-0125",
}

# 响应模型
RESPONSE_MODELS = {
    "default": "gpt-4o",
    "mini": "gpt-4o-mini",
    "minimax": "minimax-m2.7",
}

# ================================================================================
# 快捷访问
# ================================================================================

MODELS = {
    "chat": CHAT_MODELS,
    "completion": COMPLETION_MODELS,
    "embedding": EMBEDDING_MODELS,
    "image": IMAGE_MODELS,
    "audio": AUDIO_MODELS,
    "moderation": MODERATION_MODELS,
    "fine_tune": FINE_TUNE_MODELS,
    "response": RESPONSE_MODELS,
}


# ================================================================================
# 客户端创建
# ================================================================================

def create_client() -> OpenAI:
    """
    创建 OpenAI 客户端

    Returns:
        OpenAI: 配置好的客户端实例
    """
    return OpenAI(
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )