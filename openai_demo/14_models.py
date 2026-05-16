#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
OpenAI API - Models（模型管理）
================================================================================

接口：
- client.models.list: 列出所有可用模型
- client.models.retrieve: 获取特定模型信息
- client.models.delete: 删除微调模型

用途：查询可用模型、获取模型详情、删除自定义模型

字段说明：
--------------------------------------------------------------------------------
【client.models.list】
无参数

【client.models.retrieve】
1. model: 模型ID
   - 示例："gpt-4o", "ft:gpt-3.5-turbo:my-org:custom_suffix:id"
   - 必填

【client.models.delete】
1. model: 模型ID（微调模型）
   - 示例："ft:gpt-3.5-turbo:my-org:custom_suffix:id"
   - 必填
   - 只能删除自己微调的模型

模型对象字段：
- id: 模型ID
- object: 对象类型，固定为 "model"
- created: 创建时间戳
- owned_by: 所有者
- permission: 权限信息
- root: 根模型
- parent: 父模型
================================================================================
"""

from config import create_client, MODELS

client = create_client()


def list_all_models():
    """
    列出所有可用模型
    """
    models = client.models.list()
    
    print("可用模型列表:")
    print("-" * 60)
    
    for model in models.data:
        print(f"ID: {model.id}")
        print(f"  所有者: {model.owned_by}")
        print(f"  创建时间: {model.created}")
        print()


def list_gpt_models():
    """
    筛选 GPT 模型
    """
    models = client.models.list()
    
    print("GPT 系列模型:")
    print("-" * 60)
    
    gpt_models = [m for m in models.data if "gpt" in m.id.lower()]
    for model in sorted(gpt_models, key=lambda x: x.id):
        print(f"  {model.id}")


def list_embedding_models():
    """
    筛选嵌入模型
    """
    models = client.models.list()
    
    print("嵌入模型:")
    print("-" * 60)
    
    embedding_models = [m for m in models.data if "embedding" in m.id.lower()]
    for model in embedding_models:
        print(f"  {model.id}")


def retrieve_model():
    """
    获取特定模型信息
    """
    model_id = "gpt-4o"
    
    model = client.models.retrieve(model=model_id)
    
    print(f"模型ID: {model.id}")
    print(f"所有者: {model.owned_by}")
    print(f"创建时间: {model.created}")
    print(f"对象类型: {model.object}")


def retrieve_whisper_model():
    """
    获取 Whisper 模型信息
    """
    model = client.models.retrieve(model=MODELS["audio"]["whisper"])
    
    print(f"模型ID: {model.id}")
    print(f"所有者: {model.owned_by}")


def retrieve_dalle_model():
    """
    获取 DALL-E 模型信息
    """
    model = client.models.retrieve(model=MODELS["image"]["dalle3"])
    
    print(f"模型ID: {model.id}")
    print(f"所有者: {model.owned_by}")


def delete_fine_tuned_model():
    """
    删除微调模型
    """
    # 注意：只能删除自己微调的模型
    model_id = "ft:gpt-3.5-turbo:my-org:custom_suffix:id"
    
    result = client.models.delete(model=model_id)
    
    print(f"删除结果: {result.deleted}")
    print(f"模型ID: {result.id}")


def model_info_summary():
    """
    常用模型信息汇总
    """
    models_info = {
        "gpt-4o": {
            "description": "最新旗舰模型，支持文本、图像、音频",
            "context": "128K",
            "推荐用途": "通用对话、多模态任务"
        },
        "gpt-4-turbo": {
            "description": "GPT-4 增强版",
            "context": "128K",
            "推荐用途": "复杂推理任务"
        },
        "gpt-3.5-turbo": {
            "description": "快速经济模型",
            "context": "16K",
            "推荐用途": "简单对话、快速响应"
        },
        "text-embedding-3-small": {
            "description": "高效嵌入模型",
            "dimensions": "1536",
            "推荐用途": "向量搜索、RAG"
        },
        "text-embedding-3-large": {
            "description": "高质量嵌入模型",
            "dimensions": "3072",
            "推荐用途": "高精度向量检索"
        },
        "dall-e-3": {
            "description": "图像生成模型",
            "sizes": "1024x1024, 1792x1024, 1024x1792",
            "推荐用途": "图像生成"
        },
        "whisper-1": {
            "description": "语音识别模型",
            "languages": "99+ languages",
            "推荐用途": "语音转文字"
        },
        "tts-1": {
            "description": "文字转语音模型",
            "voices": "alloy, echo, fable, onyx, nova, shimmer",
            "推荐用途": "语音合成"
        }
    }
    
    print("常用模型信息汇总:")
    print("=" * 70)
    for model_id, info in models_info.items():
        print(f"\n{model_id}:")
        for key, value in info.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    print("=" * 60)
    print("Models 接口示例")
    print("=" * 60)
    
    # 取消注释运行示例
    # list_all_models()
    # list_gpt_models()
    # retrieve_model()
    # model_info_summary()
    
    print("""
可用示例函数：
- list_all_models(): 列出所有模型
- list_gpt_models(): 筛选GPT模型
- list_embedding_models(): 筛选嵌入模型
- retrieve_model(): 获取特定模型
- delete_fine_tuned_model(): 删除微调模型
- model_info_summary(): 常用模型汇总
    """)
    
    model_info_summary()