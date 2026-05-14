#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
OpenAI API - Files（文件管理）
================================================================================

接口：
- client.files.create: 上传文件
- client.files.list: 列出文件
- client.files.retrieve: 获取文件信息
- client.files.retrieve_content: 获取文件内容
- client.files.delete: 删除文件

用途：上传和管理文件，用于 Assistants、Fine-tuning、Batch 等功能

字段说明：
--------------------------------------------------------------------------------
【client.files.create】
1. file: 文件
   - 示例：open("training.jsonl", "rb")
   - 必填
   - 最大 512MB

2. purpose: 用途
   - 示例："assistants", "vision", "batch", "fine-tune"
   - 必填
   - assistants: 用于 Assistants API
   - vision: 用于图像理解
   - batch: 用于批量处理
   - fine-tune: 用于模型微调

【client.files.list】
1. purpose: 按用途筛选
   - 示例："fine-tune"

【client.files.retrieve】
1. file_id: 文件ID
   - 示例："file-abc123"
   - 必填

【client.files.delete】
1. file_id: 文件ID
   - 示例："file-abc123"
   - 必填
================================================================================
"""

from openai import OpenAI

client = OpenAI(api_key="your-api-key")


def upload_file():
    """
    上传文件示例
    """
    # 上传用于微调的文件
    with open("training_data.jsonl", "rb") as f:
        file = client.files.create(
            file=f,
            purpose="fine-tune"
        )
    
    print(f"文件ID: {file.id}")
    print(f"文件名: {file.filename}")
    print(f"字节数: {file.bytes}")
    print(f"创建时间: {file.created_at}")
    print(f"用途: {file.purpose}")


def upload_assistant_file():
    """
    上传用于 Assistants 的文件
    """
    with open("document.pdf", "rb") as f:
        file = client.files.create(
            file=f,
            purpose="assistants"
        )
    
    print(f"Assistants 文件ID: {file.id}")


def list_files():
    """
    列出所有文件
    """
    files = client.files.list()
    
    print("所有文件:")
    for file in files.data:
        print(f"  ID: {file.id}, 名称: {file.filename}, 用途: {file.purpose}")


def list_files_by_purpose():
    """
    按用途列出文件
    """
    files = client.files.list(purpose="fine-tune")
    
    print("微调文件:")
    for file in files.data:
        print(f"  ID: {file.id}, 名称: {file.filename}")


def retrieve_file():
    """
    获取文件信息
    """
    file_id = "file-abc123"
    
    file = client.files.retrieve(file_id=file_id)
    
    print(f"文件ID: {file.id}")
    print(f"文件名: {file.filename}")
    print(f"字节数: {file.bytes}")
    print(f"状态: {file.status}")


def retrieve_file_content():
    """
    获取文件内容
    """
    file_id = "file-abc123"
    
    content = client.files.retrieve_content(file_id=file_id)
    
    print(f"文件内容:\n{content}")


def delete_file():
    """
    删除文件
    """
    file_id = "file-abc123"
    
    result = client.files.delete(file_id=file_id)
    
    print(f"删除结果: {result.deleted}")


def create_training_file():
    """
    创建微调训练文件示例
    
    微调数据格式（JSONL）：
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    import json
    
    # 示例训练数据
    training_data = [
        {
            "messages": [
                {"role": "system", "content": "你是一个客服助手"},
                {"role": "user", "content": "我的订单什么时候发货？"},
                {"role": "assistant", "content": "您好，订单通常在付款后24小时内发货，您可以登录账户查看具体物流信息。"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "你是一个客服助手"},
                {"role": "user", "content": "我想退货怎么操作？"},
                {"role": "assistant", "content": "退货流程：1.登录账户->2.我的订单->3.申请退货->4.填写原因->5.提交审核。审核通过后按指引寄回即可。"}
            ]
        }
    ]
    
    # 写入 JSONL 文件
    with open("training.jsonl", "w", encoding="utf-8") as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print("训练文件已创建: training.jsonl")


if __name__ == "__main__":
    print("=" * 60)
    print("Files 接口示例")
    print("=" * 60)
    
    print("""
可用示例函数：
- upload_file(): 上传文件
- upload_assistant_file(): 上传用于 Assistants 的文件
- list_files(): 列出所有文件
- list_files_by_purpose(): 按用途列出文件
- retrieve_file(): 获取文件信息
- retrieve_file_content(): 获取文件内容
- delete_file(): 删除文件
- create_training_file(): 创建微调训练文件
    """)
    
    # 取消注释运行示例
    # create_training_file()
    # upload_file()
    # list_files()