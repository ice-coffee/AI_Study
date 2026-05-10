#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
OpenAI API - Thread Messages（线程消息）
================================================================================

接口：client.beta.threads.messages.create
用途：向线程添加消息，支持文本和多模态内容

字段说明：
--------------------------------------------------------------------------------
1. thread_id: 线程ID
   - 示例："thread_abc123"
   - 必填

2. role: 角色
   - 示例："user"
   - 必填
   - 目前仅支持 "user"

3. content: 消息内容
   - 示例：
     - 文本："帮我计算123*456"
     - 多模态：
       [
         {"type": "text", "text": "这张图片是什么？"},
         {"type": "image_file", "image_file": {"file_id": "file-123"}}
       ]

4. attachments: 附件
   - 示例：
     [
       {"file_id": "file-123", "tools": [{"type": "file_search"}]}
     ]

5. metadata: 元数据
   - 示例：{"key": "value"}
   - 最多 16 个键值对

其他接口：
- client.beta.threads.messages.list: 列出线程消息
- client.beta.threads.messages.retrieve: 获取消息
- client.beta.threads.messages.update: 更新消息
================================================================================
"""

from openai import OpenAI

client = OpenAI(api_key="your-api-key")


def create_text_message():
    """
    创建文本消息示例
    """
    thread_id = "thread_abc123"
    
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content="请帮我分析一下数据分析的基本步骤"
    )
    
    print(f"消息ID: {message.id}")
    print(f"角色: {message.role}")
    print(f"内容: {message.content[0].text.value}")


def create_message_with_image():
    """
    创建带图片的消息示例
    """
    thread_id = "thread_abc123"
    file_id = "file-abc123"  # 已上传的图片文件ID
    
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=[
            {"type": "text", "text": "请描述这张图片的内容"},
            {"type": "image_file", "image_file": {"file_id": file_id}}
        ]
    )
    
    print(f"消息ID: {message.id}")


def create_message_with_attachment():
    """
    创建带附件的消息示例
    """
    thread_id = "thread_abc123"
    file_id = "file-abc123"
    
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content="请分析这份文档的主要内容",
        attachments=[
            {
                "file_id": file_id,
                "tools": [{"type": "file_search"}]
            }
        ]
    )
    
    print(f"消息ID: {message.id}")


def create_message_with_metadata():
    """
    创建带元数据的消息
    """
    thread_id = "thread_abc123"
    
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content="你好",
        metadata={
            "source": "mobile_app",
            "user_id": "12345"
        }
    )
    
    print(f"消息ID: {message.id}")
    print(f"元数据: {message.metadata}")


def list_messages():
    """
    列出线程消息
    """
    thread_id = "thread_abc123"
    
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    
    print("线程消息:")
    for msg in messages.data:
        content = msg.content[0].text.value if msg.content else "无内容"
        print(f"  [{msg.role}] {content[:50]}...")


def retrieve_message():
    """
    获取特定消息
    """
    thread_id = "thread_abc123"
    message_id = "msg_abc123"
    
    message = client.beta.threads.messages.retrieve(
        thread_id=thread_id,
        message_id=message_id
    )
    
    print(f"消息ID: {message.id}")
    print(f"内容: {message.content[0].text.value}")


def update_message():
    """
    更新消息元数据
    """
    thread_id = "thread_abc123"
    message_id = "msg_abc123"
    
    message = client.beta.threads.messages.update(
        thread_id=thread_id,
        message_id=message_id,
        metadata={"processed": "true"}
    )
    
    print(f"消息已更新: {message.metadata}")


if __name__ == "__main__":
    print("=" * 60)
    print("Thread Messages 接口示例")
    print("=" * 60)
    
    print("""
可用示例函数：
- create_text_message(): 创建文本消息
- create_message_with_image(): 创建带图片的消息
- create_message_with_attachment(): 创建带附件的消息
- create_message_with_metadata(): 创建带元数据的消息
- list_messages(): 列出线程消息
- retrieve_message(): 获取特定消息
- update_message(): 更新消息
    """)
    
    # 取消注释运行示例
    # create_text_message()
    # list_messages()