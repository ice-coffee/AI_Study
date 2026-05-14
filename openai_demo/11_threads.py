#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
OpenAI API - Threads（对话线程）
================================================================================

接口：client.beta.threads.create
用途：创建对话线程，用于 Assistants API 中的多轮对话

字段说明：
--------------------------------------------------------------------------------
1. messages: 初始消息列表
   - 示例：[{"role": "user", "content": "你好"}]
   - 可选
   - 可以在创建线程时添加初始消息

2. tool_resources: 工具资源
   - 示例：
     {
       "code_interpreter": {"file_ids": ["file-123"]},
       "file_search": {"vector_store_ids": ["vs-123"]}
     }
   - 为线程中的工具提供资源

3. metadata: 元数据
   - 示例：{"user_id": "123"}
   - 最多 16 个键值对

注意：
- Threads 不包含模型信息，需要配合 Assistant 使用
- 一个 Thread 可以被多次 Run（运行）
================================================================================
"""

from openai import OpenAI

client = OpenAI(api_key="your-api-key")


def create_empty_thread():
    """
    创建空线程示例
    """
    thread = client.beta.threads.create()
    
    print(f"线程ID: {thread.id}")
    print(f"创建时间: {thread.created_at}")


def create_thread_with_message():
    """
    创建带初始消息的线程
    """
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": "请帮我分析一下这份数据"
            }
        ]
    )
    
    print(f"线程ID: {thread.id}")


def create_thread_with_file():
    """
    创建带文件资源的线程
    """
    # 假设已上传文件
    file_id = "file-abc123"
    
    thread = client.beta.threads.create(
        tool_resources={
            "code_interpreter": {
                "file_ids": [file_id]
            }
        }
    )
    
    print(f"线程ID: {thread.id}")
    print(f"已关联文件: {file_id}")


def create_thread_with_metadata():
    """
    创建带元数据的线程
    """
    thread = client.beta.threads.create(
        metadata={
            "user_id": "user_123",
            "session_id": "session_abc"
        }
    )
    
    print(f"线程ID: {thread.id}")
    print(f"元数据: {thread.metadata}")


def retrieve_thread():
    """
    获取线程信息
    """
    thread_id = "thread_abc123"
    
    thread = client.beta.threads.retrieve(thread_id=thread_id)
    
    print(f"线程ID: {thread.id}")
    print(f"创建时间: {thread.created_at}")
    print(f"元数据: {thread.metadata}")


def update_thread():
    """
    更新线程
    """
    thread_id = "thread_abc123"
    
    thread = client.beta.threads.update(
        thread_id=thread_id,
        metadata={"status": "completed"}
    )
    
    print(f"线程已更新: {thread.metadata}")


def delete_thread():
    """
    删除线程
    """
    thread_id = "thread_abc123"
    
    result = client.beta.threads.delete(thread_id=thread_id)
    
    print(f"删除结果: {result.deleted}")


def full_assistant_workflow():
    """
    完整的 Assistant 工作流示例
    """
    # 1. 创建助手
    assistant = client.beta.assistants.create(
        name="助手",
        instructions="你是一个友好的助手",
        model="gpt-4o"
    )
    print(f"1. 创建助手: {assistant.id}")
    
    # 2. 创建线程
    thread = client.beta.threads.create()
    print(f"2. 创建线程: {thread.id}")
    
    # 3. 添加消息
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="你好，请自我介绍"
    )
    print(f"3. 添加消息: {message.id}")
    
    # 4. 运行
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    print(f"4. 创建运行: {run.id}")
    
    # 5. 等待运行完成
    import time
    while run.status in ["queued", "in_progress"]:
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        print(f"   运行状态: {run.status}")
    
    # 6. 获取回复
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    for msg in messages.data:
        if msg.role == "assistant":
            print(f"5. 助手回复: {msg.content[0].text.value}")


if __name__ == "__main__":
    print("=" * 60)
    print("Threads 接口示例")
    print("=" * 60)
    
    print("""
可用示例函数：
- create_empty_thread(): 创建空线程
- create_thread_with_message(): 创建带初始消息的线程
- create_thread_with_file(): 创建带文件资源的线程
- create_thread_with_metadata(): 创建带元数据的线程
- retrieve_thread(): 获取线程信息
- update_thread(): 更新线程
- delete_thread(): 删除线程
- full_assistant_workflow(): 完整工作流示例
    """)
    
    # 取消注释运行示例
    # create_empty_thread()
    # full_assistant_workflow()