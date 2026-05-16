#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
OpenAI API - Assistants（助手）
================================================================================

接口：client.beta.assistants.create
用途：创建具有持久状态的 AI 助手，支持代码解释器、文件搜索、函数调用等

字段说明：
--------------------------------------------------------------------------------
1. model: 模型名
   - 示例：gpt-4o, gpt-4-turbo
   - 必填

2. name: 助手名称
   - 示例："数学导师"
   - 最大 256 字符

3. description: 描述
   - 示例："一个帮助解决数学问题的助手"
   - 最大 512 字符

4. instructions: 指令
   - 示例："你是一个专业的数学老师"
   - 最大 256000 字符
   - 定义助手的行为和角色

5. tools: 工具列表
   - 示例：
     [
       {"type": "code_interpreter"},      # 代码解释器
       {"type": "file_search"},            # 文件搜索
       {"type": "function", "function": {...}}  # 自定义函数
     ]
   - 最多 128 个工具

6. tool_resources: 工具资源
   - 示例：
     {
       "code_interpreter": {"file_ids": ["file-123"]},
       "file_search": {"vector_store_ids": ["vs-123"]}
     }

7. metadata: 元数据
   - 示例：{"key": "value"}
   - 最多 16 个键值对

8. temperature: 采样温度
   - 范围：0-2
   - 示例：0.7

9. top_p: 核采样参数
   - 范围：0-1
   - 示例：0.9

10. response_format: 响应格式
    - 示例：{"type": "text"}, {"type": "json_object"}

注意：Assistants API 需要配合 Threads 和 Runs 使用
================================================================================
"""

from config import create_client, MODELS

client = create_client()


def create_basic_assistant():
    """
    创建基础助手示例
    """
    assistant = client.beta.assistants.create(
        name="数学导师",
        instructions="你是一个专业的数学老师，擅长用简单易懂的方式解释数学概念。",
        model=MODELS["chat"]["default"]
    )
    
    print(f"助手ID: {assistant.id}")
    print(f"助手名称: {assistant.name}")


def create_assistant_with_code_interpreter():
    """
    创建带代码解释器的助手
    """
    assistant = client.beta.assistants.create(
        name="数据分析助手",
        instructions="你是一个数据分析专家，可以使用 Python 进行数据处理和可视化。",
        model=MODELS["chat"]["default"],
        tools=[{"type": "code_interpreter"}]
    )
    
    print(f"助手ID: {assistant.id}")
    print(f"已启用代码解释器")


def create_assistant_with_file_search():
    """
    创建带文件搜索的助手
    """
    # 首先创建向量存储
    vector_store = client.beta.vector_stores.create(name="文档库")
    
    assistant = client.beta.assistants.create(
        name="文档问答助手",
        instructions="你是一个文档问答助手，根据上传的文档回答用户问题。",
        model=MODELS["chat"]["default"],
        tools=[{"type": "file_search"}],
        tool_resources={
            "file_search": {
                "vector_store_ids": [vector_store.id]
            }
        }
    )
    
    print(f"助手ID: {assistant.id}")
    print(f"向量存储ID: {vector_store.id}")


def create_assistant_with_function():
    """
    创建带自定义函数的助手
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]
    
    assistant = client.beta.assistants.create(
        name="天气助手",
        instructions="你是一个天气助手，帮助用户查询天气信息。",
        model=MODELS["chat"]["default"],
        tools=tools
    )
    
    print(f"助手ID: {assistant.id}")


def list_assistants():
    """
    列出所有助手
    """
    assistants = client.beta.assistants.list()
    
    print("所有助手:")
    for assistant in assistants.data:
        print(f"  ID: {assistant.id}, 名称: {assistant.name}")


def retrieve_assistant():
    """
    获取助手信息
    """
    assistant_id = "asst_abc123"
    
    assistant = client.beta.assistants.retrieve(assistant_id=assistant_id)
    
    print(f"助手名称: {assistant.name}")
    print(f"指令: {assistant.instructions}")
    print(f"工具: {assistant.tools}")


def update_assistant():
    """
    更新助手
    """
    assistant_id = "asst_abc123"
    
    assistant = client.beta.assistants.update(
        assistant_id=assistant_id,
        instructions="更新后的指令：你是一个更友好的助手。",
        name="新名称"
    )
    
    print(f"助手已更新: {assistant.name}")


def delete_assistant():
    """
    删除助手
    """
    assistant_id = "asst_abc123"
    
    result = client.beta.assistants.delete(assistant_id=assistant_id)
    
    print(f"删除结果: {result.deleted}")


if __name__ == "__main__":
    print("=" * 60)
    print("Assistants 接口示例")
    print("=" * 60)
    
    print("""
可用示例函数：
- create_basic_assistant(): 创建基础助手
- create_assistant_with_code_interpreter(): 创建带代码解释器的助手
- create_assistant_with_file_search(): 创建带文件搜索的助手
- create_assistant_with_function(): 创建带自定义函数的助手
- list_assistants(): 列出所有助手
- retrieve_assistant(): 获取助手信息
- update_assistant(): 更新助手
- delete_assistant(): 删除助手
    """)
    
    # 取消注释运行示例
    # create_basic_assistant()
    # list_assistants()