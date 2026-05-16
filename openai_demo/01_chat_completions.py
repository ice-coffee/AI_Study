#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
OpenAI API - Chat Completions（对话补全）
================================================================================

接口：client.chat.completions.create
用途：核心对话接口，用于生成文本回复、多轮对话、Function Calling等

字段说明：
--------------------------------------------------------------------------------
1. model: 模型名
   - 示例：gpt-4o, gpt-4-turbo, gpt-3.5-turbo
   - 必填

2. messages: 对话消息数组
   - 必填
   - 格式：[{"role": "system/user/assistant/tool", "content": "..."}]
   - role 类型：
     - system: 系统指令，设定助手行为
     - user: 用户消息
     - assistant: 助手回复
     - tool: 工具调用返回结果
   - 多模态时 content 可为数组格式

3. temperature: 采样温度
   - 范围：0-2
   - 示例：0.7
   - 默认：1
   - 值越高输出越随机，值越低越确定

4. max_tokens: 最大生成token数
   - 示例：500

5. top_p: 核采样参数
   - 范围：0-1
   - 示例：0.9
   - 默认：1
   - 与temperature二选一

6. n: 生成的回复数量
   - 示例：1
   - 默认：1

7. stream: 是否流式输出
   - 示例：true, false

8. stop: 停止词
   - 示例：["END", "\n"]
   - 遇到这些词时停止生成

9. presence_penalty: 存在惩罚
   - 范围：-2 到 2
   - 示例：0.5
   - 正值鼓励谈论新话题

10. frequency_penalty: 频率惩罚
    - 范围：-2 到 2
    - 示例：0.5
    - 正值降低重复内容

11. logit_bias: token偏置
    - 示例：{"1234": -100}
    - 调整特定token的生成概率

12. user: 用户标识
    - 示例："user_123"
    - 用于监控和防滥用

13. response_format: 响应格式
    - 示例：{"type": "json_object"}, {"type": "text"}

14. seed: 随机种子
    - 示例：42
    - 使结果更可复现

15. tools: 工具定义数组
    - Function Calling 工具定义

16. tool_choice: 工具调用策略
    - 示例："auto", "none", {"type": "function", "function": {"name": "get_weather"}}

17. parallel_tool_calls: 是否并行调用多个工具
    - 示例：true
================================================================================
"""

from config import create_client, MODELS

# 初始化客户端
client = create_client()

def basic_chat():
    """
    基础对话示例
    """
    response = client.chat.completions.create(
        model=MODELS["chat"]["glm"],
        messages=[
            {"role": "system", "content": "你是一个专业的程序员"},
            {"role": "user", "content": "解释什么是 REST API"}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    print("回复内容:", response.choices[0].message.content)
    print("Token用量:", response.usage.total_tokens)


def streaming_chat():
    """
    流式输出示例
    """
    stream = client.chat.completions.create(
        model=MODELS["chat"]["default"],
        messages=[{"role": "user", "content": "写一首关于春天的诗"}],
        stream=True  # 开启流式
    )
    
    print("流式输出: ")
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


def function_calling():
    """
    Function Calling 示例
    """
    import json
    
    # 定义工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气",
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
    
    # 模拟天气函数
    def get_weather(city):
        return {"city": city, "temperature": "25°C", "weather": "晴天"}
    
    # 第一次调用 - 模型决定是否调用函数
    response = client.chat.completions.create(
        model=MODELS["chat"]["default"],
        messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
        tools=tools
    )
    
    message = response.choices[0].message
    
    # 检查是否需要调用工具
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        # 执行函数
        if function_name == "get_weather":
            result = get_weather(arguments["city"])
        
        # 第二次调用 - 返回函数结果
        response = client.chat.completions.create(
            model=MODELS["chat"]["default"],
            messages=[
                {"role": "user", "content": "北京今天天气怎么样？"},
                message,
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False)
                }
            ]
        )
        
        print("最终回复:", response.choices[0].message.content)


def json_output():
    """
    JSON格式输出示例
    """
    response = client.chat.completions.create(
        model=MODELS["chat"]["default"],
        messages=[
            {"role": "system", "content": "你是一个数据提取助手，以JSON格式返回结果"},
            {"role": "user", "content": "从以下文本中提取人名、地点和事件：张三在北京参加了技术会议"}
        ],
        response_format={"type": "json_object"}  # 强制JSON输出
    )
    
    data = json.loads(response.choices[0].message.content)
    print("JSON数据:", json.dumps(data, indent=2, ensure_ascii=False))


def multi_turn_chat():
    """
    多轮对话示例
    """
    messages = [
        {"role": "system", "content": "你是一个友好的助手"}
    ]
    
    def chat(user_input):
        messages.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model=MODELS["chat"]["default"],
            messages=messages
        )
        
        assistant_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
    
    print("第一轮:", chat("我叫张三"))
    print("第二轮:", chat("我叫什么名字？"))  # 模型会记住张三


if __name__ == "__main__":
    print("=" * 60)
    print("1. 基础对话")
    print("=" * 60)
    basic_chat()
    
    print("\n" + "=" * 60)
    print("2. 流式输出")
    print("=" * 60)
    streaming_chat()
    
    print("\n" + "=" * 60)
    print("3. Function Calling")
    print("=" * 60)
    function_calling()
    
    print("\n" + "=" * 60)
    print("4. JSON输出")
    print("=" * 60)
    json_output()
    
    print("\n" + "=" * 60)
    print("5. 多轮对话")
    print("=" * 60)
    multi_turn_chat()