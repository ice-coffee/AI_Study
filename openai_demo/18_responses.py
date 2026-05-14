#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
OpenAI API - Responses（响应式 API - 新一代接口）
================================================================================

接口：client.responses.create
用途：新一代统一接口，比 Chat Completions 更简洁，支持多模态、工具调用等

字段说明：
--------------------------------------------------------------------------------
1. model: 模型名
   - 示例：gpt-4o, gpt-4o-mini, o3, o4-mini
   - 必填

2. input: 输入内容
   - 示例："你好" 或 [{"role": "user", "content": "..."}]
   - 必填
   - 支持字符串或消息数组格式

3. instructions: 系统指令（可选）
   - 示例："你是一个专业的翻译助手"
   - 用于设定助手角色和行为

4. temperature: 采样温度
   - 范围：0-2
   - 示例：0.7
   - 默认：1

5. max_output_tokens: 最大输出token数
   - 示例：1024

6. top_p: 核采样参数
   - 范围：0-1
   - 示例：0.9

7. tools: 工具定义数组（可选）
   - Function Calling / Web Search / Computer Use 等

8. tool_choice: 工具调用策略
   - 示例："auto", "none", {"type": "function", "name": "get_weather"}

9. stream: 是否流式输出
   - 示例：True, False

10. previous_response_id: 上下文关联（可选）
    - 用于多轮对话，关联之前的响应

11. examples: 示例对话（可选）
    - 用于 few-shot 学习

12. reasoning: 推理参数（部分模型支持）
    - {"effort": "low", "generate_summary": "concise"}

13. audio: 音频参数（可选）
    - 支持语音输入输出

14. truncation: 截断策略
    - 控制输入过长时的行为

================================================================================
"""

import os
from openai import OpenAI

client = OpenAI(
                api_key=os.environ["ONE_API_KEY"],
                base_url=os.environ["BASE_URL"])


def basic_response():
    """
    基础响应示例 - 最简单的使用方式
    """
    response = client.chat.completions.create(
        model="minimax-m2.7",
        messages=[
            {"role": "system", "content": "你是一个专业的小说家"},
            {"role": "user", "content": "写一个关于独角兽的一句话睡前故事"}
        ]
    )

    print("响应结果:", response)
    # print("响应结果:", response.output_text)
    # print("响应ID:", response.id)
    # print("模型:", response.model)
    # print("完成原因:", response.finish_reason)


def response_with_instructions():
    """
    带系统指令的响应示例
    """
    response = client.responses.create(
        model="gpt-4o",
        instructions="你是一个专业的数据分析师，用简洁清晰的风格回答",
        input="解释什么是机器学习"
    )

    print("回复:", response.output_text)


def structured_input():
    """
    结构化输入示例 - 类似 messages 格式
    """
    response = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": "你是一个翻译助手"},
            {"role": "user", "content": "把下面的句子翻译成英文：今天天气真好"}
        ]
    )

    print("翻译结果:", response.output_text)


def streaming_response():
    """
    流式输出示例
    """
    stream = client.responses.create(
        model="gpt-4o-mini",
        input="写一首关于秋天的诗",
        stream=True
    )

    print("流式输出: ")
    full_content = ""
    for event in stream:
        if event.type == "content_delta":
            print(event.delta, end="", flush=True)
            full_content += event.delta if hasattr(event, 'delta') else str(event)
    print()


def tool_calling():
    """
    工具调用示例 - Function Calling
    """
    import json

    # 定义天气查询工具
    tools = [
        {
            "type": "function",
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
    ]

    # 模拟天气函数
    def get_weather(city):
        return f"{city}今天天气晴朗，气温25°C"

    # 第一次调用
    response = client.responses.create(
        model="gpt-4o",
        tools=tools,
        input="北京今天天气怎么样？"
    )

    print("第一次响应:", response.output_text)

    # 检查是否有工具调用
    if response.output[0].type == "function_call":
        func_call = response.output[0]
        function_name = func_call.name
        arguments = json.loads(func_call.arguments)

        print(f"\n调用工具: {function_name}")
        print(f"参数: {arguments}")

        # 执行函数
        result = get_weather(arguments["city"])

        # 第二次调用 - 传入工具结果
        response = client.responses.create(
            model="gpt-4o",
            tools=tools,
            previous_response_id=response.id,
            input=[
                {
                    "role": "system",
                    "content": f"工具返回结果: {result}"
                },
                {
                    "role": "user",
                    "content": "基于工具返回的结果，用中文回答用户"
                }
            ]
        )

        print("\n最终回复:", response.output_text)


def web_search_tool():
    """
    Web Search 工具示例 - 让 AI 搜索网页
    """
    tools = [
        {
            "type": "web_search",
            "name": "web_search",
            "description": "搜索互联网获取最新信息"
        }
    ]

    response = client.responses.create(
        model="gpt-4o",
        tools=tools,
        input="今天有什么重大科技新闻？"
    )

    # 检查输出类型
    for output in response.output:
        if output.type == "message":
            print("回复:", output.content[0].text)
        elif output.type == "reasoning":
            print("推理过程:", output.summary)


def multi_turn_conversation():
    """
    多轮对话示例 - 使用 previous_response_id 保持上下文
    """
    # 第一轮
    response1 = client.responses.create(
        model="gpt-4o-mini",
        instructions="你是一个友好的助手",
        input="我叫张三，请记住"
    )
    print("第一轮:", response1.output_text)

    # 第二轮 - 关联第一轮响应
    response2 = client.responses.create(
        model="gpt-4o-mini",
        instructions="你是一个友好的助手",
        input="我叫什么名字？",
        previous_response_id=response1.id
    )
    print("第二轮:", response2.output_text)

    # 第三轮
    response3 = client.responses.create(
        model="gpt-4o-mini",
        instructions="你是一个友好的助手",
        input="我喜欢吃苹果，你呢？",
        previous_response_id=response2.id
    )
    print("第三轮:", response3.output_text)


def json_mode():
    """
    JSON 模式输出示例
    """
    response = client.responses.create(
        model="gpt-4o",
        instructions="你是一个数据提取助手，输出纯JSON格式",
        input="从以下文本中提取信息：张三是一名28岁的软件工程师，工作于北京的一家互联网公司",
        # Responses API 使用 response_format 指定输出格式
        text={
            "format": {
                "type": "json_object",
                "value": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "profession": {"type": "string"},
                        "city": {"type": "string"}
                    }
                }
            }
        }
    )

    print("JSON输出:", response.output_text)


def with_temperature():
    """
    控制输出随机性示例
    """
    # 创意模式 - 高温度
    response_creative = client.responses.create(
        model="gpt-4o-mini",
        input="用创意的方式解释什么是人工智能",
        temperature=1.5
    )
    print("创意模式:", response_creative.output_text)

    # 精确模式 - 低温度
    response_precise = client.responses.create(
        model="gpt-4o-mini",
        input="解释什么是人工智能",
        temperature=0.2
    )
    print("精确模式:", response_precise.output_text)


def access_response_object():
    """
    访问完整响应对象示例
    """
    response = client.responses.create(
        model="gpt-4o-mini",
        input="解释量子计算的基本原理"
    )

    # 完整响应对象
    print("=== 响应对象详解 ===")
    print(f"ID: {response.id}")
    print(f"模型: {response.model}")
    print(f"创建时间: {response.created_at}")
    print(f"完成原因: {response.finish_reason}")
    print(f"输出内容: {response.output_text}")

    # 详细输出结构
    print("\n=== 输出结构 ===")
    for idx, output in enumerate(response.output):
        print(f"输出项 {idx + 1}:")
        print(f"  类型: {output.type}")
        if hasattr(output, 'id'):
            print(f"  ID: {output.id}")

    # 用量信息
    print("\n=== Token 用量 ===")
    print(f"输入Token: {response.usage.input_tokens}")
    print(f"输出Token: {response.usage.output_tokens}")
    print(f"总Token: {response.usage.total_tokens}")


if __name__ == "__main__":
    print("=" * 60)
    print("1. 基础响应")
    print("=" * 60)
    basic_response()

    print("\n" + "=" * 60)
    print("2. 带系统指令的响应")
    print("=" * 60)
    response_with_instructions()

    print("\n" + "=" * 60)
    print("3. 结构化输入")
    print("=" * 60)
    structured_input()

    print("\n" + "=" * 60)
    print("4. 流式输出")
    print("=" * 60)
    streaming_response()

    print("\n" + "=" * 60)
    print("5. 工具调用 (Function Calling)")
    print("=" * 60)
    tool_calling()

    print("\n" + "=" * 60)
    print("6. Web Search 工具")
    print("=" * 60)
    web_search_tool()

    print("\n" + "=" * 60)
    print("7. 多轮对话")
    print("=" * 60)
    multi_turn_conversation()

    print("\n" + "=" * 60)
    print("8. JSON 模式输出")
    print("=" * 60)
    json_mode()

    print("\n" + "=" * 60)
    print("9. 控制输出随机性")
    print("=" * 60)
    with_temperature()

    print("\n" + "=" * 60)
    print("10. 访问响应对象详解")
    print("=" * 60)
    access_response_object()