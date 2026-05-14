#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
OpenAI API - Completions（文本补全 - 旧版）
================================================================================

接口：client.completions.create
用途：旧版文本补全接口，适用于 gpt-3.5-turbo-instruct 等模型
     新项目建议使用 Chat Completions 接口

字段说明：
--------------------------------------------------------------------------------
1. model: 模型名
   - 示例：gpt-3.5-turbo-instruct
   - 必填

2. prompt: 输入提示
   - 示例："请写一首诗：" 或 ["写一首诗", "写一个故事"]
   - 必填
   - 可以是字符串或字符串数组

3. max_tokens: 最大生成token数
   - 示例：100

4. temperature: 采样温度
   - 范围：0-2
   - 示例：0.7

5. top_p: 核采样参数
   - 范围：0-1
   - 示例：0.9

6. n: 生成数量
   - 示例：1
   - 默认：1

7. stream: 流式输出
   - 示例：true, false

8. stop: 停止词
   - 示例：["\n", "END"]

9. presence_penalty: 存在惩罚
   - 范围：-2 到 2
   - 示例：0.5

10. frequency_penalty: 频率惩罚
    - 范围：-2 到 2
    - 示例：0.5

11. logit_bias: token偏置
    - 示例：{"1234": -100}

12. echo: 是否返回prompt
    - 示例：true
    - 默认：false

13. suffix: 补全后缀
    - 示例："结束"

14. user: 用户标识
    - 示例："user_123"
================================================================================
"""

from openai import OpenAI

client = OpenAI(api_key="your-api-key")


def basic_completion():
    """
    基础文本补全示例
    """
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt="请写一首关于春天的诗：\n",
        max_tokens=100,
        temperature=0.7
    )
    
    print("补全结果:", response.choices[0].text)


def multi_prompt_completion():
    """
    多提示补全示例
    """
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=[
            "翻译成英文：你好\n",
            "翻译成英文：谢谢\n"
        ],
        max_tokens=50,
        temperature=0.3
    )
    
    for i, choice in enumerate(response.choices):
        print(f"结果 {i+1}: {choice.text}")


def streaming_completion():
    """
    流式补全示例
    """
    stream = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt="写一个短故事：",
        max_tokens=200,
        stream=True
    )
    
    print("流式输出: ")
    for chunk in stream:
        if chunk.choices[0].text:
            print(chunk.choices[0].text, end="", flush=True)
    print()


def completion_with_stop():
    """
    使用停止词示例
    """
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt="列出三种水果：\n1.",
        max_tokens=50,
        stop=["\n\n", "4."]  # 遇到双换行或"4."时停止
    )
    
    print("结果:", response.choices[0].text)


if __name__ == "__main__":
    print("=" * 60)
    print("1. 基础文本补全")
    print("=" * 60)
    basic_completion()
    
    print("\n" + "=" * 60)
    print("2. 多提示补全")
    print("=" * 60)
    multi_prompt_completion()
    
    print("\n" + "=" * 60)
    print("3. 流式补全")
    print("=" * 60)
    streaming_completion()
    
    print("\n" + "=" * 60)
    print("4. 使用停止词")
    print("=" * 60)
    completion_with_stop()