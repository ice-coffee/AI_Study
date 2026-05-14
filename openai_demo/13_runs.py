#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
OpenAI API - Runs（运行）
================================================================================

接口：client.beta.threads.runs.create
用途：在线程上执行助手，生成回复

字段说明：
--------------------------------------------------------------------------------
1. thread_id: 线程ID
   - 示例："thread_abc123"
   - 必填

2. assistant_id: 助手ID
   - 示例："asst_abc123"
   - 必填

3. model: 覆盖助手模型
   - 示例："gpt-4o"
   - 可选

4. instructions: 覆盖助手指令
   - 示例："请用中文回答"
   - 可选

5. tools: 覆盖助手工具
   - 示例：[{"type": "code_interpreter"}]
   - 可选

6. metadata: 元数据
   - 示例：{"key": "value"}

7. temperature: 采样温度
   - 示例：0.7

8. top_p: 核采样参数
   - 示例：0.9

9. stream: 是否流式
   - 示例：true

运行状态说明：
- queued: 排队中
- in_progress: 执行中
- requires_action: 需要执行工具调用
- cancelling: 正在取消
- cancelled: 已取消
- failed: 失败
- completed: 完成
- expired: 超时

其他接口：
- client.beta.threads.runs.retrieve: 获取运行状态
- client.beta.threads.runs.list: 列出运行
- client.beta.threads.runs.cancel: 取消运行
- client.beta.threads.runs.submit_tool_outputs: 提交工具输出
================================================================================
"""

from openai import OpenAI
import time

client = OpenAI(api_key="your-api-key")


def create_run():
    """
    创建运行示例
    """
    thread_id = "thread_abc123"
    assistant_id = "asst_abc123"
    
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )
    
    print(f"运行ID: {run.id}")
    print(f"状态: {run.status}")


def run_with_overrides():
    """
    使用覆盖参数的运行
    """
    thread_id = "thread_abc123"
    assistant_id = "asst_abc123"
    
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        model="gpt-4o",
        instructions="请用简单易懂的语言回答，适合小学生理解。",
        temperature=0.5
    )
    
    print(f"运行ID: {run.id}")


def wait_for_run():
    """
    等待运行完成示例
    """
    thread_id = "thread_abc123"
    run_id = "run_abc123"
    
    while True:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )
        
        print(f"运行状态: {run.status}")
        
        if run.status == "completed":
            print("运行完成!")
            break
        elif run.status in ["failed", "cancelled", "expired"]:
            print(f"运行异常: {run.status}")
            break
        
        time.sleep(1)


def handle_tool_calls():
    """
    处理工具调用示例
    """
    thread_id = "thread_abc123"
    run_id = "run_abc123"
    
    run = client.beta.threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run_id
    )
    
    if run.status == "requires_action":
        tool_calls = run.required_action.submit_tool_outputs.tool_calls
        
        tool_outputs = []
        for tool_call in tool_calls:
            if tool_call.function.name == "get_weather":
                # 执行工具函数
                result = {"temperature": "25°C", "weather": "晴天"}
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": str(result)
                })
        
        # 提交工具输出
        run = client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run_id,
            tool_outputs=tool_outputs
        )
        
        print(f"已提交工具输出，状态: {run.status}")


def cancel_run():
    """
    取消运行示例
    """
    thread_id = "thread_abc123"
    run_id = "run_abc123"
    
    run = client.beta.threads.runs.cancel(
        thread_id=thread_id,
        run_id=run_id
    )
    
    print(f"运行已取消，状态: {run.status}")


def list_runs():
    """
    列出线程的所有运行
    """
    thread_id = "thread_abc123"
    
    runs = client.beta.threads.runs.list(thread_id=thread_id)
    
    print("线程运行记录:")
    for run in runs.data:
        print(f"  ID: {run.id}, 状态: {run.status}")


def streaming_run():
    """
    流式运行示例
    """
    thread_id = "thread_abc123"
    assistant_id = "asst_abc123"
    
    with client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id
    ) as stream:
        for event in stream:
            if event.event == "thread.message.delta":
                # 打印增量内容
                delta = event.data.delta.content[0].text.value
                print(delta, end="", flush=True)
            elif event.event == "thread.message.completed":
                print("\n完成!")


def complete_workflow():
    """
    完整工作流示例
    """
    # 1. 创建助手
    assistant = client.beta.assistants.create(
        name="示例助手",
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
        content="请用一句话介绍Python"
    )
    print(f"3. 添加消息")
    
    # 4. 创建运行
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    print(f"4. 创建运行: {run.id}")
    
    # 5. 等待完成
    while run.status in ["queued", "in_progress"]:
        time.sleep(0.5)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
    
    print(f"5. 运行状态: {run.status}")
    
    # 6. 获取回复
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    for msg in messages.data:
        if msg.role == "assistant":
            print(f"6. 助手回复: {msg.content[0].text.value}")
            break


if __name__ == "__main__":
    print("=" * 60)
    print("Runs 接口示例")
    print("=" * 60)
    
    print("""
可用示例函数：
- create_run(): 创建运行
- run_with_overrides(): 使用覆盖参数的运行
- wait_for_run(): 等待运行完成
- handle_tool_calls(): 处理工具调用
- cancel_run(): 取消运行
- list_runs(): 列出运行
- streaming_run(): 流式运行
- complete_workflow(): 完整工作流
    """)
    
    # 取消注释运行示例
    # complete_workflow()