#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
OpenAI API - Batch（批量处理）
================================================================================

接口：
- client.batches.create: 创建批量任务
- client.batches.list: 列出批量任务
- client.batches.retrieve: 获取任务状态
- client.batches.cancel: 取消任务

用途：批量处理大量请求，成本降低50%，适用于非实时场景

字段说明：
--------------------------------------------------------------------------------
【client.batches.create】
1. input_file_id: 输入文件ID
   - 示例："file-abc123"
   - 必填
   - 文件格式：JSONL，每行一个请求

2. endpoint: API 端点
   - 示例："/v1/chat/completions", "/v1/embeddings"
   - 必填
   - 目前支持：chat/completions, embeddings

3. completion_window: 完成时间窗口
   - 示例："24h"
   - 必填
   - 目前仅支持 "24h"

4. metadata: 元数据
   - 示例：{"key": "value"}

批量请求文件格式（JSONL）：
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}}

任务状态：
- validating: 验证中
- in_progress: 处理中
- finalizing: 完成中
- completed: 已完成
- expired: 已过期
- cancelling: 取消中
- failed: 失败
================================================================================
"""

from openai import OpenAI
import json

client = OpenAI(api_key="your-api-key")


def create_batch_input_file():
    """
    创建批量请求输入文件示例
    """
    requests = [
        {
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "用一句话介绍Python"}
                ]
            }
        },
        {
            "custom_id": "request-2",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "用一句话介绍JavaScript"}
                ]
            }
        },
        {
            "custom_id": "request-3",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "用一句话介绍Go语言"}
                ]
            }
        }
    ]
    
    # 写入 JSONL 文件
    with open("batch_requests.jsonl", "w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")
    
    print("批量请求文件已创建: batch_requests.jsonl")


def create_batch_job():
    """
    创建批量任务示例
    """
    # 上传输入文件
    with open("batch_requests.jsonl", "rb") as f:
        file = client.files.create(file=f, purpose="batch")
    
    print(f"输入文件ID: {file.id}")
    
    # 创建批量任务
    batch = client.batches.create(
        input_file_id=file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "测试批量任务"
        }
    )
    
    print(f"批量任务ID: {batch.id}")
    print(f"状态: {batch.status}")


def create_embeddings_batch():
    """
    创建嵌入批量任务示例
    """
    # 准备嵌入请求
    requests = [
        {
            "custom_id": f"embed-{i}",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "model": "text-embedding-3-small",
                "input": text
            }
        }
        for i, text in enumerate([
            "这是第一段文本",
            "这是第二段文本",
            "这是第三段文本"
        ])
    ]
    
    # 写入文件
    with open("embeddings_batch.jsonl", "w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")
    
    # 上传并创建任务
    with open("embeddings_batch.jsonl", "rb") as f:
        file = client.files.create(file=f, purpose="batch")
    
    batch = client.batches.create(
        input_file_id=file.id,
        endpoint="/v1/embeddings",
        completion_window="24h"
    )
    
    print(f"嵌入批量任务ID: {batch.id}")


def list_batches():
    """
    列出所有批量任务
    """
    batches = client.batches.list()
    
    print("批量任务列表:")
    print("-" * 60)
    for batch in batches.data:
        print(f"ID: {batch.id}")
        print(f"  状态: {batch.status}")
        print(f"  端点: {batch.endpoint}")
        print(f"  创建时间: {batch.created_at}")
        print()


def retrieve_batch():
    """
    获取批量任务状态
    """
    batch_id = "batch_abc123"
    
    batch = client.batches.retrieve(batch_id=batch_id)
    
    print(f"任务ID: {batch.id}")
    print(f"状态: {batch.status}")
    print(f"请求总数: {batch.request_counts.total}")
    print(f"已完成: {batch.request_counts.completed}")
    print(f"失败: {batch.request_counts.failed}")
    
    if batch.status == "completed":
        print(f"输出文件ID: {batch.output_file_id}")


def get_batch_results():
    """
    获取批量任务结果
    """
    batch_id = "batch_abc123"
    
    # 获取任务信息
    batch = client.batches.retrieve(batch_id=batch_id)
    
    if batch.status != "completed":
        print(f"任务未完成，当前状态: {batch.status}")
        return
    
    # 下载结果文件
    output_file_id = batch.output_file_id
    content = client.files.retrieve_content(file_id=output_file_id)
    
    # 解析结果
    for line in content.strip().split("\n"):
        result = json.loads(line)
        print(f"请求ID: {result['custom_id']}")
        print(f"响应: {result['response']['body']['choices'][0]['message']['content']}")
        print()


def cancel_batch():
    """
    取消批量任务
    """
    batch_id = "batch_abc123"
    
    batch = client.batches.cancel(batch_id=batch_id)
    
    print(f"任务已取消，状态: {batch.status}")


def batch_cost_comparison():
    """
    批量处理与实时处理成本对比
    """
    # 假设处理 10000 个请求
    num_requests = 10000
    tokens_per_request = 500
    
    total_tokens = num_requests * tokens_per_request
    
    # 实时处理成本 (GPT-4o)
    realtime_cost = (total_tokens / 1000) * 0.005  # 假设平均 $0.005/1K tokens
    
    # 批量处理成本 (50% 折扣)
    batch_cost = realtime_cost * 0.5
    
    print("成本对比:")
    print(f"  请求数量: {num_requests:,}")
    print(f"  总 tokens: {total_tokens:,}")
    print(f"  实时处理: ${realtime_cost:.2f}")
    print(f"  批量处理: ${batch_cost:.2f}")
    print(f"  节省: ${realtime_cost - batch_cost:.2f} ({(1 - batch_cost/realtime_cost)*100:.0f}%)")


if __name__ == "__main__":
    print("=" * 60)
    print("Batch 接口示例")
    print("=" * 60)
    
    print("""
可用示例函数：
- create_batch_input_file(): 创建批量请求文件
- create_batch_job(): 创建批量任务
- create_embeddings_batch(): 创建嵌入批量任务
- list_batches(): 列出批量任务
- retrieve_batch(): 获取任务状态
- get_batch_results(): 获取任务结果
- cancel_batch(): 取消任务
- batch_cost_comparison(): 成本对比
    """)
    
    batch_cost_comparison()