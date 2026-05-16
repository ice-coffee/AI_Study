#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
OpenAI API - Fine-tuning（模型微调）
================================================================================

接口：
- client.fine_tuning.jobs.create: 创建微调任务
- client.fine_tuning.jobs.list: 列出微调任务
- client.fine_tuning.jobs.retrieve: 获取任务状态
- client.fine_tuning.jobs.cancel: 取消任务
- client.fine_tuning.jobs.list_events: 获取任务事件

用途：使用自己的数据微调模型，定制模型行为

字段说明：
--------------------------------------------------------------------------------
【client.fine_tuning.jobs.create】
1. model: 基础模型
   - 示例：gpt-4o-2024-08-06, gpt-3.5-turbo-0125
   - 必填

2. training_file: 训练文件ID
   - 示例："file-abc123"
   - 必填
   - 文件格式：JSONL

3. validation_file: 验证文件ID
   - 示例："file-def456"
   - 可选

4. hyperparameters: 超参数
   - 示例：
     {
       "n_epochs": 3,
       "batch_size": 4,
       "learning_rate_multiplier": 0.1
     }
   - n_epochs: 训练轮数，默认 auto
   - batch_size: 批次大小，默认 auto
   - learning_rate_multiplier: 学习率乘数，默认 auto

5. suffix: 模型名称后缀
   - 示例："custom-model"
   - 生成的模型名格式：ft:gpt-3.5-turbo:org:suffix:id

训练数据格式（JSONL）：
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

任务状态：
- validating_files: 验证文件
- queued: 排队中
- running: 运行中
- succeeded: 成功
- failed: 失败
- cancelled: 已取消
================================================================================
"""

from config import create_client, MODELS
import json

client = create_client()


def create_training_data():
    """
    创建训练数据文件示例
    """
    # 示例训练数据
    training_examples = [
        {
            "messages": [
                {"role": "system", "content": "你是一个客服助手，负责回答产品相关问题。"},
                {"role": "user", "content": "你们的退货政策是什么？"},
                {"role": "assistant", "content": "我们的退货政策是购买后30天内可无理由退货，商品需保持原包装完好。"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "你是一个客服助手，负责回答产品相关问题。"},
                {"role": "user", "content": "如何查询我的订单状态？"},
                {"role": "assistant", "content": "您可以通过以下方式查询订单状态：1. 登录官网查看"我的订单"；2. 关注微信公众号查询；3. 致电客服热线400-xxx-xxxx。"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "你是一个客服助手，负责回答产品相关问题。"},
                {"role": "user", "content": "支持哪些支付方式？"},
                {"role": "assistant", "content": "我们支持以下支付方式：支付宝、微信支付、银联卡、信用卡以及货到付款。"}
            ]
        }
    ]
    
    # 写入 JSONL 文件
    with open("training_data.jsonl", "w", encoding="utf-8") as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print("训练数据已创建: training_data.jsonl")


def create_fine_tuning_job():
    """
    创建微调任务示例
    """
    # 先上传训练文件
    with open("training_data.jsonl", "rb") as f:
        file = client.files.create(file=f, purpose="fine-tune")
    
    print(f"训练文件已上传: {file.id}")
    
    # 创建微调任务
    job = client.fine_tuning.jobs.create(
        model=MODELS["fine_tune"]["default"],
        training_file=file.id,
        hyperparameters={
            "n_epochs": 3,
            "batch_size": 4,
            "learning_rate_multiplier": 0.1
        },
        suffix="customer-service"
    )
    
    print(f"微调任务ID: {job.id}")
    print(f"状态: {job.status}")


def create_fine_tuning_with_validation():
    """
    创建带验证集的微调任务
    """
    # 上传训练文件
    with open("training_data.jsonl", "rb") as f:
        train_file = client.files.create(file=f, purpose="fine-tune")
    
    # 上传验证文件
    with open("validation_data.jsonl", "rb") as f:
        val_file = client.files.create(file=f, purpose="fine-tune")
    
    job = client.fine_tuning.jobs.create(
        model=MODELS["fine_tune"]["default"],
        training_file=train_file.id,
        validation_file=val_file.id,
        hyperparameters={"n_epochs": 4}
    )
    
    print(f"微调任务ID: {job.id}")


def list_fine_tuning_jobs():
    """
    列出所有微调任务
    """
    jobs = client.fine_tuning.jobs.list()
    
    print("微调任务列表:")
    print("-" * 60)
    for job in jobs.data:
        print(f"ID: {job.id}")
        print(f"  模型: {job.model}")
        print(f"  状态: {job.status}")
        print(f"  创建时间: {job.created_at}")
        print()


def retrieve_fine_tuning_job():
    """
    获取微调任务状态
    """
    job_id = "ftjob-abc123"
    
    job = client.fine_tuning.jobs.retrieve(job_id=job_id)
    
    print(f"任务ID: {job.id}")
    print(f"状态: {job.status}")
    print(f"基础模型: {job.model}")
    
    if job.status == "succeeded":
        print(f"微调模型: {job.fine_tuned_model}")


def get_job_events():
    """
    获取任务事件日志
    """
    job_id = "ftjob-abc123"
    
    events = client.fine_tuning.jobs.list_events(job_id=job_id)
    
    print("任务事件:")
    print("-" * 60)
    for event in events.data:
        print(f"[{event.created_at}] {event.message}")


def cancel_fine_tuning_job():
    """
    取消微调任务
    """
    job_id = "ftjob-abc123"
    
    job = client.fine_tuning.jobs.cancel(job_id=job_id)
    
    print(f"任务已取消: {job.status}")


def use_fine_tuned_model():
    """
    使用微调后的模型
    """
    # 微调后的模型ID
    model_id = "ft:gpt-3.5-turbo:my-org:customer-service:abc123"
    
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": "你们的退货政策是什么？"}
        ]
    )
    
    print(f"回复: {response.choices[0].message.content}")


def estimate_training_cost():
    """
    估算训练成本
    """
    # 成本计算示例
    # GPT-3.5-turbo 微调成本:
    # 训练: $0.008 / 1K tokens
    # 输入: $0.003 / 1K tokens
    # 输出: $0.006 / 1K tokens
    
    training_tokens = 100000  # 假设训练数据 100K tokens
    epochs = 3
    
    training_cost = (training_tokens / 1000) * 0.008 * epochs
    
    print(f"训练估算:")
    print(f"  训练 tokens: {training_tokens:,}")
    print(f"  训练轮数: {epochs}")
    print(f"  预估成本: ${training_cost:.2f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Fine-tuning 接口示例")
    print("=" * 60)
    
    print("""
可用示例函数：
- create_training_data(): 创建训练数据
- create_fine_tuning_job(): 创建微调任务
- create_fine_tuning_with_validation(): 带验证集的微调
- list_fine_tuning_jobs(): 列出微调任务
- retrieve_fine_tuning_job(): 获取任务状态
- get_job_events(): 获取任务事件
- cancel_fine_tuning_job(): 取消任务
- use_fine_tuned_model(): 使用微调模型
- estimate_training_cost(): 估算训练成本
    """)
    
    # 取消注释运行示例
    # create_training_data()
    # estimate_training_cost()