#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
OpenAI API - Moderations（内容审核）
================================================================================

接口：client.moderations.create
用途：检测文本和图像是否包含不当内容，如仇恨言论、暴力、色情等

字段说明：
--------------------------------------------------------------------------------
1. model: 模型名
   - 示例："omni-moderation-latest", "text-moderation-stable", "text-moderation-007"
   - 必填
   - omni-moderation-latest: 最新多模态模型，支持文本和图像
   - text-moderation-stable: 稳定文本审核模型

2. input: 审核内容
   - 示例：
     - 文本："需要审核的文本"
     - 多模态：
       [
         {"type": "text", "text": "文本内容"},
         {"type": "image_url", "image_url": {"url": "https://..."}}
       ]
   - 必填
   - 最多 32 个输入项

审核类别说明：
- hate: 仇恨言论
- hate/threatening: 仇恨威胁
- harassment: 骚扰
- harassment/threatening: 骚扰威胁
- self-harm: 自残
- self-harm/instructions: 自残指导
- self-harm/intent: 自残意图
- sexual: 性内容
- sexual/minors: 未成年人性内容
- violence: 暴力
- violence/graphic: 血腥暴力

返回字段：
- flagged: 是否标记为不当内容
- categories: 各类别标记结果
- category_scores: 各类别置信度分数
================================================================================
"""

from openai import OpenAI

client = OpenAI(api_key="your-api-key")


def moderate_text():
    """
    文本审核示例
    """
    response = client.moderations.create(
        model="text-moderation-latest",
        input="这是一段需要审核的文本内容。"
    )
    
    result = response.results[0]
    
    print(f"是否标记: {result.flagged}")
    print(f"类别得分: {result.category_scores}")


def check_specific_categories():
    """
    检查特定类别
    """
    text = "这段文本包含一些可能敏感的内容"
    
    response = client.moderations.create(
        model="text-moderation-latest",
        input=text
    )
    
    result = response.results[0]
    
    print("审核结果:")
    print(f"  整体标记: {result.flagged}")
    print("\n各类别标记:")
    
    categories = result.categories
    for category, flagged in categories:
        if flagged:
            print(f"  ⚠️ {category}: {flagged}")


def get_category_scores():
    """
    获取类别置信度分数
    """
    text = "需要分析的文本内容"
    
    response = client.moderations.create(
        model="text-moderation-latest",
        input=text
    )
    
    result = response.results[0]
    scores = result.category_scores
    
    print("类别置信度分数:")
    for category, score in scores:
        print(f"  {category}: {score:.4f}")


def moderate_multiple_texts():
    """
    批量文本审核
    """
    texts = [
        "第一段需要审核的文本",
        "第二段需要审核的文本",
        "第三段需要审核的文本"
    ]
    
    response = client.moderations.create(
        model="text-moderation-latest",
        input=texts
    )
    
    for i, result in enumerate(response.results):
        print(f"文本 {i+1}: 标记={result.flagged}")


def moderate_image():
    """
    图像审核示例（多模态）
    """
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=[
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"}
            }
        ]
    )
    
    result = response.results[0]
    print(f"图像审核结果: 标记={result.flagged}")


def moderate_text_and_image():
    """
    文本+图像联合审核
    """
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=[
            {"type": "text", "text": "这是一段文字"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    )
    
    result = response.results[0]
    print(f"联合审核结果: 标记={result.flagged}")


def content_filter_example():
    """
    内容过滤器示例
    """
    def is_content_safe(text: str) -> tuple[bool, list]:
        """
        检查内容是否安全
        
        返回: (是否安全, 违规类别列表)
        """
        response = client.moderations.create(
            model="text-moderation-latest",
            input=text
        )
        
        result = response.results[0]
        
        if not result.flagged:
            return True, []
        
        # 获取违规类别
        violated = [cat for cat, flagged in result.categories if flagged]
        return False, violated
    
    # 测试
    test_texts = [
        "这是一段正常的文本",
        "可能敏感的内容需要被检测"
    ]
    
    for text in test_texts:
        safe, categories = is_content_safe(text)
        status = "✅ 安全" if safe else f"❌ 违规: {categories}"
        print(f"内容: {text[:30]}... -> {status}")


def detailed_moderation_report():
    """
    详细审核报告
    """
    text = "需要详细分析的文本内容"
    
    response = client.moderations.create(
        model="text-moderation-latest",
        input=text
    )
    
    result = response.results[0]
    
    print("=" * 60)
    print("内容审核报告")
    print("=" * 60)
    print(f"\n审核内容: {text}")
    print(f"整体标记: {'⚠️ 已标记' if result.flagged else '✅ 未标记'}")
    
    print("\n类别详情:")
    print("-" * 60)
    
    # 按分数排序
    scores = [(cat, score) for cat, score in result.category_scores]
    scores.sort(key=lambda x: x[1], reverse=True)
    
    for category, score in scores:
        flagged = getattr(result.categories, category, False)
        status = "⚠️" if flagged else "  "
        print(f"{status} {category}: {score:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Moderations 接口示例")
    print("=" * 60)
    
    print("""
可用示例函数：
- moderate_text(): 文本审核
- check_specific_categories(): 检查特定类别
- get_category_scores(): 获取类别置信度
- moderate_multiple_texts(): 批量审核
- moderate_image(): 图像审核
- moderate_text_and_image(): 联合审核
- content_filter_example(): 内容过滤器
- detailed_moderation_report(): 详细报告
    """)
    
    # 取消注释运行示例
    # moderate_text()
    # detailed_moderation_report()