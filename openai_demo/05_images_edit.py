#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
OpenAI API - Images Edit（图像编辑）
================================================================================

接口：client.images.edit
用途：对现有图像进行编辑（需要 DALL-E-2 模型）

字段说明：
--------------------------------------------------------------------------------
1. model: 模型名
   - 示例：dall-e-2
   - 必填
   - 目前仅支持 DALL-E-2

2. image: 原图像文件
   - 示例：open("image.png", "rb")
   - 必填
   - 必须是 PNG 格式
   - 必须是正方形
   - 最大 4MB
   - 最大尺寸 1024x1024

3. prompt: 编辑描述
   - 示例："把背景换成海滩"
   - 必填
   - 最大 1000 字符

4. mask: 遮罩文件
   - 示例：open("mask.png", "rb")
   - 可选
   - 必须是 PNG 格式
   - 透明区域将被编辑
   - 与原图像尺寸相同

5. n: 生成数量
   - 示例：1
   - 范围：1-10

6. size: 图像尺寸
   - 示例："256x256", "512x512", "1024x1024"
   - 默认：1024x1024

7. response_format: 返回格式
   - 示例："url", "b64_json"

8. user: 用户标识
   - 示例："user_123"
================================================================================
"""

from openai import OpenAI

client = OpenAI(api_key="your-api-key")


def edit_image_with_mask():
    """
    使用遮罩编辑图像示例
    
    遮罩说明：
    - 遮罩图像中透明（alpha=0）的区域会被编辑
    - 不透明区域保持不变
    """
    # 注意：需要准备原图和遮罩文件
    response = client.images.edit(
        model="dall-e-2",
        image=open("original.png", "rb"),      # 原图像
        mask=open("mask.png", "rb"),            # 遮罩图像
        prompt="一只金毛犬坐在草地上",           # 编辑描述
        n=1,
        size="1024x1024"
    )
    
    print(f"编辑后的图片链接: {response.data[0].url}")


def edit_image_without_mask():
    """
    不使用遮罩编辑图像示例
    
    不使用遮罩时，模型会根据prompt决定编辑哪些区域
    """
    response = client.images.edit(
        model="dall-e-2",
        image=open("original.png", "rb"),
        prompt="把背景换成日落海滩",
        n=1,
        size="1024x1024"
    )
    
    print(f"编辑后的图片链接: {response.data[0].url}")


def create_simple_mask():
    """
    创建简单遮罩示例

    这个函数演示如何使用 PIL 创建一个简单的遮罩
    遮罩中黑色区域将被编辑，白色区域保持不变
    """
    from PIL import Image, ImageDraw
    
    # 创建透明背景图像
    size = 1024
    mask = Image.new('RGBA', (size, size), (255, 255, 255, 255))
    
    # 在中心创建一个透明圆形区域（将被编辑）
    draw = ImageDraw.Draw(mask)
    center = size // 2
    radius = 200
    draw.ellipse(
        [center - radius, center - radius, center + radius, center + radius],
        fill=(0, 0, 0, 0)  # 透明
    )
    
    mask.save("mask.png")
    print("遮罩已创建: mask.png")


if __name__ == "__main__":
    print("=" * 60)
    print("Images Edit 接口示例")
    print("=" * 60)
    print("""
使用前需要准备：
1. original.png - 原图像（PNG格式，正方形，最大1024x1024）
2. mask.png - 遮罩图像（透明区域将被编辑）

取消下方注释运行实际编辑：
    """)
    
    # edit_image_with_mask()
    # edit_image_without_mask()
    
    # 创建遮罩示例
    # create_simple_mask()