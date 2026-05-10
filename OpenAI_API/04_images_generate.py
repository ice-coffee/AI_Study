#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
OpenAI API - Images Generate（图像生成）
================================================================================

接口：client.images.generate
用途：使用 DALL-E 模型根据文本描述生成图像

字段说明：
--------------------------------------------------------------------------------
1. model: 模型名
   - 示例：dall-e-3, dall-e-2
   - 必填

2. prompt: 图像描述
   - 示例："一只在草地上奔跑的金毛犬"
   - 必填
   - DALL-E-3 最大 4000 字符
   - DALL-E-2 最大 1000 字符

3. n: 生成数量
   - 示例：1
   - DALL-E-3 仅支持 1
   - DALL-E-2 支持 1-10

4. size: 图像尺寸
   - 示例："1024x1024", "1792x1024", "1024x1792"
   - DALL-E-3 支持：1024x1024, 1792x1024, 1024x1792
   - DALL-E-2 支持：256x256, 512x512, 1024x1024

5. quality: 图像质量
   - 示例："standard", "hd"
   - 仅 DALL-E-3 支持
   - hd 提供更细腻的细节

6. response_format: 返回格式
   - 示例："url", "b64_json"
   - url: 返回图片链接（2小时有效）
   - b64_json: 返回Base64编码

7. style: 图像风格
   - 示例："vivid", "natural"
   - 仅 DALL-E-3 支持
   - vivid: 更具戏剧性和超现实
   - natural: 更自然和真实

8. user: 用户标识
   - 示例："user_123"
================================================================================
"""

from openai import OpenAI
import base64

client = OpenAI(api_key="your-api-key")


def basic_image_generation():
    """
    基础图像生成示例
    """
    response = client.images.generate(
        model="dall-e-3",
        prompt="一只在草地上奔跑的金毛犬，油画风格",
        size="1024x1024",
        quality="standard",
        n=1
    )
    
    image_url = response.data[0].url
    print(f"图片链接: {image_url}")
    print(f"修订后的提示词: {response.data[0].revised_prompt}")


def hd_quality_image():
    """
    HD 高清图像生成示例
    """
    response = client.images.generate(
        model="dall-e-3",
        prompt="一座漂浮在云端的未来城市，霓虹灯光，科幻风格",
        size="1792x1024",  # 横向宽屏
        quality="hd",      # 高清质量
        style="vivid"      # 生动风格
    )
    
    print(f"高清图片链接: {response.data[0].url}")


def natural_style_image():
    """
    自然风格图像生成示例
    """
    response = client.images.generate(
        model="dall-e-3",
        prompt="一杯放在木质桌子上的咖啡，阳光透过窗户",
        size="1024x1024",
        quality="standard",
        style="natural"  # 自然风格
    )
    
    print(f"自然风格图片链接: {response.data[0].url}")


def save_image_locally():
    """
    保存图片到本地示例
    """
    import requests
    
    response = client.images.generate(
        model="dall-e-3",
        prompt="一只可爱的卡通猫咪",
        size="1024x1024",
        n=1
    )
    
    image_url = response.data[0].url
    
    # 下载并保存图片
    image_data = requests.get(image_url).content
    with open("generated_image.png", "wb") as f:
        f.write(image_data)
    
    print("图片已保存为 generated_image.png")


def base64_image():
    """
    获取 Base64 编码图片示例
    """
    response = client.images.generate(
        model="dall-e-3",
        prompt="一朵盛开的红玫瑰",
        size="1024x1024",
        response_format="b64_json"
    )
    
    # 解码并保存
    image_data = base64.b64decode(response.data[0].b64_json)
    with open("rose.png", "wb") as f:
        f.write(image_data)
    
    print("Base64 图片已解码并保存为 rose.png")


def dall_e_2_generation():
    """
    DALL-E-2 图像生成示例
    """
    response = client.images.generate(
        model="dall-e-2",
        prompt="A serene lake surrounded by mountains at sunset",
        size="1024x1024",
        n=1
    )
    
    print(f"DALL-E-2 图片链接: {response.data[0].url}")


if __name__ == "__main__":
    print("=" * 60)
    print("1. 基础图像生成")
    print("=" * 60)
    basic_image_generation()
    
    print("\n" + "=" * 60)
    print("2. HD高清图像生成")
    print("=" * 60)
    hd_quality_image()
    
    print("\n" + "=" * 60)
    print("3. 自然风格图像生成")
    print("=" * 60)
    natural_style_image()
    
    print("\n" + "=" * 60)
    print("4. DALL-E-2 图像生成")
    print("=" * 60)
    dall_e_2_generation()