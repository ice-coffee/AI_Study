#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
OpenAI API - Audio Translations（语音翻译）
================================================================================

接口：client.audio.translations.create
用途：将音频文件翻译成英语（Whisper 模型目前仅支持翻译到英语）

字段说明：
--------------------------------------------------------------------------------
1. model: 模型名
   - 示例：whisper-1
   - 必填
   - 目前仅支持 whisper-1

2. file: 音频文件
   - 示例：open("audio.mp3", "rb")
   - 必填
   - 支持格式：mp3, mp4, mpeg, mpga, m4a, wav, webm
   - 最大 25MB

3. prompt: 提示词
   - 示例："Translate to English"
   - 可选
   - 帮助模型理解上下文
   - 建议使用英语提示词

4. response_format: 返回格式
   - 示例："json", "text", "srt", "vtt"
   - 默认："json"

5. temperature: 采样温度
   - 范围：0-1
   - 示例：0
   - 默认：0

注意：
- 翻译结果始终是英语
- 如果源音频是英语，相当于转录
================================================================================
"""

from config import create_client, MODELS

client = create_client()


def basic_translation():
    """
    基础语音翻译示例
    """
    with open("chinese_audio.mp3", "rb") as audio_file:
        translation = client.audio.translations.create(
            model=MODELS["audio"]["whisper"],
            file=audio_file
        )
    
    print(f"翻译结果: {translation.text}")


def translation_with_prompt():
    """
    使用提示词的翻译示例
    """
    with open("tech_talk.mp3", "rb") as audio_file:
        translation = client.audio.translations.create(
            model=MODELS["audio"]["whisper"],
            file=audio_file,
            prompt="This is a technical discussion about artificial intelligence and machine learning."
        )
    
    print(f"翻译结果: {translation.text}")


def translation_to_text():
    """
    以纯文本格式返回翻译结果
    """
    with open("speech.mp3", "rb") as audio_file:
        translation = client.audio.translations.create(
            model=MODELS["audio"]["whisper"],
            file=audio_file,
            response_format="text"
        )
    
    print(f"纯文本翻译: {translation}")


if __name__ == "__main__":
    print("=" * 60)
    print("Audio Translations 接口示例")
    print("=" * 60)
    print("""
使用前需要准备音频文件

注意：翻译结果始终是英语

取消下方注释运行实际翻译：
    """)
    
    # basic_translation()
    # translation_with_prompt()