#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
OpenAI API - Audio Transcriptions（语音转文字）
================================================================================

接口：client.audio.transcriptions.create
用途：使用 Whisper 模型将音频文件转录为文字

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

3. language: 语言代码
   - 示例："zh", "en", "ja"
   - 可选，但建议指定以提高准确性
   - 常用语言代码：
     - zh: 中文
     - en: 英语
     - ja: 日语
     - ko: 韩语
     - fr: 法语
     - de: 德语
     - es: 西班牙语

4. prompt: 提示词
   - 示例："这是一段关于技术的对话"
   - 可选
   - 帮助模型理解上下文
   - 可以包含专有名词、术语等

5. response_format: 返回格式
   - 示例："json", "text", "srt", "vtt", "verbose_json"
   - 默认："json"
   - json: {"text": "..."}
   - text: 纯文本
   - srt: 字幕格式
   - vtt: WebVTT 字幕格式
   - verbose_json: 包含时间戳等详细信息

6. temperature: 采样温度
   - 范围：0-1
   - 示例：0
   - 默认：0
   - 建议使用较低值（0-0.2）

7. timestamp_granularities: 时间戳粒度
   - 示例：["word"], ["segment"]
   - 仅在 response_format="verbose_json" 时有效
   - word: 词级别时间戳
   - segment: 段落级别时间戳（默认）
================================================================================
"""

from config import create_client, MODELS

client = create_client()


def basic_transcription():
    """
    基础语音转文字示例
    """
    with open("audio.mp3", "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=MODELS["audio"]["whisper"],
            file=audio_file
        )
    
    print(f"转录结果: {transcript.text}")


def transcription_with_language():
    """
    指定语言的转录示例
    """
    with open("chinese_audio.mp3", "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=MODELS["audio"]["whisper"],
            file=audio_file,
            language="zh"  # 指定中文
        )
    
    print(f"中文转录结果: {transcript.text}")


def transcription_with_prompt():
    """
    使用提示词的转录示例
    
    提示词可以帮助模型：
    - 识别专有名词
    - 理解上下文
    - 保持一致的输出风格
    """
    with open("tech_talk.mp3", "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=MODELS["audio"]["whisper"],
            file=audio_file,
            language="zh",
            prompt="这是一段关于人工智能和机器学习的技术讨论。"
        )
    
    print(f"转录结果: {transcript.text}")


def transcription_with_timestamps():
    """
    带时间戳的转录示例
    """
    with open("audio.mp3", "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=MODELS["audio"]["whisper"],
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word"]  # 词级别时间戳
        )
    
    print(f"完整转录: {transcript.text}")
    print("\n词级别时间戳:")
    for segment in transcript.segments:
        print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")


def generate_srt_subtitle():
    """
    生成 SRT 字幕文件示例
    """
    with open("video.mp4", "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=MODELS["audio"]["whisper"],
            file=audio_file,
            response_format="srt"
        )
    
    # 保存为 SRT 文件
    with open("subtitle.srt", "w", encoding="utf-8") as f:
        f.write(transcript)
    
    print("字幕文件已保存: subtitle.srt")


def generate_vtt_subtitle():
    """
    生成 WebVTT 字幕文件示例
    """
    with open("video.mp4", "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=MODELS["audio"]["whisper"],
            file=audio_file,
            response_format="vtt"
        )
    
    # 保存为 VTT 文件
    with open("subtitle.vtt", "w", encoding="utf-8") as f:
        f.write(transcript)
    
    print("WebVTT 字幕文件已保存: subtitle.vtt")


if __name__ == "__main__":
    print("=" * 60)
    print("Audio Transcriptions 接口示例")
    print("=" * 60)
    print("""
使用前需要准备音频文件，如 audio.mp3

取消下方注释运行实际转录：
    """)
    
    # basic_transcription()
    # transcription_with_language()
    # transcription_with_timestamps()
    # generate_srt_subtitle()