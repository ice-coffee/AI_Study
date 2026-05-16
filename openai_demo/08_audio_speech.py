#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
OpenAI API - Audio Speech（文字转语音）
================================================================================

接口：client.audio.speech.create
用途：使用 TTS 模型将文本转换为语音

字段说明：
--------------------------------------------------------------------------------
1. model: 模型名
   - 示例：tts-1, tts-1-hd
   - 必填
   - tts-1: 标准质量，延迟更低
   - tts-1-hd: 高清质量，延迟稍高

2. input: 输入文本
   - 示例："你好，欢迎来到这里"
   - 必填
   - 最大 4096 字符

3. voice: 声音类型
   - 示例："alloy", "echo", "fable", "onyx", "nova", "shimmer"
   - 必填
   - alloy: 中性声音
   - echo: 男性声音
   - fable: 英国口音
   - onyx: 深沉男性声音
   - nova: 女性声音
   - shimmer: 柔和女性声音

4. response_format: 输出格式
   - 示例："mp3", "opus", "aac", "flac", "wav", "pcm"
   - 默认："mp3"
   - mp3: 通用格式
   - opus: 流媒体最佳
   - aac: 数字音频
   - flac: 无损压缩
   - wav: 无压缩
   - pcm: 原始音频

5. speed: 语速
   - 范围：0.25 - 4.0
   - 示例：1.0
   - 默认：1.0
================================================================================
"""

from config import create_client, MODELS

client = create_client()


def basic_speech():
    """
    基础文字转语音示例
    """
    response = client.audio.speech.create(
        model=MODELS["audio"]["tts1"],
        voice="alloy",
        input="你好，欢迎使用 OpenAI 文字转语音服务。"
    )
    
    # 保存音频文件
    response.stream_to_file("output.mp3")
    print("音频已保存: output.mp3")


def high_quality_speech():
    """
    高清质量语音示例
    """
    response = client.audio.speech.create(
        model=MODELS["audio"]["tts1_hd"],  # 高清模型
        voice="nova",       # 女性声音
        input="这是一个高清质量的语音示例，声音更加自然清晰。"
    )
    
    response.stream_to_file("output_hd.mp3")
    print("高清音频已保存: output_hd.mp3")


def different_voices():
    """
    不同声音类型示例
    """
    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    text = "这是一段测试文本，用于展示不同的声音效果。"
    
    for voice in voices:
        response = client.audio.speech.create(
            model=MODELS["audio"]["tts1"],
            voice=voice,
            input=text
        )
        filename = f"voice_{voice}.mp3"
        response.stream_to_file(filename)
        print(f"声音 {voice} 已保存: {filename}")


def adjust_speed():
    """
    调整语速示例
    """
    speeds = [0.5, 1.0, 1.5, 2.0]
    text = "这是一段测试不同语速的文本。"
    
    for speed in speeds:
        response = client.audio.speech.create(
            model=MODELS["audio"]["tts1"],
            voice="alloy",
            input=text,
            speed=speed
        )
        filename = f"speed_{speed}.mp3"
        response.stream_to_file(filename)
        print(f"语速 {speed}x 已保存: {filename}")


def save_as_flac():
    """
    保存为 FLAC 无损格式示例
    """
    response = client.audio.speech.create(
        model=MODELS["audio"]["tts1_hd"],
        voice="shimmer",
        input="这是无损音频格式示例。",
        response_format="flac"
    )
    
    response.stream_to_file("output.flac")
    print("FLAC 格式音频已保存: output.flac")


def streaming_speech():
    """
    流式处理语音示例
    """
    response = client.audio.speech.create(
        model=MODELS["audio"]["tts1"],
        voice="alloy",
        input="这是一段较长的文本，用于演示流式处理。当我们处理大量文本时，流式处理可以提高效率。"
    )
    
    # 流式写入文件
    with open("streaming_output.mp3", "wb") as f:
        for chunk in response.iter_bytes(chunk_size=1024):
            f.write(chunk)
    
    print("流式音频已保存: streaming_output.mp3")


if __name__ == "__main__":
    print("=" * 60)
    print("Audio Speech 接口示例")
    print("=" * 60)
    
    # 取消注释运行示例
    # basic_speech()
    # high_quality_speech()
    # different_voices()
    # adjust_speed()
    # save_as_flac()
    # streaming_speech()
    
    print("""
运行前请确保已设置 API Key

可用示例函数：
- basic_speech(): 基础文字转语音
- high_quality_speech(): 高清质量语音
- different_voices(): 不同声音类型
- adjust_speed(): 调整语速
- save_as_flac(): 保存为 FLAC 格式
- streaming_speech(): 流式处理语音
    """)