# OpenAI API 示例代码目录

本目录包含 OpenAI API 所有主要接口的示例代码，每个文件对应一个接口。

## 文件列表

| 文件 | 接口 | 用途 |
|------|------|------|
| `01_chat_completions.py` | Chat Completions | 对话补全，核心接口 |
| `02_completions.py` | Completions | 文本补全（旧版） |
| `03_embeddings.py` | Embeddings | 向量嵌入 |
| `04_images_generate.py` | Images Generate | 图像生成 |
| `05_images_edit.py` | Images Edit | 图像编辑 |
| `06_audio_transcriptions.py` | Audio Transcriptions | 语音转文字 |
| `07_audio_translations.py` | Audio Translations | 语音翻译 |
| `08_audio_speech.py` | Audio Speech | 文字转语音 |
| `09_files.py` | Files | 文件管理 |
| `10_assistants.py` | Assistants | 创建助手 |
| `11_threads.py` | Threads | 对话线程 |
| `12_thread_messages.py` | Thread Messages | 线程消息 |
| `13_runs.py` | Runs | 运行执行 |
| `14_models.py` | Models | 模型管理 |
| `15_moderations.py` | Moderations | 内容审核 |
| `16_fine_tuning.py` | Fine-tuning | 模型微调 |
| `17_batch.py` | Batch | 批量处理 |

## 使用方法

### 1. 安装依赖

```bash
pip install openai
```

### 2. 设置 API Key

方式一：环境变量
```bash
export OPENAI_API_KEY="your-api-key"
```

方式二：代码中设置
```python
from openai import OpenAI
client = OpenAI(api_key="your-api-key")
```

### 3. 运行示例

```bash
python examples/01_chat_completions.py
```

## 学习顺序建议

1. **Chat Completions** - 必学，核心接口
2. **Embeddings** - 向量检索、RAG 基础
3. **Audio Speech/Transcriptions** - 语音相关
4. **Images** - 图像生成
5. **Assistants API** - 有状态对话（需要学习 Threads、Runs）
6. **Fine-tuning** - 模型定制
7. **Batch** - 批量处理，降低成本

## 注意事项

- 示例中的 `your-api-key` 需要替换为你的实际 API Key
- 部分示例需要准备音频、图片等文件
- 运行前取消注释相应的函数调用