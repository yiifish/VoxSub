<p align="center">
  <a href="#english">English</a> &nbsp;|&nbsp;
  <a href="#chinese">中文</a>
</p>

---

<h1 align="center">VoxSub</h1>
<p align="center"><em>Your voice, subbed.</em></p>

---

<div id="english">

## English

VoxSub is a fully offline toolset for video/audio subtitle generation. Drop in a video or audio file and get SRT subtitles in one shot — audio extraction, speech transcription, subtitle generation, and machine translation — all running locally with CUDA GPU acceleration powered by faster-whisper and CTranslate2.

### Features

| Feature | Description |
|---------|-------------|
| Audio Extraction | Extract WAV / MP3 from video files, with GPU-accelerated decoding |
| Speech to Subtitle | Auto-split long audio into 60s chunks, transcribe in parallel, output standard SRT |
| Full Pipeline | Drop a video → extract → transcribe → SRT subtitles, all in one pass |
| Subtitle Translation | One-click SRT translation via Google Translate |
| Real-time Progress | Server-Sent Events push live progress percentage to the Web UI |
| GPU / CPU Adaptive | Auto-detects CUDA and uses GPU float16, falls back to CPU int8 |

### Quick Start

#### Prerequisites

- Python 3.9+
- [FFmpeg](https://ffmpeg.org/) installed and on PATH
- NVIDIA GPU + CUDA driver (optional; CPU-only also works)

#### Install

```bash
git clone https://github.com/yiifish/VoxSub.git
cd VoxSub
pip install -r requirements.txt
```

#### Prepare the Whisper Model

VoxSub uses the CTranslate2 format, which runs significantly faster than the original Whisper. You have two options:

**Option A: Download a pre-converted model (recommended)**

faster-whisper provides pre-converted CTranslate2 models on HuggingFace. Choose one:

| Model | Size | Memory | Speed | Quality |
|-------|------|--------|-------|---------|
| `tiny` | ~150 MB | ~1 GB | Fastest | Basic |
| `base` | ~290 MB | ~1 GB | Fast | Moderate |
| `small` | ~970 MB | ~2 GB | Moderate | Good |
| `medium` | ~3.1 GB | ~5 GB | Moderate | Great |
| `large-v3` | ~3.1 GB | ~5 GB | Slower | Best |
| `turbo` | ~1.6 GB | ~3 GB | Fast | Excellent |

Download via Python:

```python
from faster_whisper import download_model
download_model("turbo", output_dir="models/whisper-turbo-ct2")
# Or: download_model("large-v3", output_dir="models/whisper-large-v3-ct2")
```

Then update `MODEL_PATH` in `server.py` to match the directory.

**Option B: Convert from HuggingFace yourself**

```bash
pip install ctranslate2 transformers
ct2-transformers-converter --model openai/whisper-turbo --output_dir models/whisper-turbo-ct2
ct2-transformers-converter --model openai/whisper-large-v3 --output_dir models/whisper-large-v3-ct2
```

The resulting directory should contain: `config.json`, `model.bin`, `tokenizer.json`, `vocabulary.json`, and `preprocessor_config.json`.

#### Run

```bash
# Windows — double-click start.bat, or:
python server.py
# Open http://127.0.0.1:8768
```

#### Standalone Services

```bash
python extract_server.py    # Audio extraction only → :8767
python subtitle.py          # Audio to SRT only     → :8765
```

### Project Structure

```
├── server.py           # Main server :8768 — Web UI + pipeline + translation
├── extract_server.py   # Audio extraction server :8767
├── subtitle.py         # Transcription server :8765
├── index.html          # Frontend (for separate deployment)
├── start.bat           # Windows one-click launcher
├── requirements.txt
└── models/
    └── whisper-turbo-ct2/   # CTranslate2 model directory
```

### API

Main server (`server.py` :8768):

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| POST | `/transcribe` | Audio/video to SRT |
| POST | `/transcribe-upload` | Upload file to SRT |
| POST | `/extract` | Video to audio |
| POST | `/extract-upload` | Upload file to audio |
| POST | `/pipeline` | Full pipeline: extract + transcribe + SRT |
| POST | `/translate` | SRT text translation |

All POST endpoints stream progress via Server-Sent Events.

### Configuration

Tweak these at the top of each source file:

| Parameter | File(s) | Description |
|-----------|---------|-------------|
| `PORT` | All files | Server port |
| `MODEL_PATH` | `server.py`, `subtitle.py` | Path to CTranslate2 model |
| `FFMPEG` | All files | FFmpeg binary path |
| `CHUNK_SEC` | `subtitle.py` | Audio segment duration (seconds) |
| `MAX_WORKERS` | `server.py` | Parallel transcription workers |

### Tech Stack

[![Python](https://img.shields.io/badge/python-3.9+-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-accelerated Whisper inference
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) — High-performance Transformer inference engine
- [FFmpeg](https://ffmpeg.org/) — Audio/video codec
- [deep-translator](https://github.com/nidhaloff/deep-translator) — Google Translate integration
- Python `http.server` + SSE — Lightweight HTTP + real-time streaming
- Lucide Icons — Web UI icons

### License

MIT

</div>

---

<div id="chinese">

## 中文

VoxSub 是一套本地离线运行的音视频字幕处理工具集。拖入视频或音频，一键完成：提取音频、语音转录、生成 SRT 字幕、翻译为中文。基于 faster-whisper + CTranslate2 引擎，支持 CUDA GPU 加速。

### 功能

| 功能 | 说明 |
|------|------|
| 音频提取 | 从视频中提取 WAV / MP3，支持 GPU 硬解码 |
| 语音转字幕 | 长音频自动切割为 60s 段落并行转录，输出标准 SRT |
| 全流程管线 | 视频拖入 → 提取 → 转录 → 字幕，一步到位 |
| 字幕翻译 | 调用 Google 翻译，SRT 一键英转中 |
| 实时进度 | Server-Sent Events 推送进度百分比，UI 实时更新 |
| GPU / CPU 自适应 | 检测到 CUDA 自动使用 GPU float16，否则回退 CPU int8 |

### 快速开始

#### 前置条件

- Python 3.9+
- [FFmpeg](https://ffmpeg.org/) 已安装并加入 PATH
- NVIDIA GPU + CUDA 驱动（可选，纯 CPU 也能跑）

#### 安装

```bash
git clone https://github.com/yiifish/VoxSub.git
cd VoxSub
pip install -r requirements.txt
```

#### 准备 Whisper 模型

VoxSub 使用 CTranslate2 格式的 Whisper 模型，推理速度比原始模型快数倍。有两种方式准备：

**方式 A：下载预转换模型（推荐）**

faster-whisper 在 HuggingFace 上提供了预转换的 CTranslate2 模型，可选以下规格：

| 模型 | 大小 | 内存 | 速度 | 质量 |
|------|------|------|------|------|
| `tiny` | ~150 MB | ~1 GB | 极快 | 基础 |
| `base` | ~290 MB | ~1 GB | 快 | 一般 |
| `small` | ~970 MB | ~2 GB | 中等 | 良好 |
| `medium` | ~3.1 GB | ~5 GB | 中等 | 优秀 |
| `large-v3` | ~3.1 GB | ~5 GB | 较慢 | 最佳 |
| `turbo` | ~1.6 GB | ~3 GB | 快 | 出色 |

Python 下载：

```python
from faster_whisper import download_model
download_model("turbo", output_dir="models/whisper-turbo-ct2")
# 或: download_model("large-v3", output_dir="models/whisper-large-v3-ct2")
```

下载后修改 `server.py` 中的 `MODEL_PATH` 指向对应目录。

**方式 B：自行从 HuggingFace 转换**

```bash
pip install ctranslate2 transformers
ct2-transformers-converter --model openai/whisper-turbo --output_dir models/whisper-turbo-ct2
ct2-transformers-converter --model openai/whisper-large-v3 --output_dir models/whisper-large-v3-ct2
```

转换后的目录应包含：`config.json`、`model.bin`、`tokenizer.json`、`vocabulary.json`、`preprocessor_config.json`。

#### 启动

```bash
# Windows 双击 start.bat，或直接运行：
python server.py
# 浏览器打开 http://127.0.0.1:8768
```

#### 独立服务（按需启动）

```bash
python extract_server.py    # 纯音频提取 → :8767
python subtitle.py          # 纯音频转字幕 → :8765
```

### 项目结构

```
├── server.py           # 主服务 :8768 — Web UI + 全流程 + 翻译
├── extract_server.py   # 音频提取服务 :8767
├── subtitle.py         # 字幕转录服务 :8765
├── index.html          # 前端页面（分离部署用）
├── start.bat           # Windows 一键启动脚本
├── requirements.txt
└── models/
    └── whisper-turbo-ct2/   # CTranslate2 模型目录
```

### API 端点

主服务 (`server.py` :8768)：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | Web UI |
| POST | `/transcribe` | 音频/视频转 SRT |
| POST | `/transcribe-upload` | 上传文件转 SRT |
| POST | `/extract` | 视频提取音频 |
| POST | `/extract-upload` | 上传文件提取音频 |
| POST | `/pipeline` | 全流程：提取 + 转录 + SRT |
| POST | `/translate` | SRT 文本翻译 |

所有 POST 接口通过 Server-Sent Events 返回实时进度，前端自动消费进度流。

### 配置

需要本地化调整的参数在源文件顶部：

| 参数 | 位置 | 说明 |
|------|------|------|
| `PORT` | 各文件 | 服务端口 |
| `MODEL_PATH` | `server.py` / `subtitle.py` | CTranslate2 模型路径 |
| `FFMPEG` | 各文件 | FFmpeg 可执行文件路径 |
| `CHUNK_SEC` | `subtitle.py` | 长音频切割秒数 |
| `MAX_WORKERS` | `server.py` | 并行转录线程数 |

### 技术选型

[![Python](https://img.shields.io/badge/python-3.9+-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 加速 Whisper
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) — Transformer 推理引擎
- [FFmpeg](https://ffmpeg.org/) — 音视频编解码
- [deep-translator](https://github.com/nidhaloff/deep-translator) — Google 翻译
- Python `http.server` + SSE — 轻量 HTTP + 实时推送
- Lucide Icons — Web UI 图标

### License

MIT

</div>
