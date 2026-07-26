# VoxSub

*Your voice, subbed.*

---

VoxSub 是一套本地离线运行的音视频字幕处理工具集。拖入视频或音频，一键完成：提取音频、语音转录、生成 SRT 字幕、翻译为中文。基于 faster-whisper + CTranslate2 引擎，支持 CUDA GPU 加速。

---

## 功能

| 功能 | 说明 |
|------|------|
| 音频提取 | 从视频中提取 WAV / MP3，支持 GPU 硬解码 |
| 语音转字幕 | 长音频自动切割为 60s 段落并行转录，输出标准 SRT |
| 全流程管线 | 视频拖入 → 提取 → 转录 → 字幕，一步到位 |
| 字幕翻译 | 调用 Google 翻译，SRT 一键英转中 |
| 实时进度 | Server-Sent Events 推送进度百分比，UI 实时更新 |
| GPU / CPU 自适应 | 检测到 CUDA 自动使用 GPU float16，否则回退 CPU int8 |

## 快速开始

### 前置条件

- Python 3.9+
- [FFmpeg](https://ffmpeg.org/) 已安装并加入 PATH
- NVIDIA GPU + CUDA 驱动（可选，纯 CPU 也能跑）

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/yiifish/VoxSub.git
cd VoxSub

# 2. 安装依赖
pip install -r requirements.txt

# 3. 准备 Whisper 模型（下载或自行转换）
#    faster-whisper 兼容的 CTranslate2 模型放入 models/whisper-turbo-ct2/
#    转换命令：
#    ct2-transformers-converter --model openai/whisper-turbo --output_dir models/whisper-turbo-ct2
```

### 启动

```bash
# 双击 start.bat（Windows）或直接运行：
python server.py

# 浏览器打开 http://127.0.0.1:8768
```

### 独立服务（按需启动）

```bash
python extract_server.py    # 纯音频提取 → http://127.0.0.1:8767
python subtitle.py          # 纯音频转字幕 → http://127.0.0.1:8765
```

## 项目结构

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

## API 端点

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

## 配置

需要本地化调整的参数在源文件顶部：

| 参数 | 位置 | 说明 |
|------|------|------|
| `PORT` | 各文件 | 服务端口 |
| `MODEL_PATH` | `server.py` / `subtitle.py` | CTranslate2 模型路径 |
| `FFMPEG` | 各文件 | FFmpeg 可执行文件路径 |
| `CHUNK_SEC` | `subtitle.py` | 长音频切割秒数 |
| `MAX_WORKERS` | `server.py` | 并行转录线程数 |

## 技术选型

[![Python](https://img.shields.io/badge/python-3.9+-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 加速 Whisper
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) — Transformer 推理引擎
- [FFmpeg](https://ffmpeg.org/) — 音视频编解码
- [deep-translator](https://github.com/nidhaloff/deep-translator) — Google 翻译
- Python `http.server` + SSE — 轻量 HTTP + 实时推送
- Lucide Icons — Web UI 图标

## License

MIT
