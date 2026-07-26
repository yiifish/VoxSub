# VoiceCut

> 视频音频提取 & 语音转字幕 & 字幕翻译 —— 一站式本地处理工具

VoiceCut 是一套基于 **faster-whisper** 和 **FFmpeg** 的本地音视频处理工具集，提供视频音频提取、长音频/视频自动分段转录、字幕导出（SRT）以及字幕翻译功能。全部在本地运行，无需联网，可选 CUDA GPU 加速。

---

## 功能

- **视频提取音频** — 从视频文件中提取 WAV / MP3 音频，支持 GPU 硬解码
- **语音转字幕** — 基于 faster-whisper（CTranslate2 格式模型），将长音频自动切分为 60s 片段并行转录，输出标准 SRT 字幕
- **全流程管道** — 拖入视频一键完成：提取音频 → 转录 → 生成字幕
- **字幕翻译** — 内置 Google 翻译集成，一键将 SRT 字幕翻译为中文
- **Web UI** — 内置 HTTP 服务 + 现代化暗色 Web 界面，支持拖拽添加文件、实时进度推送
- **GPU / CPU 自动切换** — 检测到 CUDA 则自动使用 GPU float16，否则回退 CPU int8

## 架构

voicecut/
├── server.py          # 主服务（端口 8768）：Web UI + 全流程管道 + 翻译
├── extract_server.py  # 独立音频提取服务（端口 8767）
├── subtitle.py        # 独立字幕转录服务（端口 8765）
├── index.html         # 独立前端页面（可用于分离部署）
├── start.bat          # Windows 启动脚本（自动杀旧进程 + 启动 + 打开浏览器）
├── models/            # Whisper 模型文件（CTranslate2 格式）
│   └── whisper-turbo-ct2/
├── requirements.txt   # Python 依赖
└── .gitignore

## 快速开始

### 环境要求

- Python 3.9+
- FFmpeg（已安装并可调用）
- NVIDIA GPU + CUDA（可选，用于加速）

### 安装

`ash
# 1. 安装 Python 依赖
pip install -r requirements.txt

# 2. 准备 Whisper 模型
# 将 faster-whisper 兼容的 CTranslate2 模型放入 models/whisper-turbo-ct2/
# 可以使用以下命令转换（需安装 ctranslate2）：
# ct2-transformers-converter --model openai/whisper-turbo --output_dir models/whisper-turbo-ct2

# 3. 安装 FFmpeg
# Windows: 下载 ffmpeg 并配置 PATH，或修改 server.py 中的 FFMPEG 路径
`

### 使用

`ash
# Windows —— 双击 start.bat 或执行：
python server.py

# 浏览器打开 http://127.0.0.1:8768
`

在 Web 界面中：
1. 选择模式：**提取音频** / **转录字幕** / **全流程**
2. 输入或拖拽文件路径
3. 点击开始，等待处理完成
4. 下载 SRT 字幕文件

### 独立服务

`ash
# 仅音频提取
python extract_server.py    # http://127.0.0.1:8767

# 仅字幕转录（适用于已有音频文件）
python subtitle.py          # http://127.0.0.1:8765
`

## 配置说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| PORT | 8768 | 主服务端口 |
| MODEL_PATH | models/whisper-turbo-ct2 | CTranslate2 模型路径 |
| FFMPEG | 见各文件 | FFmpeg 可执行文件路径 |
| CHUNK_SEC | 60 | 长音频分段秒数 |
| MAX_WORKERS | 4 | 并行转录线程数 |

## 技术栈

- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** — CTranslate2 加速的 Whisper 推理
- **[CTranslate2](https://github.com/OpenNMT/CTranslate2)** — 高效 Transformer 推理引擎，支持 CUDA
- **[FFmpeg](https://ffmpeg.org/)** — 音视频编解码
- **[deep-translator](https://github.com/nidhaloff/deep-translator)** — Google 翻译集成
- **Python http.server** — 轻量 HTTP 服务 + Server-Sent Events 实时进度
- **Lucide Icons** — Web UI 图标

## License

MIT
