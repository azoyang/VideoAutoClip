### <p align="center">「简体中文 | [English](./README_EN.md)」</p>
# VideoAutoClip

VideoAutoClip 是一个全自动化的短剧推广视频生成工具，旨在通过 AI 技术简化从原始素材到成品推广视频的制作流程。项目集成了网盘下载、语音识别 (ASR)、大语言模型 (LLM) 内容分析以及自动化视频剪辑功能。

![Task UI](docs/images/task.png)

## ✨ 核心功能

- **全自动流水线**：输入百度网盘链接，自动完成下载、合并、识别、剪辑、字幕挂载、BGM混合。
- **AI 智能分析**：
  - **ASR**: 使用 FunASR (Qwen) 将音频转为带时间轴的字幕。
  - **LLM**: 利用大模型 (Qwen-plus 等) 分析剧情，提取“高能片段”和“黄金3秒”钩子。
- **任务队列管理**：基于 SQLite 持久化存储，支持任务排队、并发控制、断点续传和状态追踪。
- **可视化界面**：基于 Gradio 的 Web UI，支持任务管理、日志查看、视频预览和配置管理。
- **多平台支持**：支持 Docker 容器化部署，开箱即用。

## 🚀 快速开始

### 方式一：Docker 部署 (推荐)

确保已安装 [Docker](https://www.docker.com/) 和 [Docker Compose](https://docs.docker.com/compose/)。

1. **克隆项目**
   ```bash
   git clone https://github.com/azoyang/VideoAutoClip.git
   cd VideoAutoClip
   ```

2. **启动服务**
   ```bash
   cd docker
   docker build -t video-autoclip:0.0.1 .
   docker-compose up -d
   ```

3. **访问界面**
   打开浏览器访问 `http://localhost:7862`。

### 方式二：本地运行

**环境要求**：
- Python 3.10+
#### Ubuntu
```shell
apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        ffmpeg \
        imagemagick \
        libsndfile1 \
        libgl1 \
        libglib2.0-0
find /etc -name "policy.xml" -exec sed -i 's/none/read,write/g' {} +        
```
#### windows
- FFmpeg 
    ([下载FFmpeg](https://ffmpeg.org/download.html)),重命名放在根目录下，./ffmpeg/bin/ffmpeg.exe
- ImageMagick 
    ([下载并安装imagemagick](https://imagemagick.org/script/download.php#windows))
    然后确定您的Python安装位置，在其中的`site-packages\moviepy\config_defaults.py`文件中修改`IMAGEMAGICK_BINARY`为imagemagick的exe路径

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **运行应用**
   ```bash
   python videoautoclip/task_queue.py
   ```

## ⚙️ 配置说明

在 Web UI 的“设置”标签页中配置以下关键信息：

![Settings UI](docs/images/setting.png)

- **DASHSCOPE_API_KEY**: 阿里云 DashScope API Key (用于调用 Qwen 模型)。
- **BAIDU_PCS_COOKIES**: 百度网盘 Cookie (BDUSS 也是必须的)。
- **BAIDU_PCS_BDUSS**: 百度网盘 BDUSS。
- **MODEL_NAME**: 使用的 LLM 模型名称 (默认 `qwen-plus`)。
- **MINIO_ENDPOINT**: MinIO 服务端地址 (默认 `http://minio:9000`)。
- **MINIO_ACCESS_KEY**: MinIO 访问密钥。
- **MINIO_SECRET_KEY**: MinIO 密钥。
- **MINIO_BASE_URL**: MinIO 基础 URL (默认 `http://localhost:9000`)。
ASR需要公网可以访问的MinIO服务端，需要配置MINIO_BASE_URL为公网地址。
## 🛠️ 技术栈

- **语言**: Python 3.10
- **Web 框架**: Gradio
- **视频处理**: MoviePy, FFmpeg
- **AI 模型**: FunASR, Qwen-plus (via DashScope)
- **存储**: SQLite, MinIO
- **工具**: BaiduPCS-Py

## 📝 工作流程

1. **任务创建**：用户输入网盘链接。
2. **资源获取**：系统自动下载视频素材。
3. **AI 处理**：提取音频 -> ASR 转字幕 -> LLM 分析剧情。
4. **智能剪辑**：根据 AI 建议剪辑高能片段，添加封面、字幕和 BGM。
5. **成品导出**：生成最终 MP4 视频供下载。

## 📄 许可证

[MIT License](LICENSE)
