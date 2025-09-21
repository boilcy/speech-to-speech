# QArobo - Speech to Speech AI Assistant

基于语音到语音(Speech-to-Speech)的实时AI助手，支持多平台部署。

**Codebase**: https://github.com/huggingface/speech-to-speech  
**Key Components**: silero-vad + Whisper STT + Transformers LLM + Kokoro TTS

## 环境设置

### 前置要求

- [Pixi](https://pixi.sh/) 包管理器
- CUDA 12.0+ (GPU加速)
- Python 3.10+ (自动管理)

### 支持平台

- **Windows 64位** (`win-64`)
- **Linux 64位** (`linux-64`) 
- **Linux ARM64** (`linux-aarch64`) - Jetson设备

### 快速安装

1. **安装Pixi**
   ```bash
   # Linux/macOS
   curl -fsSL https://pixi.sh/install.sh | bash
   
   # Windows PowerShell
   iwr -useb https://pixi.sh/install.ps1 | iex
   ```

2. **克隆项目并安装依赖**
   ```bash
   git clone <repository-url>
   cd qarobo
   pixi install
   ```

### 环境配置

项目提供三种预配置环境：

#### 桌面GPU环境 (推荐)
```bash
# 使用Python 3.12 + CUDA PyTorch
pixi shell -e desktop
```

#### Jetson Orin Nano (JP6)
```bash
# 使用Python 3.10 + Jetson PyTorch
pixi shell -e orin-nano
```

#### Jetson Thor (SBSA)
```bash
# 使用Python 3.12 + SBSA PyTorch  
pixi shell -e thor
```

### 模型准备

下载所需模型到 `./models/` 目录：

- **LLM模型**: Qwen2.5-3B-Instruct 或 Qwen3-0.6B
- **STT模型**: whisper-large-v3-turbo
- **TTS模型**: Kokoro (自动安装)

### 快速启动

#### 本地模式 (单机测试)
```bash
pixi run local
```

#### 服务器模式
```bash
# 启动服务器
pixi run server

# 客户端连接 (另一个终端)
pixi run client --host <server_ip>
```

### 故障排除

#### Windows平台
- 由于Triton支持限制，编译模式已禁用
- 确保CUDA环境变量正确设置

#### Jetson设备
对于Jetson 设备，可能需要额外安装CUDSS, see [Nvidia Official Doc](https://developer.nvidia.com/cudss-downloads)：

## Docker 部署
维护中