# QArobo - Speech to Speech AI Assistant

**实时、多平台语音到语音(Speech-to-Speech) AI 助手。**

核心组件：`silero-vad` (VAD) + `Whisper STT` (ASR) + `Transformers LLM` (NLU) + `Kokoro TTS` (TTS)。
**Codebase**：[Hugging Face Speech-to-Speech](https://github.com/huggingface/speech-to-speech)

## 快速上手

### 前置要求

- [Pixi](https://pixi.sh/) 包管理器 (Linux/macOS/Windows)
- CUDA 12.0+ (用于 GPU 加速), 尤其是 conda 尚未支持的 jetson 版本 cuda

### 支持平台

- **Windows 64 位** (`win-64`)
- **Linux 64 位** (`linux-64`)
- **Linux ARM64** (`linux-aarch64`) - 特指 Jetson 设备

### 安装与设置

1.  **安装 Pixi**：
    - Linux/macOS: `curl -fsSL https://pixi.sh/install.sh | bash`
    - Windows PowerShell: `iwr -useb https://pixi.sh/install.ps1 | iex`
2.  **克隆项目并安装依赖**：
    ```bash
    git clone https://github.com/boilcy/speech-to-speech
    cd qarobo
    pixi install -e desktop # desktop, orin-nano, thor. see pixi.toml
    ```
3.  **激活环境**：

- 桌面 GPU 环境: `pixi shell -e desktop` (Python 3.12 + CUDA PyTorch)
- Jetson Orin Nano (JP6): `pixi shell -e orin-nano` (Python 3.10 + Jetson PyTorch)
- Jetson Thor (SBSA): `pixi shell -e thor` (Python 3.12 + SBSA PyTorch)

### 模型准备

将所需模型下载到 ./models/ 目录：

- LLM 模型: Qwen2.5-3B-Instruct 或 Qwen3-0.6B
- STT 模型: whisper-large-v3-turbo
- TTS 模型: Kokoro (会自动下载安装)

### 运行 QArobo

- **本地模式 (单机测试)**:
  ```bash
  pixi run local
  ```
- **服务器/客户端模式**:
  1.  **启动服务器**:
      ```bash
      pixi run server
      ```
  2.  **连接客户端 (在另一个终端)**:
      ```bash
      pixi run client --host <server_ip>
      ```

## 开发指南
### 代码格式
请在提交前使用`pixi run format`执行格式化

### 自定义环境
`pixi.toml` 定义了命名环境 (`desktop`, `orin-nano`, `thor`)。要创建或修改环境：

1.  编辑 `pixi.toml` 的 `[environments]` 部分。
2.  为每个环境定义特定的依赖 (PyTorch版本、CUDA工具包) 或自定义 `activation_scripts`。
3.  使用 `pixi shell -e <您的环境名称>` 激活。

### 故障排除与常见问题
*   **Windows (Triton)**: 由于Triton支持限制，编译模式已被禁用。
*   **CUDA**: 确保您的系统CUDA环境变量 (`PATH`, `CUDA_HOME`) 设置正确。
*   **Jetson设备**: 需要手动安装CUDA, CUDNN, CUDSS等库。请参考 [Nvidia Official Doc](https://developer.nvidia.com/cudss-downloads)。
*   **依赖冲突**: 如果 `pixi install` 失败，请检查 `pixi.toml` 中不同环境的包版本是否存在不兼容。请谨慎使用 `pixi solve --frozen` 或 `pixi install --force`。

## Docker 部署
*(维护中 - 准备就绪后将在此处提供更多详细信息。)*
