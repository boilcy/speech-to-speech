Jetson 默认不开始蓝牙音频，原因和开启方法详见 https://docs.nvidia.com/jetson/archives/r38.2/DeveloperGuide/SD/Communications/EnablingBluetoothAudio.html

Linux音频技术说明
1. ALSA (Advanced Linux Sound Architecture)
Linux 内核中的一个驱动程序层，它提供对声卡硬件的直接访问。它是最底层的音频接口。
2. PulseAudio
一个声音服务器，运行在用户空间，位于应用程序和 ALSA 之间。它旨在解决 ALSA 的一些局限性，提供更高级和用户友好的音频管理。
pactl 命令就是 PulseAudio 的命令行客户端。
3. PipeWire
新兴的多媒体处理服务器，旨在成为 Linux 上所有音频和视频流的核心基础设施，统一和取代 PulseAudio、JACK (专业音频) 和处理视频流 (如用于 Wayland 的屏幕共享)。
- 兼容性： 提供 PulseAudio 和 JACK 的兼容层，这意味着为 PulseAudio 或 JACK 编写的应用程序可以在 PipeWire 上无缝运行，而无需修改。
- pactl list sinks short 会显示设备由 PipeWire 管理

```bash
$ pactl list sinks short
49      alsa_output.platform-sound.analog-stereo        PipeWire        s24-32le 2ch 48000Hz    SUSPENDED
58      bluez_output.41_42_A4_EB_FE_9F.1        PipeWire        s16le 2ch 48000Hz       SUSPENDED
```

4. PortAudio
开源、跨平台的音频 I/O 库。它提供了一个简单的、统一的 API，用于在各种操作系统（Windows, macOS, Linux, etc.）和音频后端（ASIO, CoreAudio, WASAPI, ALSA, PulseAudio, JACK, etc.）上进行音频输入和输出。

5. sounddevice
Python 的 sounddevice 库是 PortAudio 的一个包装器。
选择 HostAPI： PortAudio 允许你选择它应该使用哪个底层 HostAPI（如 ALSA 或 PulseAudio）。如果 PortAudio 没有被编译为支持或没有检测到 PulseAudio (或 PipeWire 的 PulseAudio 兼容层) HostAPI，那么即使 PulseAudio/PipeWire 正在运行，它也只会看到 ALSA 设备。

# 常用流程
1. 进入bluetoothctl连接蓝牙
[bluetooth]# power on
开启代理
[bluetooth]# agent on
[bluetooth]# default-agent
扫描
[bluetooth]# scan on
例如 4C:87:5D:XX:YY:ZZ Name: MyHeadphones

[bluetooth]# scan off
[bluetooth]# pair 4C:87:5D:XX:YY:ZZ
[bluetooth]# trust 4C:87:5D:XX:YY:ZZ
[bluetooth]# connect 4C:87:5D:XX:YY:ZZ
[bluetooth]# quit

2. 查看设备
pactl list sinks short
pactl list sources short

wpctl更直观
```bash
$wpctl status

PipeWire 0.3.XX

Audio
├─ Devices:
│          ...
│       1. NVIDIA Corporation ...                      [alsa_card]
│       2. MyBluetoothDevice                 [bluetooth]
├─ Sinks:
│       1. MyBluetoothDevice (A2DP)           [active]
│       2. NVIDIA Corporation ...                      [alsa_output]
├─ Sources:
│       1. MyBluetoothDevice (HFP)            [active]
│       2. NVIDIA Corporation ...                      [alsa_input]
```

3. 默认播放/录音设备
```bash
pactl set-default-sink bluez_sink.AA_BB_CC_DD_EE_FF.a2dp_sink

pactl set-default-source bluez_source.AA_BB_CC_DD_EE_FF.a2dp_source
```

4. 监控
pw-top 会显示所有通过 PipeWire 传输音频的应用程序。

# Conclusion

当通过 `conda install librosa` 安装 `librosa` 时，conda 会在其环境中安装 `pysoundfile` 以及它所依赖的 C 库 `libsndfile` 和 `alsa-lib`。由于 conda 环境中的库在 `$PATH` 和 `$LD_LIBRARY_PATH` 环境变量中通常会优先于系统库被查找，`sounddevice`（以及任何依赖ALSA的Python包，如`pysoundfile`）在运行时会优先加载并使用 conda 环境中提供的 `alsa-lib`。

这种情况下，conda 环境中沙盒化的 `alsa-lib` 版本可能与系统环境（或其他用户配置）不完全兼容，或者其内部查找配置文件的逻辑（例如 `~/.asoundrc` 或 `/etc/alsa/conf.d/`）与系统预期的行为有所偏差。这可能导致 `sounddevice` 无法正确检测和应用由 `~/.asoundrc` 定义的自定义ALSA设备配置。

而当通过 `pip install librosa` 安装 `librosa` 时，`pip` 通常只安装 Python 包及其纯 Python 依赖。对于像 `pysoundfile` 这样依赖 C 库的包，它会尝试链接到**系统已安装的 `libsndfile` 和 `alsa-lib`**。由于此时 `sounddevice` 直接使用系统级别的 `alsa-lib` 库，而系统级别的 `alsa-lib` 能够正确且完整地识别和处理 `~/.asoundrc`，因此自定义的ALSA设备配置就能被正确应用。
