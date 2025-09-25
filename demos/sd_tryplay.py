import sounddevice as sd
import numpy as np

# A. 查询设备，找到 'sysdefault' 的准确名称或索引
print("Available devices:")
print(sd.query_devices())


# 简单测试播放
samplerate = 44100  # samples per second
duration = 2.0  # seconds
frequency = 440  # Hz (A4 note)
t = np.linspace(0.0, duration, int(samplerate * duration), False)
data = 0.5 * np.sin(2 * np.pi * frequency * t)

print("Playing a test tone on the new default device...")
sd.play(data, samplerate)
sd.wait()
print("Playback finished.")
