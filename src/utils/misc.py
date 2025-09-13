from loguru import logger
import numpy as np


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound

def test_audio_devices():
    """测试音频设备"""
    try:
        import sounddevice as sd
        
        logger.info("🔍 检测可用的音频设备...")
        devices = sd.query_devices()
        
        logger.info("\n📱 可用的音频设备:")
        for i, device in enumerate(devices):
            device_type = []
            if device['max_input_channels'] > 0:
                device_type.append("输入")
            if device['max_output_channels'] > 0:
                device_type.append("输出")
            
            logger.info(f"  {i}: {device['name']} ({', '.join(device_type)})")
        
        logger.info(f"\n🎤 默认输入设备: {sd.query_devices(kind='input')['name']}")
        logger.info(f"🔊 默认输出设备: {sd.query_devices(kind='output')['name']}")
        
        # 测试录音
        logger.info("\n🎙️  测试录音 (3秒)...")
        duration = 3
        sample_rate = 16000
        
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        
        logger.info("✅ 录音测试完成")
        
        # 测试播放
        logger.info("🔊 测试播放录音...")
        sd.play(recording, sample_rate)
        sd.wait()
        
        logger.info("✅ 播放测试完成")
        logger.info("🎉 音频设备测试成功!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 音频设备测试失败: {e}")
        return False