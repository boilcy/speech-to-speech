import base64
import mimetypes
import os
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


def get_image_mime_type(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type and mime_type.startswith("image/"):
        return mime_type

    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".jpg" or ext == ".jpeg":
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    return "application/octet-stream"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def img2url(image_path):
    base64_image = encode_image(image_path)
    mime_type = get_image_mime_type(image_path)
    return f"data:{mime_type};base64,{base64_image}"


"""
img_url = img2url(image_path)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": img_url}}
            ],
        }
    ]
)
"""


def test_audio_devices():
    """测试音频设备"""
    try:
        import sounddevice as sd

        logger.info("🔍 检测可用的音频设备...")
        devices = sd.query_devices()

        logger.info("\n📱 可用的音频设备:")
        for i, device in enumerate(devices):
            device_type = []
            if device["max_input_channels"] > 0:
                device_type.append("输入")
            if device["max_output_channels"] > 0:
                device_type.append("输出")

            logger.info(f"  {i}: {device['name']} ({', '.join(device_type)})")

        logger.info(f"\n🎤 默认输入设备: {sd.query_devices(kind='input')['name']}")
        logger.info(f"🔊 默认输出设备: {sd.query_devices(kind='output')['name']}")

        # 测试录音
        logger.info("\n🎙️  测试录音 (3秒)...")
        duration = 3
        sample_rate = 44100

        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
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
