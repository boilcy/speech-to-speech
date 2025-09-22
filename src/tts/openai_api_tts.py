import threading
import queue
from threading import Thread
from time import perf_counter
from src.base import BaseHandler
import numpy as np
import requests
import io
from loguru import logger

WHISPER_LANGUAGE_TO_OPENAI_VOICE = {
    "en": "alloy",
    "zh": "alloy",
    "es": "nova",
    "fr": "shimmer",
    "de": "echo",
    "pt": "fable",
    "it": "onyx",
    "ja": "alloy",
    "ko": "alloy",
}


class OpenAITTSHandler(BaseHandler):
    """
    OpenAI API 兼容的 TTS 处理器
    """

    def setup(
        self,
        should_listen: threading.Event,
        api_key: str = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "tts-1",
        voice: str = "alloy",
        response_format: str = "wav",
        speed: float = 1.0,
        timeout: float = 30.0,
        blocksize: int = 512,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.should_listen = should_listen
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.default_voice = voice
        self.response_format = response_format
        self.speed = speed
        self.timeout = timeout
        self.blocksize = blocksize
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # 构建 API 端点
        self.tts_endpoint = f"{self.base_url}/audio/speech"

        # 当前语音设置
        self.current_voice = voice

        # 设置请求头
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
        }

        logger.info(f"OpenAI TTS Handler initialized:")
        logger.info(f"  Base URL: {self.base_url}")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  Default voice: {self.default_voice}")
        logger.info(f"  Response format: {self.response_format}")

        self.warmup()

    def warmup(self):
        """预热 API 连接"""
        logger.info(f"Warming up {self.__class__.__name__}")

        start_time = perf_counter()

        # 测试 API 连接
        try:
            test_text = "Hello"
            self._generate_speech(test_text)
            logger.info("API connection test successful")
        except Exception as e:
            logger.warning(f"API connection test failed: {e}")

        end_time = perf_counter()
        logger.info(
            f"{self.__class__.__name__}: warmed up! time: {end_time - start_time:.3f} s"
        )

    def _generate_speech(self, text: str, voice: str = None) -> bytes:
        """调用 OpenAI API 生成语音"""
        if not voice:
            voice = self.current_voice

        payload = {
            "model": self.model,
            "input": text,
            "voice": voice,
            "response_format": self.response_format,
            "speed": self.speed,
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.tts_endpoint,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    return response.content
                else:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    if attempt == self.max_retries - 1:
                        raise Exception(error_msg)
                    else:
                        logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")

            except requests.exceptions.Timeout:
                error_msg = f"Request timeout after {self.timeout}s"
                if attempt == self.max_retries - 1:
                    raise Exception(error_msg)
                else:
                    logger.warning(f"Attempt {attempt + 1} timed out")

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                else:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")

            # 重试延迟
            if attempt < self.max_retries - 1:
                import time

                time.sleep(self.retry_delay * (attempt + 1))

        raise Exception(f"All {self.max_retries} attempts failed")

    def _audio_to_chunks(self, audio_data: bytes):
        """将音频数据转换为音频块"""
        try:
            # 使用 soundfile 读取音频数据
            import soundfile as sf

            audio_io = io.BytesIO(audio_data)
            audio_array, sample_rate = sf.read(audio_io)

            # 确保是单声道
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)

            # 重采样到 16kHz（如果需要）
            if sample_rate != 16000:
                import librosa

                audio_array = librosa.resample(
                    audio_array, orig_sr=sample_rate, target_sr=16000
                )

            # 转换为 int16 格式
            audio_array = (audio_array * 32768).astype(np.int16)

            return audio_array

        except Exception as e:
            logger.error(f"Error processing audio data: {e}")
            return np.array([], dtype=np.int16)

    def process(self, llm_sentence):
        """处理文本并生成音频"""
        language_code = "en"

        if isinstance(llm_sentence, tuple):
            llm_sentence, language_code = llm_sentence
            # 根据语言代码选择语音
            self.current_voice = WHISPER_LANGUAGE_TO_OPENAI_VOICE.get(
                language_code, self.default_voice
            )

        logger.info(f"ASSISTANT: {llm_sentence}")

        # 创建音频队列用于线程间通信
        audio_queue = queue.Queue()
        generation_complete = threading.Event()

        def generate_audio():
            """在独立线程中生成音频"""
            try:
                # 调用 OpenAI API 生成语音
                audio_data = self._generate_speech(llm_sentence, self.current_voice)

                if not audio_data:
                    logger.warning("Generated empty audio")
                    audio_queue.put(None)
                    return

                # 转换音频数据为数组
                audio_array = self._audio_to_chunks(audio_data)
                audio_queue.put(audio_array)

            except Exception as e:
                logger.error(f"Error generating audio: {e}", exc_info=True)
                audio_queue.put(None)
            finally:
                generation_complete.set()

        # 在新线程中启动音频生成
        thread = Thread(target=generate_audio)
        thread.start()

        # 等待音频生成完成并获取结果
        generation_complete.wait()
        audio = audio_queue.get()

        if audio is not None and len(audio) > 0:
            # 分块输出音频
            for i in range(0, len(audio), self.blocksize):
                global pipeline_start
                if i == 0 and "pipeline_start" in globals():
                    logger.info(
                        f"Time to first audio: {perf_counter() - pipeline_start:.3f}"
                    )
                chunk = audio[i : i + self.blocksize]
                if len(chunk) < self.blocksize:
                    chunk = np.pad(chunk, (0, self.blocksize - len(chunk)))
                yield chunk

        # 等待线程完成
        thread.join()
        self.should_listen.set()

    def cleanup(self):
        """清理资源"""
        logger.info("Cleaning up OpenAI TTS handler")
