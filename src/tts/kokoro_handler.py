import threading
import queue
from threading import Thread
from time import perf_counter
import sys
from src.base import BaseHandler
import numpy as np
import torch
from kokoro import KModel, KPipeline
from loguru import logger


# Whisper 语言代码到 Kokoro 语音的映射
WHISPER_LANGUAGE_TO_KOKORO_VOICE = {
    "zh": "zf_001",  # 中文女声
    "en": "af_001",  # 英文女声 (假设有的话)
}

# Kokoro 支持的语言代码映射
WHISPER_LANGUAGE_TO_KOKORO_LANG = {
    "zh": "z",  # 中文
    "en": "a",  # 英文
}


class KokoroTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen: threading.Event,
        model_name="hexgrad/Kokoro-82M-v1.1-zh",
        device="cuda",
        torch_dtype="float16",
        compile_mode=None,
        sample_rate=24000,
        default_voice="zf_001",
        use_default_voice=True,
        default_language="zh",
        blocksize=512,
        use_speed_adjustment=True,
        n_zeros=5000,
        gen_kwargs={},
    ):
        if compile_mode and sys.platform != "linux":
            logger.warning("Torch compile is only available on Linux. Disabling compile mode.")
            compile_mode = None
        self.should_listen = should_listen
        self.device = device if torch.cuda.is_available() else "cpu"
        self.torch_dtype = getattr(torch, torch_dtype)
        self.compile_mode = compile_mode
        self.sample_rate = sample_rate
        self.default_voice = default_voice
        self.default_language = default_language
        self.blocksize = blocksize
        self.use_speed_adjustment = use_speed_adjustment
        self.n_zeros = n_zeros
        self.gen_kwargs = gen_kwargs
        self.use_default_voice = use_default_voice

        logger.info(f"Loading Kokoro model {model_name} on {self.device}")

        self.model = KModel(repo_id=model_name).to(self.device).eval()
        if self.compile_mode:
            self.model.generation_config.cache_implementation = "static"
            self.model.forward = torch.compile(
                self.model.forward, mode=self.compile_mode, fullgraph=True
            )

        self.en_pipeline = KPipeline(lang_code="a", repo_id=model_name, model=False)

        self.zh_pipeline = KPipeline(
            lang_code="z",
            repo_id=model_name,
            model=self.model,
            en_callable=self._en_callable,
        )

        # 当前语音和语言设置
        self.current_voice = default_voice
        self.current_language = default_language

        self.warmup()

    def _en_callable(self, text):
        """英文文本的音素处理回调"""
        if text == "Kokoro":
            return "kˈOkəɹO"
        elif text == "Sol":
            return "sˈOl"
        return next(self.en_pipeline(text)).phonemes

    def _speed_callable(self, len_ps):
        """根据音素长度调整语速的回调函数"""
        if not self.use_speed_adjustment:
            return 1.0

        speed = 0.8
        if len_ps <= 83:
            speed = 1
        elif len_ps < 183:
            speed = 1 - (len_ps - 83) / 500
        return speed * 1.1

    def warmup(self):
        """预热模型"""
        logger.info(f"Warming up {self.__class__.__name__}")

        if self.device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
        else:
            start_time = perf_counter()

        # 预热生成
        dummy_text = "这是一个测试句子。"
        try:
            generator = self.zh_pipeline(
                dummy_text, voice=self.current_voice, speed=self._speed_callable
            )
            result = next(generator)
            _ = result.audio
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

        if self.device == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            logger.info(
                f"{self.__class__.__name__}: warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s"
            )
        else:
            end_time = perf_counter()
            logger.info(
                f"{self.__class__.__name__}: warmed up! time: {end_time - start_time:.3f} s"
            )

    def process(self, llm_sentence):
        """处理文本并生成音频"""
        language_code = self.default_language

        if isinstance(llm_sentence, tuple):
            llm_sentence, language_code = llm_sentence
            if not self.use_default_voice:
                self.current_voice = WHISPER_LANGUAGE_TO_KOKORO_VOICE.get(
                    language_code, self.default_voice
                )

            self.current_language = language_code

        logger.info(f"ASSISTANT: {llm_sentence}")

        # 创建音频队列用于线程间通信
        audio_queue = queue.Queue()
        generation_complete = threading.Event()

        def generate_audio():
            """在独立线程中生成音频"""
            try:
                kokoro_lang = WHISPER_LANGUAGE_TO_KOKORO_LANG.get(language_code, "z")

                if kokoro_lang == "z":
                    pipeline = self.zh_pipeline
                else:
                    pipeline = self.zh_pipeline
                    logger.warning(
                        f"Language {language_code} not fully supported, using Chinese pipeline"
                    )

                generator = pipeline(
                    llm_sentence, voice=self.current_voice, speed=self._speed_callable
                )

                result = next(generator)
                audio = result.audio
                logger.debug(f"Generated audio: {type(audio)} {audio.shape}")

                if audio is None or audio.numel() == 0:
                    logger.warning("Generated empty audio")
                    audio_queue.put(None)
                    return

                audio_numpy = audio.cpu().numpy()

                if self.sample_rate != 16000:
                    import librosa

                    audio_resampled = librosa.resample(
                        audio_numpy, orig_sr=self.sample_rate, target_sr=16000
                    )

                # 转换为 int16 格式
                audio_chunk = (audio_resampled * 32768).astype(np.int16)
                audio_queue.put(audio_chunk)

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

        if audio is not None:
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
        logger.info("Cleaning up Kokoro TTS handler")
        pass
