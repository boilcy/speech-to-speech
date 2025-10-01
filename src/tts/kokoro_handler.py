import os
import re
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
import librosa

KOKORO_ZH_REPO_ID = "hexgrad/Kokoro-82M-v1.1-zh"


# Kokoro 语音模型文件名列表（由ls结果转换而来）
KOKORO_ZH_VOICES = [
    "af_maple",
    "af_sol",
    "bf_vale",
    "zf_001",
    "zf_002",
    "zf_003",
    "zf_004",
    "zf_005",
    "zf_006",
    "zf_007",
    "zf_008",
    "zf_017",
    "zf_018",
    "zf_019",
    "zf_021",
    "zf_022",
    "zf_023",
    "zf_024",
    "zf_026",
    "zf_027",
    "zf_028",
    "zf_032",
    "zf_036",
    "zf_038",
    "zf_039",
    "zf_040",
    "zf_042",
    "zf_043",
    "zf_044",
    "zf_046",
    "zf_047",
    "zf_048",
    "zf_049",
    "zf_051",
    "zf_059",
    "zf_060",
    "zf_067",
    "zf_070",
    "zf_071",
    "zf_072",
    "zf_073",
    "zf_074",
    "zf_075",
    "zf_076",
    "zf_077",
    "zf_078",
    "zf_079",
    "zf_083",
    "zf_084",
    "zf_085",
    "zf_086",
    "zf_087",
    "zf_088",
    "zf_090",
    "zf_092",
    "zf_093",
    "zf_094",
    "zf_099",
    "zm_009",
    "zm_010",
    "zm_011",
    "zm_012",
    "zm_013",
    "zm_014",
    "zm_015",
    "zm_016",
    "zm_020",
    "zm_025",
    "zm_029",
    "zm_030",
    "zm_031",
    "zm_033",
    "zm_034",
    "zm_035",
    "zm_037",
    "zm_041",
    "zm_045",
    "zm_050",
    "zm_052",
    "zm_053",
    "zm_054",
    "zm_055",
    "zm_056",
    "zm_057",
    "zm_058",
    "zm_061",
    "zm_062",
    "zm_063",
    "zm_064",
    "zm_065",
    "zm_066",
    "zm_068",
    "zm_069",
    "zm_080",
    "zm_081",
    "zm_082",
    "zm_089",
    "zm_091",
    "zm_095",
    "zm_096",
    "zm_097",
    "zm_098",
    "zm_100",
]

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
        model_name=KOKORO_ZH_REPO_ID,
        device="cuda",
        torch_dtype="float16",
        compile_mode=None,
        sample_rate=24000,
        target_sample_rate=16000,
        default_voice="zf_001",
        use_default_voice=True,
        default_language="zh",
        blocksize=512,
        use_speed_adjustment=True,
        n_zeros=5000,
        gen_kwargs={},
    ):
        if compile_mode and sys.platform != "linux":
            logger.warning(
                "Torch compile is only available on Linux. Disabling compile mode."
            )
            compile_mode = None
        self.should_listen = should_listen
        self.device = device if torch.cuda.is_available() else "cpu"
        self.torch_dtype = getattr(torch, torch_dtype)
        self.compile_mode = compile_mode
        self.sample_rate = sample_rate
        self.target_sample_rate = target_sample_rate
        self.default_voice = default_voice
        self.default_language = default_language
        self.blocksize = blocksize
        self.use_speed_adjustment = use_speed_adjustment
        self.n_zeros = n_zeros
        self.gen_kwargs = gen_kwargs
        self.use_default_voice = use_default_voice

        logger.info(f"Loading Kokoro model {model_name} on {self.device}")
        
        self.voice = None
        # check if model_name is a valid path
        if os.path.exists(model_name) or os.path.exists(os.path.expanduser(model_name)):
            model_dir = os.path.expanduser(model_name)
            model_weights_path = os.path.join(model_dir, "kokoro-v1_1-zh.pth")
            model_config_path = os.path.join(model_dir, "config.json")
            self.model = KModel(model = model_weights_path, config = model_config_path, repo_id=KOKORO_ZH_REPO_ID).to(self.device).eval()
            self.voice = torch.load(f'{model_dir}/voices/{self.default_voice}.pt', weights_only=True)
        else:
            self.model = KModel(repo_id=KOKORO_ZH_REPO_ID).to(self.device).eval()

        if self.compile_mode:
            self.model.generation_config.cache_implementation = "static"
            self.model.forward = torch.compile(
                self.model.forward, mode=self.compile_mode, fullgraph=True
            )

        self.en_pipeline = KPipeline(
            lang_code="a", repo_id=KOKORO_ZH_REPO_ID, model=False
        )

        self.zh_pipeline = KPipeline(
            lang_code="z",
            repo_id=KOKORO_ZH_REPO_ID,
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
                dummy_text, voice=self.voice, speed=self._speed_callable
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

    def _cleanup_sentence(self, llm_sentence):
        """
        清理句子，以防止 TTS 合成意外停止。
        主要针对中文场景，去除换行符、多余空格、特殊标点等。
        """
        # 1. 将所有换行符替换为空格
        cleaned_sentence = llm_sentence.replace('\n', ' ').replace('\r', ' ')
        # 2. 将多个连续的空格替换为单个空格
        cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence)
        # 3. 去除句子两端的空格
        cleaned_sentence = cleaned_sentence.strip()
        # 4. 移除一些可能干扰 TTS 的特殊控制字符或不可见字符
        # 这是一个常见的 Unicode 范围，包含了一些可能不需要的控制字符
        cleaned_sentence = re.sub(r'[\u0000-\u001F\u007F-\u009F]', '', cleaned_sentence)
        return cleaned_sentence

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
        
        llm_sentence = self._cleanup_sentence(llm_sentence)

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
                    llm_sentence, voice=self.voice, speed=self._speed_callable
                )

                result = next(generator)
                audio = result.audio
                logger.debug(
                    f"Audio generated, shape: {audio.shape if audio is not None else 'None'}"
                )

                if audio is None or audio.numel() == 0:
                    logger.warning("Generated empty audio")
                    audio_queue.put(None)
                    return

                audio_numpy = audio.cpu().numpy()

                if self.sample_rate != self.target_sample_rate:
                    audio_resampled = librosa.resample(
                        audio_numpy,
                        orig_sr=self.sample_rate,
                        target_sr=self.target_sample_rate,
                    )
                else:
                    audio_resampled = audio_numpy

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
            logger.debug(f"Outputting audio in chunks, total length: {len(audio)}")
            for i in range(0, len(audio), self.blocksize):
                global pipeline_start
                if i == 0 and "pipeline_start" in globals():
                    logger.info(
                        f"Time to first audio: {perf_counter() - pipeline_start:.3f}s"
                    )
                chunk = audio[i : i + self.blocksize]
                if len(chunk) < self.blocksize:
                    chunk = np.pad(chunk, (0, self.blocksize - len(chunk)))
                yield chunk
        else:
            logger.warning("No audio generated, skipping output")

        # 等待线程完成
        thread.join()
        self.should_listen.set()

    def cleanup(self):
        """清理资源"""
        logger.info("Cleaning up Kokoro TTS handler")
        pass
