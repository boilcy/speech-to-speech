from src.base import BaseHandler
from loguru import logger
import time
import json
import numpy as np
import os
from typing import Generator, Any


class TTSDebugHandler(BaseHandler):
    """
    TTS 调试处理器，用于记录和调试文本转语音的输入输出
    """

    def setup(
        self,
        tts_model: str = "unknown",
        log_inputs: bool = True,
        log_outputs: bool = True,
        log_timing: bool = True,
        log_audio_stats: bool = True,
        save_to_file: bool = False,
        debug_file_path: str = "tts_debug.jsonl",
        save_audio_samples: bool = False,
        audio_samples_dir: str = "tts_debug_audio",
        max_audio_samples: int = 10,
    ):
        self.tts_model = tts_model
        self.log_inputs = log_inputs
        self.log_outputs = log_outputs
        self.log_timing = log_timing
        self.log_audio_stats = log_audio_stats
        self.save_to_file = save_to_file
        self.debug_file_path = debug_file_path
        self.save_audio_samples = save_audio_samples
        self.audio_samples_dir = audio_samples_dir
        self.max_audio_samples = max_audio_samples

        self.request_count = 0
        self.audio_samples_saved = 0

        # 创建音频样本目录
        if self.save_audio_samples:
            os.makedirs(self.audio_samples_dir, exist_ok=True)

        logger.info(f"TTS Debug Handler initialized:")
        logger.info(f"  Model: {self.tts_model}")
        logger.info(f"  Log inputs: {self.log_inputs}")
        logger.info(f"  Log outputs: {self.log_outputs}")
        logger.info(f"  Log audio stats: {self.log_audio_stats}")
        logger.info(f"  Save audio samples: {self.save_audio_samples}")

    def process(self, tts_input):
        """
        处理 TTS 输入，记录调试信息并透传音频数据
        """
        self.request_count += 1
        start_time = time.time()

        # 记录输入
        if self.log_inputs:
            logger.debug(f"[TTS Debug #{self.request_count}] Input: {tts_input}")

        # 创建调试记录
        debug_record = {
            "request_id": self.request_count,
            "timestamp": start_time,
            "model": self.tts_model,
            "input": tts_input if self.log_inputs else "[HIDDEN]",
            "input_type": type(tts_input).__name__,
        }

        # 解析输入信息
        if isinstance(tts_input, tuple):
            text, language_code = tts_input
            debug_record["text"] = text if self.log_inputs else "[HIDDEN]"
            debug_record["language_code"] = language_code
            debug_record["text_length"] = len(text) if text else 0
        else:
            debug_record["text"] = str(tts_input) if self.log_inputs else "[HIDDEN]"
            debug_record["text_length"] = len(str(tts_input)) if tts_input else 0

        # 处理音频流
        try:
            audio_chunks = []
            chunk_count = 0
            total_audio_length = 0
            first_chunk_time = None

            # 透传音频块并收集统计信息
            for audio_chunk in self._passthrough_generator(tts_input):
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    debug_record["time_to_first_audio"] = first_chunk_time - start_time
                    if self.log_timing:
                        logger.debug(
                            f"[TTS Debug #{self.request_count}] Time to first audio: {debug_record['time_to_first_audio']:.4f}s"
                        )

                chunk_count += 1

                if isinstance(audio_chunk, np.ndarray):
                    total_audio_length += len(audio_chunk)
                    if (
                        self.save_audio_samples
                        and self.audio_samples_saved < self.max_audio_samples
                    ):
                        audio_chunks.append(audio_chunk.copy())

                yield audio_chunk

            # 记录处理完成时间
            end_time = time.time()
            total_processing_time = end_time - start_time

            # 更新调试记录
            debug_record["chunk_count"] = chunk_count
            debug_record["total_audio_length"] = total_audio_length
            debug_record["total_processing_time"] = total_processing_time
            debug_record["status"] = "success"

            if self.log_timing:
                logger.debug(
                    f"[TTS Debug #{self.request_count}] Total processing time: {total_processing_time:.4f}s"
                )

            if self.log_audio_stats:
                logger.debug(
                    f"[TTS Debug #{self.request_count}] Audio stats: {chunk_count} chunks, {total_audio_length} samples"
                )
                debug_record["audio_stats"] = {
                    "chunk_count": chunk_count,
                    "total_samples": total_audio_length,
                    "estimated_duration_ms": (
                        (total_audio_length / 16000) * 1000
                        if total_audio_length > 0
                        else 0
                    ),
                }

            # 保存音频样本
            if (
                self.save_audio_samples
                and audio_chunks
                and self.audio_samples_saved < self.max_audio_samples
            ):
                self._save_audio_sample(audio_chunks, self.request_count)
                self.audio_samples_saved += 1

            # 保存调试记录到文件
            if self.save_to_file:
                self._save_debug_record(debug_record)

        except Exception as e:
            end_time = time.time()
            total_processing_time = end_time - start_time

            logger.error(f"[TTS Debug #{self.request_count}] Error: {e}")
            debug_record["total_processing_time"] = total_processing_time
            debug_record["status"] = "error"
            debug_record["error"] = str(e)

            if self.save_to_file:
                self._save_debug_record(debug_record)

            # 重新抛出异常
            raise e

    def _passthrough_generator(self, tts_input) -> Generator[Any, None, None]:
        """
        透传生成器，这里应该调用实际的 TTS 处理器
        在实际使用中，这个方法会被管道系统替换
        """
        # 这是一个占位符实现，实际使用时会被管道系统处理
        logger.warning("TTS Debug Handler: _passthrough_generator is a placeholder")
        yield np.array([0] * 512, dtype=np.int16)  # 返回静音作为占位符

    def _save_audio_sample(self, audio_chunks, request_id):
        """保存音频样本到文件"""
        try:
            # 合并所有音频块
            if audio_chunks:
                combined_audio = np.concatenate(audio_chunks)

                # 保存为 WAV 文件
                import soundfile as sf

                filename = f"tts_debug_{request_id:04d}.wav"
                filepath = os.path.join(self.audio_samples_dir, filename)
                sf.write(filepath, combined_audio, 16000)

                logger.debug(
                    f"[TTS Debug #{request_id}] Saved audio sample: {filepath}"
                )
        except Exception as e:
            logger.warning(f"Failed to save audio sample for request {request_id}: {e}")

    def _save_debug_record(self, record):
        """保存调试记录到文件"""
        try:
            with open(self.debug_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to save debug record: {e}")

    def get_stats(self):
        """获取调试统计信息"""
        return {
            "total_requests": self.request_count,
            "model": self.tts_model,
            "average_time": self.average_time,
            "audio_samples_saved": self.audio_samples_saved,
        }

    def cleanup(self):
        """清理资源"""
        logger.info(
            f"TTS Debug Handler cleanup: processed {self.request_count} requests"
        )
        if self.save_audio_samples:
            logger.info(
                f"Saved {self.audio_samples_saved} audio samples to {self.audio_samples_dir}"
            )
