from src.base import BaseHandler
from loguru import logger
import time
import json


class LLMDebugHandler(BaseHandler):
    """
    LLM 调试处理器，用于记录和调试语言模型的输入输出
    """

    def setup(
        self,
        llm_model: str = "unknown",
        llm_url: str = "unknown",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
        log_inputs: bool = True,
        log_outputs: bool = True,
        log_timing: bool = True,
        save_to_file: bool = False,
        debug_file_path: str = "llm_debug.jsonl",
    ):
        self.llm_model = llm_model
        self.llm_url = llm_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stream = stream
        self.log_inputs = log_inputs
        self.log_outputs = log_outputs
        self.log_timing = log_timing
        self.save_to_file = save_to_file
        self.debug_file_path = debug_file_path

        self.request_count = 0

        logger.info(f"LLM Debug Handler initialized:")
        logger.info(f"  Model: {self.llm_model}")
        logger.info(f"  URL: {self.llm_url}")
        logger.info(f"  Max tokens: {self.max_tokens}")
        logger.info(f"  Temperature: {self.temperature}")
        logger.info(f"  Stream: {self.stream}")

    def process(self, llm_input):
        """
        处理 LLM 输入，记录调试信息并透传数据
        """
        self.request_count += 1
        start_time = time.time()

        # 记录输入
        if self.log_inputs:
            logger.debug(f"[LLM Debug #{self.request_count}] Input: {llm_input}")

        # 创建调试记录
        debug_record = {
            "request_id": self.request_count,
            "timestamp": start_time,
            "model": self.llm_model,
            "input": llm_input if self.log_inputs else "[HIDDEN]",
            "input_type": type(llm_input).__name__,
            "input_length": len(str(llm_input)) if llm_input else 0,
        }

        # 透传输入到下一个处理器
        try:
            # 如果输入是元组（包含语言代码），保持原样
            if isinstance(llm_input, tuple):
                text, language_code = llm_input
                debug_record["language_code"] = language_code
                debug_record["text_length"] = len(text) if text else 0
                output = llm_input
            else:
                output = llm_input

            # 记录处理时间
            end_time = time.time()
            processing_time = end_time - start_time

            if self.log_timing:
                logger.debug(
                    f"[LLM Debug #{self.request_count}] Processing time: {processing_time:.4f}s"
                )

            debug_record["processing_time"] = processing_time
            debug_record["status"] = "success"

            # 记录输出
            if self.log_outputs:
                logger.debug(f"[LLM Debug #{self.request_count}] Output: {output}")
                debug_record["output"] = output if self.log_outputs else "[HIDDEN]"

            # 保存到文件
            if self.save_to_file:
                self._save_debug_record(debug_record)

            yield output

        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time

            logger.error(f"[LLM Debug #{self.request_count}] Error: {e}")
            debug_record["processing_time"] = processing_time
            debug_record["status"] = "error"
            debug_record["error"] = str(e)

            if self.save_to_file:
                self._save_debug_record(debug_record)

            # 重新抛出异常
            raise e

    def _save_debug_record(self, record):
        """保存调试记录到文件"""
        try:
            with open(self.debug_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to save debug record: {e}")

    def get_stats(self):
        """获取调试统计信息"""
        return {
            "total_requests": self.request_count,
            "model": self.llm_model,
            "url": self.llm_url,
            "average_time": self.average_time,
        }
