from datetime import datetime
from queue import Queue
from threading import Event
from typing import Optional
from loguru import logger
import os
import sys
from copy import copy

from transformers import HfArgumentParser

# import argument classes
from src.arguments.open_api_tts_arguments import OpenApiTTSHandlerArguments
from src.arguments.socket_receiver_arguments import SocketReceiverArguments
from src.arguments.socket_sender_arguments import SocketSenderArguments
from src.arguments.kokoro_tts_arguments import KokoroTTSHandlerArguments
from src.arguments.language_model_arguments import LanguageModelHandlerArguments
from src.arguments.module_arguments import ModuleArguments
from src.arguments.open_api_language_model_arguments import (
    OpenApiLanguageModelHandlerArguments,
)
from src.arguments.vad_arguments import VADHandlerArguments
from src.arguments.whisper_stt_arguments import WhisperSTTHandlerArguments
from pathlib import Path
import nltk

from src.utils.thread_manager import ThreadManager
from src.vad.vad_handler import VADHandler

# Ensure that the necessary NLTK resources are available
try:
    nltk.data.find("tokenizers/punkt_tab")
except (LookupError, OSError):
    nltk.download("punkt_tab")
try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except (LookupError, OSError):
    nltk.download("averaged_perceptron_tagger_eng")


# caching allows ~50% compilation time reduction
# see https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.o2asbxsrp1ma
CURRENT_DIR = Path(__file__).resolve().parent
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(CURRENT_DIR, "tmp")


def rename_args(args, prefix, add_gen_kwargs=True):
    """
    Rename arguments by removing the prefix and prepares the gen_kwargs.
    """
    gen_kwargs = {}
    for key in copy(args.__dict__):
        if key.startswith(prefix):
            value = args.__dict__.pop(key)
            new_key = key[len(prefix) + 1 :]  # Remove prefix and underscore
            if new_key.startswith("gen_"):
                gen_kwargs[new_key[4:]] = value  # Remove 'gen_' and add to dict
            else:
                args.__dict__[new_key] = value

    if add_gen_kwargs:
        args.__dict__["gen_kwargs"] = gen_kwargs


def parse_args():
    parser = HfArgumentParser(
        (
            ModuleArguments,
            SocketReceiverArguments,
            SocketSenderArguments,
            WhisperSTTHandlerArguments,
            LanguageModelHandlerArguments,
            OpenApiLanguageModelHandlerArguments,
            KokoroTTSHandlerArguments,
            OpenApiTTSHandlerArguments,
            VADHandlerArguments,
        )
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Parse configurations from a JSON file if specified
        return parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # Parse arguments from command line if no JSON file is provided
        return parser.parse_args_into_dataclasses()

def setup_global_logging(log_level: str):
    global logger
    logger.remove()
    log_level = log_level.upper() if log_level else "INFO"
    valid_levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in valid_levels:
        print(f"警告: 无效的日志级别 '{log_level}'。使用默认级别 'INFO'。", file=sys.stderr)
        log_level = "INFO"

    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )
    log_file = os.path.join(CURRENT_DIR, "logs", f"s2s_pipeline_{log_level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger.add(
        log_file,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        colorize=True,
        rotation="10 MB",
        retention="10 days",
        enqueue=False,
    )
    if log_level == "DEBUG":
        import torch
        torch._logging.set_logs(graph_breaks=True, recompiles=True, cudagraphs=True)

    logger.info(f"全局日志级别已设置为: {log_level}")

def overwrite_device_argument(common_device: Optional[str], *handler_kwargs):
    if common_device:
        for kwargs in handler_kwargs:
            if hasattr(kwargs, "lm_device"):
                kwargs.lm_device = common_device
            if hasattr(kwargs, "tts_device"):
                kwargs.tts_device = common_device
            if hasattr(kwargs, "stt_device"):
                kwargs.stt_device = common_device
            if hasattr(kwargs, "paraformer_stt_device"):
                kwargs.paraformer_stt_device = common_device
            if hasattr(kwargs, "facebook_mms_device"):
                kwargs.facebook_mms_device = common_device


def prepare_module_args(module_kwargs, *handler_kwargs):
    overwrite_device_argument(module_kwargs.device, *handler_kwargs)


def prepare_all_args(
    module_kwargs,
    whisper_stt_handler_kwargs,
    language_model_handler_kwargs,
    open_api_language_model_handler_kwargs,
    kokoro_tts_handler_kwargs,
    open_api_tts_handler_kwargs,
    vad_handler_kwargs,
):
    prepare_module_args(
        module_kwargs,
        whisper_stt_handler_kwargs,
        language_model_handler_kwargs,
        open_api_language_model_handler_kwargs,
        kokoro_tts_handler_kwargs,
        open_api_tts_handler_kwargs,
        vad_handler_kwargs,
    )

    rename_args(whisper_stt_handler_kwargs, "stt")
    rename_args(language_model_handler_kwargs, "lm")
    rename_args(open_api_language_model_handler_kwargs, "open_api")
    rename_args(kokoro_tts_handler_kwargs, "tts")
    rename_args(vad_handler_kwargs, "vad", add_gen_kwargs=False)


def initialize_queues_and_events():
    return {
        "stop_event": Event(),
        "should_listen": Event(),
        "recv_audio_chunks_queue": Queue(),
        "send_audio_chunks_queue": Queue(),
        "spoken_prompt_queue": Queue(),
        "text_prompt_queue": Queue(),
        "lm_response_queue": Queue(),
    }


def build_pipeline(
    module_kwargs,
    socket_receiver_kwargs,
    socket_sender_kwargs,
    vad_handler_kwargs,
    whisper_stt_handler_kwargs,
    language_model_handler_kwargs,
    open_api_language_model_handler_kwargs,
    kokoro_tts_handler_kwargs,
    open_api_tts_handler_kwargs,
    queues_and_events,
):
    stop_event = queues_and_events["stop_event"]
    should_listen = queues_and_events["should_listen"]
    recv_audio_chunks_queue = queues_and_events["recv_audio_chunks_queue"]
    send_audio_chunks_queue = queues_and_events["send_audio_chunks_queue"]
    spoken_prompt_queue = queues_and_events["spoken_prompt_queue"]
    text_prompt_queue = queues_and_events["text_prompt_queue"]
    lm_response_queue = queues_and_events["lm_response_queue"]
    if module_kwargs.mode == "local":
        from connections.local_audio_streamer import LocalAudioStreamer

        local_audio_streamer = LocalAudioStreamer(
            input_queue=recv_audio_chunks_queue, output_queue=send_audio_chunks_queue, sounddevice_device=module_kwargs.sounddevice_device
        )
        comms_handlers = [local_audio_streamer]
        should_listen.set()
    else:
        from connections.socket_receiver import SocketReceiver
        from connections.socket_sender import SocketSender

        comms_handlers = [
            SocketReceiver(
                stop_event,
                recv_audio_chunks_queue,
                should_listen,
                host=socket_receiver_kwargs.recv_host,
                port=socket_receiver_kwargs.recv_port,
                chunk_size=socket_receiver_kwargs.chunk_size,
            ),
            SocketSender(
                stop_event,
                send_audio_chunks_queue,
                host=socket_sender_kwargs.send_host,
                port=socket_sender_kwargs.send_port,
            ),
        ]

    vad = VADHandler(
        stop_event,
        input_queue=recv_audio_chunks_queue,
        output_queue=spoken_prompt_queue,
        setup_args=(should_listen,),
        setup_kwargs=vars(vad_handler_kwargs),
    )

    stt = get_stt_handler(
        module_kwargs,
        stop_event,
        spoken_prompt_queue,
        text_prompt_queue,
        whisper_stt_handler_kwargs,
    )
    lm = get_llm_handler(
        module_kwargs,
        stop_event,
        text_prompt_queue,
        lm_response_queue,
        language_model_handler_kwargs,
        open_api_language_model_handler_kwargs,
    )
    tts = get_tts_handler(
        module_kwargs,
        stop_event,
        lm_response_queue,
        send_audio_chunks_queue,
        should_listen,
        kokoro_tts_handler_kwargs,
        open_api_tts_handler_kwargs,
    )
    return ThreadManager([*comms_handlers, vad, stt, lm, tts])


def get_stt_handler(
    module_kwargs,
    stop_event,
    spoken_prompt_queue,
    text_prompt_queue,
    whisper_stt_handler_kwargs,
):
    if module_kwargs.stt == "whisper":
        from src.stt.whisper_stt_handler import WhisperSTTHandler
        return WhisperSTTHandler(
            stop_event,
            input_queue=spoken_prompt_queue,
            output_queue=text_prompt_queue,
            setup_kwargs=vars(whisper_stt_handler_kwargs),
        )
    else:
        raise ValueError(
            "The STT should be either whisper, whisper-mlx, or paraformer."
        )


def get_llm_handler(
    module_kwargs,
    stop_event,
    text_prompt_queue,
    lm_response_queue,
    language_model_handler_kwargs,
    open_api_language_model_handler_kwargs,
):
    if module_kwargs.llm == "transformers":
        from src.llm.language_model import LanguageModelHandler

        return LanguageModelHandler(
            stop_event,
            input_queue=text_prompt_queue,
            output_queue=lm_response_queue,
            setup_kwargs=vars(language_model_handler_kwargs),
        )
    elif module_kwargs.llm == "open_api":
        from src.llm.openai_api_language_model import OpenApiModelHandler

        return OpenApiModelHandler(
            stop_event,
            input_queue=text_prompt_queue,
            output_queue=lm_response_queue,
            setup_kwargs=vars(open_api_language_model_handler_kwargs),
        )

    else:
        raise ValueError("The LLM should be either transformers or open_api")


def get_tts_handler(
    module_kwargs,
    stop_event,
    lm_response_queue,
    send_audio_chunks_queue,
    should_listen,
    kokoro_tts_handler_kwargs,
    open_api_tts_handler_kwargs,
):
    if module_kwargs.tts == "kokoro":
        from src.tts.kokoro_handler import KokoroTTSHandler

        logger.info(f"prepared args: {vars(kokoro_tts_handler_kwargs)}")
        return KokoroTTSHandler(
            stop_event,
            input_queue=lm_response_queue,
            output_queue=send_audio_chunks_queue,
            setup_args=(should_listen,),
            setup_kwargs=vars(kokoro_tts_handler_kwargs),
        )
    elif module_kwargs.tts == "open_api":
        from src.tts.openai_api_tts import OpenAITTSHandler

        return OpenAITTSHandler(
            stop_event,
            input_queue=lm_response_queue,
            output_queue=send_audio_chunks_queue,
            setup_kwargs=vars(open_api_tts_handler_kwargs),
        )
    else:
        raise ValueError(
            "The TTS should be kokoro or open_api"
        )


def main():
    (
        module_kwargs,
        socket_receiver_kwargs,
        socket_sender_kwargs,
        whisper_stt_handler_kwargs,
        language_model_handler_kwargs,
        open_api_language_model_handler_kwargs,
        kokoro_tts_handler_kwargs,
        open_api_tts_handler_kwargs,
        vad_handler_kwargs,
    ) = parse_args()

    setup_global_logging(module_kwargs.log_level)

    prepare_all_args(
        module_kwargs,
        whisper_stt_handler_kwargs,
        language_model_handler_kwargs,
        open_api_language_model_handler_kwargs,
        kokoro_tts_handler_kwargs,
        open_api_tts_handler_kwargs,
        vad_handler_kwargs,
    )

    logger.debug(f"Starting pipeline with args: {module_kwargs}")

    queues_and_events = initialize_queues_and_events()

    pipeline_manager = build_pipeline(
        module_kwargs,
        socket_receiver_kwargs,
        socket_sender_kwargs,
        vad_handler_kwargs,
        whisper_stt_handler_kwargs,
        language_model_handler_kwargs,
        open_api_language_model_handler_kwargs,
        kokoro_tts_handler_kwargs,
        open_api_tts_handler_kwargs,
        queues_and_events,
    )

    try:
        pipeline_manager.start()
        pipeline_manager.join()
    except KeyboardInterrupt:
        pipeline_manager.stop()


if __name__ == "__main__":
    main()
