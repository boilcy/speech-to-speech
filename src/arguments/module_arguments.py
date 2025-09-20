from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModuleArguments:
    device: Optional[str] = field(
        default=None,
        metadata={"help": "If specified, overrides the device for all handlers."},
    )
    mode: Optional[str] = field(
        default="socket",
        metadata={
            "help": "The mode to run the pipeline in. Either 'local' or 'socket'. Default is 'socket'."
        },
    )
    stt: Optional[str] = field(
        default="whisper",
        metadata={
            "help": "The STT to use. Either 'whisper', 'whisper-mlx', 'faster-whisper', and 'paraformer'. Default is 'whisper'."
        },
    )
    llm: Optional[str] = field(
        default="transformers",
        metadata={
            "help": "The LLM to use. Either 'transformers' or 'mlx-lm'. Default is 'transformers'"
        },
    )
    tts: Optional[str] = field(
        default="parler",
        metadata={
            "help": "The TTS to use. Either 'parler', 'melo', 'chatTTS' or 'facebookMMS'. Default is 'parler'"
        },
    )
    log_level: str = field(
        default="info",
        metadata={
            "help": "Provide logging level. Example --log_level debug, default=info."
        },
    )
    sounddevice_device: Optional[str] = field(
        default=None,
        metadata={
            "help": "The sounddevice device to use. Default is None. Example --sounddevice_device 0,25."
        },
    )
